import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import re
import io
import json
import math
import tqdm
import torch
import numpy as np
from PIL import Image
import soundfile as sf
from pathlib import Path
from transformers import pipeline
from typing import List, Dict, Any, Tuple, Optional, Set

try:
    from moviepy import VideoFileClip  # MoviePy 2.x
except Exception:
    from moviepy.editor import VideoFileClip  # MoviePy 1.x 兼容

# ================= Gemini SDK =================
try:
    from google import genai
    from google.genai import types
except Exception as e:
    raise RuntimeError("google-genai SDK is required. Please `pip install google-genai`.") from e

# ================= 配置 =================
BASE_DIR: str = "./teleego_data"
P_NUM: str = os.getenv("P_NUM", "0")
QA_SUFFIX: str = os.getenv("QA_SUFFIX", "A")
VIDEO_PATH: str = f"{BASE_DIR}/video_merged/merged_P{P_NUM}.mp4"
JSON_PATH: str  = f"{BASE_DIR}/QAs/merged_P{P_NUM}_{QA_SUFFIX}.json"
SAVE_PRED_PATH: str = f"{BASE_DIR}/outputs/gemini25_pro/p{P_NUM}/eval_predictions.json"
TIMELINE_JSON_PATH: str = f"{BASE_DIR}/video_merged/timeline_P{P_NUM}.json"

print(f"[INFO] Using VIDEO_PATH: {VIDEO_PATH}")
print(f"[INFO] Saving to SAVE_PRED_PATH: {SAVE_PRED_PATH}")

GEMINI_MODEL_NAME = os.getenv("GEMINI_MODEL_NAME", "gemini-2.5-pro")
TEMPERATURE = 0
TOP_P = 0.9
MAX_NEW_TOKENS = 1024   # 对应 SDK 的 max_output_tokens
MIN_NEW_TOKENS = 1      # Gemini 无 min_new_tokens；此处用于一致性（不直接生效）
SEED = 42

ENABLE_RETRY = True
RETRY_TEMPERATURE = 0.3
RETRY_TOP_P = 0.9

RECALL_DELAY_SEC = 60.0
MAX_RECALL_ROUNDS = 10

USE_ASR = True
ASR_MODEL_NAME = "./weights/whisper-medium"
ASR_LANGUAGE   = "zh"
ASR_DEVICE     = 0 if torch.cuda.is_available() else -1

SAVE_UNIT_WAV = True
OUTPUT_AUDIO_BASENAME = "unit.wav"

PREFER_SIMPLE_VISION = True

API_KEY = os.environ.get("AZURE_OPENAI_API_KEY", "")
API_VERSION = os.environ.get("AZURE_OPENAI_API_VERSION", "2024-08-01-preview")
END_POINT = os.environ.get("AZURE_OPENAI_ENDPOINT", "")
ENGINE = os.environ.get("AZURE_OPENAI_DEPLOYMENT", "4o")

LLM_EVAL_CONFIG = {
    "enabled": bool(API_KEY),
    "provider": "azure",
    "api_key": API_KEY,
    "azure_endpoint": END_POINT,
    "azure_api_version": API_VERSION,
    "azure_deployment": ENGINE,
    "temperature": 0.0,
    "timeout": 30,
    "prompt": "open_ended_cn_v1",
}

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

def set_global_seed(seed: int = 42):
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import random as _random; _random.seed(seed)
    except Exception: pass
    try:
        np.random.seed(seed)
    except Exception: pass
    try:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except Exception: pass

def parse_hhmmss(ts: str) -> float:
    ts = ts.strip()
    parts = ts.split(":")
    if len(parts) == 3:
        h, m, s = parts
    elif len(parts) == 2:
        h, m, s = "0", parts[0], parts[1]
    else:
        raise ValueError(f"Bad timestamp: {ts}")
    return int(h) * 3600 + int(m) * 60 + float(s)

class TimelineIndex:
    def __init__(self, segments, day_prefix="D"):
        self.segments = sorted(segments, key=lambda s: s["merged_offset_start_seconds"])
        self.day_prefix = day_prefix

    @staticmethod
    def _parse_day_label(label: str, day_prefix: str = "D") -> Optional[int]:
        if not isinstance(label, str):
            return None
        s = label.strip()
        s = (s.replace("：", ":")
               .replace("—", "-")
               .replace("–", "-")
               .replace("-", "-"))

        m = re.match(
            r"^(?:{dp}|d|day)\s*(\d+)[\-\s]?(\d{{1,2}}:\d{{2}}(?::\d{{2}})?)$".format(
                dp=re.escape(day_prefix)
            ),
            s,
            flags=re.IGNORECASE
        )
        if not m:
            return None

        day = int(m.group(1))
        hhmmss = m.group(2)
        parts = hhmmss.split(":")
        if len(parts) == 2:
            h, m1 = int(parts[0]), int(parts[1]); sec = 0
        else:
            h, m1, sec = int(parts[0]), int(parts[1]), int(parts[2])
        return (day - 1) * 86400 + h * 3600 + m1 * 60 + sec

    @classmethod
    def from_json(cls, path: str):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        day_prefix = (data.get("day_prefix_used") or "D")
        segs = []
        for vname, meta in (data.get("mapping_by_input_label") or {}).items():
            st_lab = meta.get("start_label"); en_lab = meta.get("end_label")
            st_abs = cls._parse_day_label(st_lab, day_prefix)
            en_abs = cls._parse_day_label(en_lab, day_prefix)
            if st_abs is None or en_abs is None: continue
            segs.append({
                "video": vname,
                "start_label": st_lab,
                "end_label": en_lab,
                "start_abs": st_abs,
                "end_abs": en_abs,
                "merged_offset_start_seconds": float(meta.get("merged_offset_start_seconds", 0)),
                "merged_offset_end_seconds": float(meta.get("merged_offset_end_seconds", 0)),
                "description": meta.get("description", ""),
            })
        return cls(segs, day_prefix=day_prefix)

    def label_to_merged_seconds(self, label: Any) -> Optional[float]:
        if label is None: return None
        if isinstance(label, (int, float)): return float(label)
        s = str(label).strip()
        abs_s = self._parse_day_label(s, self.day_prefix)
        if abs_s is not None:
            for seg in self.segments:
                if seg["start_abs"] <= abs_s <= seg["end_abs"]:
                    return seg["merged_offset_start_seconds"] + (abs_s - seg["start_abs"])
            return None
        try:
            if ":" in s: return parse_hhmmss(s)
        except Exception: pass
        return None

    def find_segment_for_offset(self, merged_sec: float) -> Optional[Dict[str, Any]]:
        for seg in self.segments:
            if seg["merged_offset_start_seconds"] <= merged_sec <= seg["merged_offset_end_seconds"]:
                return seg
        return None

def open_video_clip(path: str) -> "VideoFileClip":
    clip = VideoFileClip(path)
    safe_dur = max(0.0, float(clip.duration) - 0.05)
    if hasattr(clip, "subclipped"):
        return clip.subclipped(0, safe_dur)  # MoviePy 2.x
    else:
        return clip.subclip(0, safe_dur)     # MoviePy 1.x

def get_image_at_time(video: "VideoFileClip", t_sec: float) -> Image.Image:
    t = min(max(0.0, float(t_sec)), max(0.0, float(video.duration) - 1e-3))
    frame = video.get_frame(t)
    return Image.fromarray(frame.astype(np.uint8)).convert("RGB")

def get_audio_1s_at_time(video: "VideoFileClip", t_start: float, sr: int = 16000) -> np.ndarray:
    t0 = max(0.0, float(t_start))
    t1 = min(float(video.duration), t0 + 1.0)
    if getattr(video, "audio", None) is None or t1 <= t0:
        return np.zeros(int(sr), dtype=np.float32)
    try:
        snd = video.audio.subclip(t0, t1).to_soundarray(fps=sr)  # [N, C]
        if snd.ndim == 2:
            snd = snd.mean(axis=1)
        if len(snd) < sr:
            pad = np.zeros(sr - len(snd), dtype=snd.dtype)
            snd = np.concatenate([snd, pad], axis=0)
        snd = np.clip(np.nan_to_num(snd, nan=0.0, posinf=0.0, neginf=0.0), -1.0, 1.0)
        return snd.astype(np.float32, copy=False)
    except Exception:
        return np.zeros(int(sr), dtype=np.float32)

def unit_index_for_time(t_end: float, total_units: int) -> int:
    if total_units <= 0:
        return 0
    if not math.isfinite(t_end):
        return total_units - 1
    eps = 1e-6
    idx = int(math.floor(max(0.0, float(t_end) - eps)))
    return max(0, min(idx, total_units - 1))

def build_question_prompt(item: Dict[str, Any]) -> str:
    qtype = (item.get("QA_type") or "").lower()
    question = (item.get("question") or "").strip()
    options: List[str] = item.get("options", []) if isinstance(item.get("options"), list) else []

    if qtype == "mc_single":
        opt_str = "\n".join(options)
        instr = (
            "请根据当前秒的视频画面与音频转写回答一个单选题。"
            "严格只输出一个大写字母（A/B/C/...），不含空格、不含标点、不含其他字符。"
        )
        return f"{instr}\n题目：{question}\n选项：\n{opt_str}\n你的答案："

    if qtype == "mc_multi":
        opt_str = "\n".join(options)
        instr = (
            "请根据当前秒的视频画面与音频转写回答一个多选题。"
            "严格只输出若干大写字母并用英文逗号连接（如 A,B），不含空格、不含其他字符。"
        )
        return f"{instr}\n题目：{question}\n选项：\n{opt_str}\n你的答案："

    if qtype == "binary":
        instr = (
            "请判断对错（True/False）。严格只输出 True 或 False（首字母大写），不含其他字符。"
        )
        return f"{instr}\n命题：{question}\n你的答案："

    instr = "请简要作答本题。优先给出关键词或短句，尽量在20个字以内，勿添加前后缀。"
    return f"{instr}\n问题：{question}\n你的答案："

def _extract_allowed_letters(options: List[str]) -> List[str]:
    if not options:
        return []
    cand: List[str] = []
    for opt in options:
        s = str(opt or "")
        m = re.match(r"\s*([A-Za-z])\s*[\.)、\)\:]?", s)
        if m:
            cand.append(m.group(1).upper())
    if cand:
        seen: Set[str] = set(); out=[]
        for c in cand:
            if c not in seen:
                seen.add(c); out.append(c)
        return out
    n = len(options)
    base = [chr(ord('A')+i) for i in range(min(26, n))]
    return base

def parse_prediction(text: str, qtype: str, options: List[str]):
    t = (text or "").strip()
    qtype = (qtype or "").lower()
    allowed = set(_extract_allowed_letters(options))

    if qtype == "mc_single":
        m = re.fullmatch(r"\s*([A-Za-z])\s*", t)
        if m:
            c = m.group(1).upper()
            return [c] if (not allowed or c in allowed) else []
        letters = [x for x in re.findall(r"[A-Za-z]", t)]
        for x in letters:
            u = x.upper()
            if not allowed or u in allowed:
                return [u]
        return []

    if qtype == "mc_multi":
        m = re.fullmatch(r"\s*([A-Za-z](?:\s*[,，]\s*[A-Za-z])*)\s*", t)
        out = []
        if m:
            out = re.findall(r"[A-Za-z]", m.group(1).upper())
        else:
            out = re.findall(r"[A-Za-z]", t.upper())
        seen: Set[str] = set(); filtered=[]
        for x in out:
            if (not allowed or x in allowed) and x not in seen:
                seen.add(x); filtered.append(x)
        return filtered

    if qtype == "binary":
        tl = t.strip().lower()
        if re.fullmatch(r"true", tl): return True
        if re.fullmatch(r"false", tl): return False
        truish = {"true", "t", "1", "y", "yes", "是", "对", "正确"}
        falsish = {"false", "f", "0", "n", "no", "否", "不对", "错误"}
        if tl in truish: return True
        if tl in falsish: return False
        if "true" in tl: return True
        if "false" in tl: return False
        return None

    return t

def is_invalid_output(qtype: str, raw_text: str, options: List[str]) -> bool:
    q = (qtype or "").lower()
    if q in {"mc_single", "mc_multi"}:
        allowed = set(_extract_allowed_letters(options))
        t = (raw_text or "").strip()
        if t == "":
            return True
        if q == "mc_single":
            m = re.fullmatch(r"\s*([A-Za-z])\s*", t)
            if not m:
                letters = [x.upper() for x in re.findall(r"[A-Za-z]", t)]
                for u in letters:
                    if (not allowed or u in allowed):
                        return False
                return True
            c = m.group(1).upper()
            return (bool(allowed) and c not in allowed)
        else:
            letters = [x.upper() for x in re.findall(r"[A-Za-z]", t)]
            if not letters:
                return True
            if not allowed:
                return False
            return not any(u in allowed for u in letters)

    if q == "binary":
        pred = parse_prediction(raw_text, q, [])
        return pred is None
    return False

def evaluate_item(gt: Dict[str, Any], pred) -> Dict[str, Any]:
    qtype = (gt.get("QA_type") or "").lower()
    ans = gt.get("answer", {}) or {}
    raw = ans.get("value", None)

    def _to_letters(x) -> list[str]:
        if isinstance(x, list):
            s = ",".join(map(str, x))
        else:
            s = "" if x is None else str(x)
        return re.findall(r"[A-Z]", s.upper())

    def _to_bool(x) -> bool:
        if isinstance(x, bool): return x
        if isinstance(x, (int, float)): return bool(x)
        if isinstance(x, list) and x: return _to_bool(x[0])
        if isinstance(x, str):
            low = x.strip().lower()
            truish = {"true","t","1","y","yes","是","对","正确"}
            falsish = {"false","f","0","n","no","否","不对","错误"}
            if low in truish: return True
            if low in falsish: return False
        return False

    result = {"correct": False, "metric": ""}

    if qtype in {"mc_single", "mc_multi"}:
        gt_letters = _to_letters(raw)
        if qtype == "mc_single":
            pred_letters = [str(x).upper() for x in (pred or [])]
            result["correct"] = (len(pred_letters) == 1 and pred_letters[0] in gt_letters)
            result["metric"] = "accuracy"
            return result
        pred_set = set([str(x).upper() for x in (pred or [])])
        gt_set = set(gt_letters)
        result["correct"] = (len(gt_set) > 0 and pred_set == gt_set)
        result["metric"] = "exact_set_match"
        return result

    if qtype == "binary":
        gt_bool = _to_bool(raw)
        result["correct"] = (pred is not None and bool(pred) == gt_bool)
        result["metric"] = "accuracy"
        return result

    if isinstance(raw, list) and raw:
        gt_text = str(raw[0])
    elif isinstance(raw, (str, int, float, bool)):
        gt_text = str(raw)
    else:
        gt_text = ""
    pred_text = (pred or "").strip().lower()
    gt_text_l = gt_text.strip().lower()

    def tokens(s: str):
        return [w for w in re.findall(r"[一-龥A-Za-z0-9]+", s) if w]

    gts = tokens(gt_text_l)
    prs = set(tokens(pred_text))
    overlap = [w for w in gts if w in prs]
    result["correct"] = len(overlap) >= max(1, len(gts) // 4)
    result["metric"] = "token_overlap>=25%"
    result["overlap_tokens"] = overlap
    return result

# -----------------------------
# LLM judge prompt + caller (for open_ended)
# -----------------------------
prompt_dict = {
    "open_ended_cn_v1": (
        "你是一个客观判分器。请只输出一个 JSON 对象（严格），不要包含多余文本或解释。\n"
        "给定：\n"
        "【问题】{question}\n"
        "【参考答案】{ground_truth}\n"
        "【模型回答】{prediction}\n\n"
        "请根据模型回答与参考答案在事实与要点上的一致程度打分，分值为 1～5 的整数：\n"
        "5 = 完全正确；4 = 基本正确；3 = 部分正确；2 = 大多不正确；1 = 完全错误。\n\n"
        "只输出以下 JSON（严格）：例如 {{\"score\": 4}}\n"
    ),
}

def evaluate_with_llm(question: str, ground_truth: str, prediction: str, llm_config: Dict = None) -> Dict[str, Any]:
    import json as _json
    try:
        from openai import AzureOpenAI
    except Exception:
        AzureOpenAI = None

    cfg = {
        "enabled": False,
        "provider": "azure",
        "api_key": LLM_EVAL_CONFIG["api_key"],
        "azure_endpoint": LLM_EVAL_CONFIG["azure_endpoint"],
        "azure_api_version": LLM_EVAL_CONFIG["azure_api_version"],
        "azure_deployment": LLM_EVAL_CONFIG["azure_deployment"],
        "temperature": 0.0,
        "timeout": 30,
        "prompt": "open_ended_cn_v1",
    }
    if llm_config:
        cfg.update({k: v for k, v in llm_config.items() if v is not None})

    if not cfg.get("enabled"): return {"llm_score": 0}
    try:
        prompt = prompt_dict[cfg["prompt"]].format(
            question=str(question or ""),
            ground_truth=str(ground_truth or ""),
            prediction=str(prediction or "")
        )
        if cfg["provider"] == "azure" and AzureOpenAI is not None:
            client = AzureOpenAI(
                api_key=cfg["api_key"],
                api_version=cfg["azure_api_version"],
                azure_endpoint=cfg["azure_endpoint"],
            )
            resp = client.chat.completions.create(
                model=cfg["azure_deployment"],
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=float(cfg.get("temperature", 0.0) or 0.0),
                timeout=cfg.get("timeout", 30),
            )
            data = _json.loads(resp.choices[0].message.content)
            score = int(str(data.get("score", 0)).strip())
            return {"llm_score": score if 1 <= score <= 5 else 0}
    except Exception:
        print("⚠️ LLM evaluation error.")
    return {"llm_score": 0}

_ASR = None
def init_asr():
    global _ASR
    if _ASR is None and USE_ASR:
        dtype = torch.float16 if (torch.cuda.is_available() and ASR_DEVICE != -1) else None
        _ASR = pipeline(
            "automatic-speech-recognition",
            model=ASR_MODEL_NAME,
            device=ASR_DEVICE,
            torch_dtype=dtype,   # CPU 下必须 None
        )
        # ★ 关键：清空 forced_decoder_ids，避免与 task/lang 冲突
        try:
            if hasattr(_ASR.model, "generation_config"):
                _ASR.model.generation_config.forced_decoder_ids = None
            if hasattr(_ASR.model, "config"):
                _ASR.model.config.forced_decoder_ids = None
        except Exception as e:
            print(f"[ASR] forced_decoder_ids reset failed: {e}")
    return _ASR

from transformers import WhisperProcessor, WhisperForConditionalGeneration

processor = WhisperProcessor.from_pretrained(ASR_MODEL_NAME)
model = WhisperForConditionalGeneration.from_pretrained(ASR_MODEL_NAME).to(
    "cuda" if torch.cuda.is_available() and ASR_DEVICE != -1 else "cpu"
)
model.eval()

processor = WhisperProcessor.from_pretrained(ASR_MODEL_NAME)
model = WhisperForConditionalGeneration.from_pretrained(ASR_MODEL_NAME).to(
    "cuda" if torch.cuda.is_available() and ASR_DEVICE != -1 else "cpu"
)
model.eval()

def transcribe_audio_1s(audio_np: np.ndarray, sr: int = 16000) -> str:
    if audio_np is None or len(audio_np) == 0:
        return ""
    audio = np.asarray(audio_np, dtype=np.float32)
    audio = np.clip(np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0), -1.0, 1.0)

    # 关键：要 attention_mask（batch 场景尤其重要）
    inputs = processor(
        audio, sampling_rate=sr,
        return_tensors="pt",
        return_attention_mask=True,
        padding="longest",   # 不截断，便于 mask 正确
        truncation=False
    )
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    forced_ids = processor.get_decoder_prompt_ids(language="zh", task="transcribe")

    with torch.no_grad():
        gen_ids = model.generate(
            input_features=inputs["input_features"],           # ✅ 新接口名（非 inputs）
            attention_mask=inputs.get("attention_mask", None), # ✅ 显式传入
            forced_decoder_ids=forced_ids,
            do_sample=False, max_new_tokens=128
        )
    text = processor.batch_decode(gen_ids, skip_special_tokens=True)[0].strip()
    return text[:160] + "…" if len(text) > 160 else text

_GEMINI_CLIENT: Optional["genai.Client"] = None
def get_gemini_client() -> "genai.Client":
    global _GEMINI_CLIENT
    if _GEMINI_CLIENT is None:
        # API Key 将自动从 GEMINI_API_KEY 或 GOOGLE_API_KEY 环境变量读取
        _GEMINI_CLIENT = genai.Client()
    return _GEMINI_CLIENT

def image_to_part(image: Optional[Image.Image]) -> Optional[types.Part]:
    if image is None:
        return None
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return types.Part.from_bytes(data=buf.getvalue(), mime_type="image/png")

def audio_to_part_1s(audio_1s: Optional[np.ndarray], sr: int = 16000) -> Optional[types.Part]:
    if audio_1s is None:
        return None
    try:
        buf = io.BytesIO()
        sf.write(buf, np.asarray(audio_1s, dtype=np.float32), samplerate=sr, format="WAV", subtype="PCM_16")
        wav_bytes = buf.getvalue()
        return types.Part.from_bytes(data=wav_bytes, mime_type="audio/wav")
    except Exception as e:
        print(f"[Audio encode] error: {e}")
        return None

BASE_SYS_TEXT = "你是一个多模态助手。根据当前秒的视频画面与1秒音频转写，严格按题目要求简洁作答。"

def run_streaming_round(
    client: "genai.Client",
    sys_text: str,
    image: Optional[Image.Image],
    audio_1s: Optional[np.ndarray],
    audio_sr: int,
    prompt_text: str,
    *,
    phase: str,
    item_index: Any,
    temperature: float = TEMPERATURE,
    top_p: float = TOP_P,
    max_new_tokens: int = MAX_NEW_TOKENS,
    min_new_tokens: int = MIN_NEW_TOKENS,  # Gemini 无 min，参数留存以对齐接口
    save_unit_wav: bool = SAVE_UNIT_WAV,
    audio_base_path: Optional[str] = None,
    round_id: int = 0,
    prefer_simple_vision: bool = PREFER_SIMPLE_VISION,  # 保留占位
) -> Tuple[str, Optional[str]]:
    # 1) 保存音频
    saved_audio_path = None
    if save_unit_wav and audio_1s is not None and audio_base_path:
        try:
            base, ext = os.path.splitext(audio_base_path or "unit.wav")
            uniq = f"{base}_{phase}_q{item_index}" + (f"_r{round_id}" if round_id else "") + (ext or ".wav")
            sf.write(uniq, np.asarray(audio_1s, dtype=np.float32), samplerate=audio_sr)
            saved_audio_path = uniq
        except Exception as e:
            print(f"[Audio save] error: {e}")

    # 2) ASR
    transcript = transcribe_audio_1s(audio_1s, sr=audio_sr) if audio_1s is not None else ""

    # 3) 构造消息（Gemini：用 Part 表达多模态输入）
    user_text = prompt_text if not transcript else f"{prompt_text}\n\n[Audio transcript]: {transcript}"
    parts = []
    img_part = image_to_part(image)
    if img_part is not None:
        parts.append(img_part)
    # ❗ 修复：避免 Part.from_text() 的已知问题，改用字段构造 Part(text=...)
    parts.append(types.Part(text=user_text))

    aud_part = audio_to_part_1s(audio_1s, sr=audio_sr)
    if aud_part is not None:
        parts.append(aud_part)

    contents = [
        types.Content(role="user", parts=parts)
    ]

    # 4) 生成配置
    gen_cfg = types.GenerateContentConfig(
        system_instruction=sys_text or BASE_SYS_TEXT,
        temperature=float(temperature or 0.0),
        top_p=float(top_p or 0.9),
        max_output_tokens=int(max_new_tokens or 256),
    )

    # 5) 流式生成
    print_prefix = f"[{phase} q={item_index}] "
    printed_prefix = False
    out_parts: List[str] = []

    stream = None
    try:
        # 新版 SDK
        stream = client.models.generate_content_stream(
            model=GEMINI_MODEL_NAME,
            contents=contents,
            config=gen_cfg,
        )
    except Exception:
        # 兼容老式：generate_content(stream=True)
        stream = client.models.generate_content(
            model=GEMINI_MODEL_NAME,
            contents=contents,
            config=gen_cfg,
            stream=True,
        )

    for chunk in stream:
        text_piece = getattr(chunk, "text", None)
        if not text_piece:
            # 某些版本将增量文本放在 chunk.candidates[0].content.parts[...] 中
            try:
                if getattr(chunk, "candidates", None):
                    for c in chunk.candidates:
                        for p in getattr(c.content, "parts", []) or []:
                            if getattr(p, "text", None):
                                text_piece = p.text
                                break
                        if text_piece:
                            break
            except Exception:
                text_piece = None

        if text_piece:
            if not printed_prefix:
                print(print_prefix, end="", flush=True)
                printed_prefix = True
            out_parts.append(text_piece)
            print(text_piece, end="", flush=True)

    # 某些 SDK 需要 resolve() 才能补全最终响应
    try:
        if hasattr(stream, "resolve"):
            stream.resolve()
    except Exception:
        pass

    print()
    return "".join(out_parts).strip(), saved_audio_path

# =============== QA 调度（按需取帧/取音频 + retry） ===============
def collect_questionstamps(qa_items: List[Dict[str, Any]], label_mapper: Optional[Any] = None) -> List[Tuple[float, List[Dict[str, Any]]]]:
    rows: List[Tuple[float, int, Dict[str, Any]]] = []
    for pos, it in enumerate(qa_items):
        ts = (it.get("evidence") or {}).get("timestep") or {}
        end_raw = ts.get("end") or ts.get("End") or ts.get("to")
        t_end = None
        if label_mapper is not None:
            try:
                t_end = label_mapper(end_raw)
            except Exception:
                t_end = None
        if t_end is None:
            try:
                if end_raw is None:
                    t_end = float("inf")
                elif isinstance(end_raw, (int, float)):
                    t_end = float(end_raw)
                else:
                    s = str(end_raw).strip()
                    t_end = parse_hhmmss(s) if ":" in s else float(s)
            except Exception:
                t_end = float("inf")
        rows.append((t_end, pos, it))
    rows.sort(key=lambda r: (r[0], r[1]))
    grouped: List[Tuple[float, List[Dict[str, Any]]]] = []
    for t_end, _, it in rows:
        grouped.append((t_end, [it]))
    return grouped

def evaluate_single(video_path: str, json_path: str, save_dir: str, timeline_path: Optional[str] = None) -> Dict[str, Any]:
    set_global_seed(SEED)
    assert os.path.exists(json_path), f"QA json not found: {json_path}"
    assert os.path.exists(video_path), f"Video not found: {video_path}"

    timeline_idx = None
    if timeline_path and os.path.exists(timeline_path):
        try:
            timeline_idx = TimelineIndex.from_json(timeline_path)
        except Exception as e:
            print(f"⚠️ Failed to load timeline: {timeline_path} -> {e}")

    with open(json_path, "r", encoding="utf-8") as f:
        qa_items: List[Dict[str, Any]] = json.load(f)

    groups = collect_questionstamps(qa_items, label_mapper=(timeline_idx.label_to_merged_seconds if timeline_idx else None))
    max_t = 0.0 if not groups else max([t for t, _ in groups if math.isfinite(t)] + [0.0])

    # 打开视频一次；每题“按需取帧+音频 1s”
    video = open_video_clip(video_path)
    total_seconds = int(math.ceil(video.duration))
    _ = int(math.ceil(max_t)) if math.isfinite(max_t) else total_seconds

    os.makedirs(save_dir, exist_ok=True)
    audio_base = str(Path(save_dir) / OUTPUT_AUDIO_BASENAME)

    client = get_gemini_client()

    results = []
    type_stats: Dict[str, Dict[str, int]] = {}
    category_stats: Dict[str, Dict[str, int]] = {}
    cat_sub_stats: Dict[Tuple[str, str], Dict[str, int]] = {}
    recall_round_stats: Dict[int, Dict[str, Any]] = {}

    correct_for_recall: List[Tuple[float, Dict[str, Any]]] = []
    announced_videos = set()

    try:
        # 初始阶段
        for t_end, items in tqdm.tqdm(groups, desc="QA by questionstamp"):
            idx = unit_index_for_time(t_end, total_seconds)

            unit_image = get_image_at_time(video, t_sec=idx + 0.5)
            unit_audio = get_audio_1s_at_time(video, t_start=idx, sr=16000)

            # 段首信息（仅首次出现）
            sys_text_local = BASE_SYS_TEXT
            if timeline_idx and math.isfinite(t_end):
                seg = timeline_idx.find_segment_for_offset(t_end)
                if seg and seg["video"] not in announced_videos:
                    intro_txt = (
                        f"【当前子视频段信息】起始时间：{seg['start_label']}；"
                        f"人物/场景描述：{seg.get('description','') or '（无）'}；"
                        f"该段在合并视频中的起点：{int(seg['merged_offset_start_seconds'])}s。"
                    )
                    sys_text_local = BASE_SYS_TEXT + "\n" + intro_txt
                    announced_videos.add(seg["video"])

            for item in items:
                qtype = (item.get("QA_type") or "").lower()
                prompt = build_question_prompt(item)
                options = item.get("options", []) if isinstance(item.get("options"), list) else []

                # ---------- 首轮 ----------
                try:
                    pred_text, saved_audio = run_streaming_round(
                        client=client,
                        sys_text=sys_text_local,
                        image=unit_image,
                        audio_1s=unit_audio,
                        audio_sr=16000,
                        prompt_text=prompt,
                        phase="initial",
                        item_index=item.get("index", "na"),
                        temperature=TEMPERATURE,
                        top_p=TOP_P,
                        max_new_tokens=MAX_NEW_TOKENS,
                        min_new_tokens=MIN_NEW_TOKENS,
                        save_unit_wav=SAVE_UNIT_WAV,
                        audio_base_path=audio_base,
                        prefer_simple_vision=PREFER_SIMPLE_VISION,
                    )
                except Exception as e:
                    pred_text, saved_audio = f"<ERROR: {e}>", None

                # ---------- 如不合法，重试一次（小温度） ----------
                if ENABLE_RETRY and is_invalid_output(qtype, pred_text, options):
                    retry_prompt = prompt + "\n【只输出答案，不要输出任何其他字符】"
                    try:
                        pred_text_retry, saved_audio2 = run_streaming_round(
                            client=client,
                            sys_text=sys_text_local,
                            image=unit_image,
                            audio_1s=unit_audio,
                            audio_sr=16000,
                            prompt_text=retry_prompt,
                            phase="retry",
                            item_index=item.get("index", "na"),
                            temperature=RETRY_TEMPERATURE,
                            top_p=RETRY_TOP_P,
                            max_new_tokens=MAX_NEW_TOKENS,
                            min_new_tokens=max(1, MIN_NEW_TOKENS),
                            save_unit_wav=SAVE_UNIT_WAV,
                            audio_base_path=audio_base,
                            prefer_simple_vision=PREFER_SIMPLE_VISION,
                        )
                        if not is_invalid_output(qtype, pred_text_retry, options):
                            pred_text = pred_text_retry
                            if saved_audio2:
                                saved_audio = saved_audio2
                    except Exception as e:
                        print(f"[WARN] retry failed: {e}")

                parsed = parse_prediction(pred_text, qtype, options)
                eval_res = evaluate_item(item, parsed)

                llm_score = 0
                if qtype not in {"mc_single", "mc_multi", "binary"} and LLM_EVAL_CONFIG.get("enabled", False):
                    gt_vals = (item.get("answer", {}) or {}).get("value", []) or []
                    ground_truth = "" if not gt_vals else str(gt_vals[0])
                    llm_out = evaluate_with_llm(
                        question=item.get("question", ""),
                        ground_truth=ground_truth,
                        prediction=pred_text,
                        llm_config=LLM_EVAL_CONFIG
                    )
                    llm_score = int(llm_out.get("llm_score", 0))

                rec = {
                    "phase": "initial",
                    "index": item.get("index"),
                    "category": item.get("category"),
                    "subcategory": item.get("subcategory"),
                    "QA_type": item.get("QA_type"),
                    "question": item.get("question"),
                    "options": options,
                    "gold": (item.get("answer", {}) or {}).get("value", []) or [],
                    "questionstamp": None if not math.isfinite(t_end) else t_end,
                    "pred_text": pred_text,
                    "parsed_pred": parsed,
                    "correct": bool(eval_res.get("correct", False)),
                    "metric": eval_res.get("metric", ""),
                    "llm_score": llm_score if qtype not in {"mc_single","mc_multi","binary"} else None,
                }
                if saved_audio:
                    rec["audio_path"] = saved_audio
                if "overlap_tokens" in eval_res:
                    rec["overlap_tokens"] = eval_res["overlap_tokens"]
                results.append(rec)

                # 统计（选择/判断计入 acc）
                tkey = (rec.get("QA_type") or "unknown").lower()
                s = type_stats.setdefault(tkey, {"n": 0, "ok": 0})
                s["n"] += 1
                if rec["metric"] != "token_overlap>=25%":
                    s["ok"] += int(rec["correct"])
                    cat = str(rec.get("category") or "unknown")
                    sub = str(rec.get("subcategory") or "unknown")
                    s1 = category_stats.setdefault(cat, {"n": 0, "ok": 0})
                    s1["n"] += 1; s1["ok"] += int(rec["correct"])
                    s2 = cat_sub_stats.setdefault((cat, sub), {"n": 0, "ok": 0})
                    s2["n"] += 1; s2["ok"] += int(rec["correct"])

                if rec["correct"] and math.isfinite(t_end):
                    correct_for_recall.append((t_end, item))

                # Logging
                if tkey in {"mc_single", "mc_multi", "binary"}:
                    print(f"[t={t_end:.2f}s][#{rec['index']}] {tkey} | correct={rec['correct']} | pred={rec['parsed_pred']} | raw={rec['pred_text'][:120]!r}")
                else:
                    score_str = "N/A" if rec.get("llm_score") is None else str(rec.get("llm_score"))
                    print(f"[t={t_end:.2f}s][#{rec['index']}] {tkey} | score={score_str} | raw={rec['pred_text'][:120]!r}")

        # 初始阶段汇总
        print("===== Summary (Initial) =====")
        denom = sum(v["n"] for k, v in type_stats.items() if k in {"mc_single", "mc_multi", "binary"})
        acc_n = sum(v["ok"] for k, v in type_stats.items() if k in {"mc_single", "mc_multi", "binary"})
        overall = (acc_n / max(1, denom)) if denom else 0.0
        for k, v in type_stats.items():
            if k in {"mc_single", "mc_multi", "binary"}:
                acc = v["ok"] / max(1, v["n"]) if v["n"] else 0.0
                print(f"  {k}: {v['ok']}/{v['n']} = {acc:.3f}")
            else:
                print(f"  {k}: {v['n']} samples (text metric)")
        print(f"Overall (choice+binary only): {acc_n}/{denom} = {overall:.3f}")

        print("===== Accuracy by category (Initial, choice+binary only) =====")
        for k, v in category_stats.items():
            acc = v["ok"] / max(1, v["n"]) if v["n"] else 0.0
            print(f"  {k}: {v['ok']}/{v['n']} = {acc:.3f}")

        print("===== Accuracy by (category, subcategory) (Initial, choice+binary only) =====")
        for (ck, sk), v in cat_sub_stats.items():
            acc = v["ok"] / max(1, v["n"]) if v["n"] else 0.0
            print(f"  {ck} / {sk}: {v['ok']}/{v['n']} = {acc:.3f}")

        # 召回阶段
        if RECALL_DELAY_SEC > 0 and correct_for_recall:
            active: List[Tuple[float, Dict[str, Any]]] = list(correct_for_recall)
            round_id = 1
            while active and round_id <= MAX_RECALL_ROUNDS:
                rr_type_stats: Dict[str, Dict[str, int]] = {}
                rr_cat_stats: Dict[str, Dict[str, int]] = {}
                rr_cat_sub_stats: Dict[Tuple[str, str], Dict[str, int]] = {}

                events: List[Tuple[float, Dict[str, Any]]] = []
                for t_end, item in active:
                    recall_t = t_end + round_id * RECALL_DELAY_SEC
                    events.append((recall_t, item))
                events.sort(key=lambda x: x[0])

                kept_for_next: List[Tuple[float, Dict[str, Any]]] = []
                print(f"===== Recall round {round_id} (+{round_id*RECALL_DELAY_SEC}s) =====")
                for recall_t, item in tqdm.tqdm(events, desc=f"Recall r{round_id}"):
                    idx2 = unit_index_for_time(recall_t, total_seconds)

                    unit_image2 = get_image_at_time(video, t_sec=idx2 + 0.5)
                    unit_audio2 = get_audio_1s_at_time(video, t_start=idx2, sr=16000)

                    qtype = (item.get("QA_type") or "").lower()
                    prompt = build_question_prompt(item)
                    options = item.get("options", []) if isinstance(item.get("options"), list) else []

                    try:
                        pred2, saved_audio2 = run_streaming_round(
                            client=client,
                            sys_text=BASE_SYS_TEXT,
                            image=unit_image2,
                            audio_1s=unit_audio2,
                            audio_sr=16000,
                            prompt_text=prompt,
                            phase="recall",
                            item_index=item.get("index", "na"),
                            temperature=TEMPERATURE,
                            top_p=TOP_P,
                            max_new_tokens=MAX_NEW_TOKENS,
                            min_new_tokens=MIN_NEW_TOKENS,
                            save_unit_wav=SAVE_UNIT_WAV,
                            audio_base_path=audio_base,
                            round_id=round_id,
                            prefer_simple_vision=PREFER_SIMPLE_VISION,
                        )
                    except Exception as e:
                        pred2, saved_audio2 = f"<ERROR: {e}>", None

                    if ENABLE_RETRY and is_invalid_output(qtype, pred2, options):
                        retry_prompt = prompt + "\n【只输出答案，不要输出任何其他字符】"
                        try:
                            pred2_retry, _ = run_streaming_round(
                                client=client,
                                sys_text=BASE_SYS_TEXT,
                                image=unit_image2,
                                audio_1s=unit_audio2,
                                audio_sr=16000,
                                prompt_text=retry_prompt,
                                phase="recall-retry",
                                item_index=item.get("index", "na"),
                                temperature=RETRY_TEMPERATURE,
                                top_p=RETRY_TOP_P,
                                max_new_tokens=MAX_NEW_TOKENS,
                                min_new_tokens=max(1, MIN_NEW_TOKENS),
                                save_unit_wav=False,
                                audio_base_path=None,
                                round_id=round_id,
                                prefer_simple_vision=PREFER_SIMPLE_VISION,
                            )
                            if not is_invalid_output(qtype, pred2_retry, options):
                                pred2 = pred2_retry
                        except Exception as e:
                            print(f"[WARN] recall retry failed: {e}")

                    parsed2 = parse_prediction(pred2, qtype, options)
                    eval2 = evaluate_item(item, parsed2)

                    llm_score = 0
                    if qtype not in {"mc_single", "mc_multi", "binary"} and LLM_EVAL_CONFIG.get("enabled", False):
                        gt_vals = (item.get("answer", {}) or {}).get("value", []) or []
                        ground_truth = "" if not gt_vals else str(gt_vals[0])
                        llm_out = evaluate_with_llm(
                            question=item.get("question", ""),
                            ground_truth=ground_truth,
                            prediction=pred2,
                            llm_config=LLM_EVAL_CONFIG
                        )
                        llm_score = int(llm_out.get("llm_score", 0))

                    rec2 = {
                        "phase": "recall",
                        "round": round_id,
                        "index": item.get("index"),
                        "category": item.get("category"),
                        "subcategory": item.get("subcategory"),
                        "QA_type": item.get("QA_type"),
                        "question": item.get("question"),
                        "options": options,
                        "gold": (item.get("answer", {}) or {}).get("value", []) or [],
                        "questionstamp": recall_t,
                        "recall_delay": round_id * RECALL_DELAY_SEC,
                        "pred_text": pred2,
                        "parsed_pred": parsed2,
                        "correct": bool(eval2.get("correct", False)),
                        "metric": eval2.get("metric", ""),
                        "llm_score": llm_score if qtype not in {"mc_single","mc_multi","binary"} else None,
                    }
                    if saved_audio2:
                        rec2["audio_path"] = saved_audio2
                    if "overlap_tokens" in eval2:
                        rec2["overlap_tokens"] = eval2["overlap_tokens"]
                    results.append(rec2)

                    qtype_key = (rec2.get("QA_type") or "unknown").lower()
                    tstats = rr_type_stats.setdefault(qtype_key, {"n": 0, "ok": 0})
                    tstats["n"] += 1
                    if rec2["metric"] != "token_overlap>=25%":
                        tstats["ok"] += int(rec2["correct"])
                        cat = str(rec2.get("category") or "unknown")
                        sub = str(rec2.get("subcategory") or "unknown")
                        c1 = rr_cat_stats.setdefault(cat, {"n": 0, "ok": 0})
                        c1["n"] += 1; c1["ok"] += int(rec2["correct"])
                        c2 = rr_cat_sub_stats.setdefault((cat, sub), {"n": 0, "ok": 0})
                        c2["n"] += 1; c2["ok"] += int(rec2["correct"])

                    # Logging
                    if (rec2.get("QA_type","" ).lower() in {"mc_single","mc_multi","binary"}):
                        print(f"[RECALL r={round_id} t={recall_t:.2f}s][#{rec2['index']}] {rec2['QA_type']} | correct={rec2['correct']} | pred={rec2['parsed_pred']} | raw={rec2['pred_text'][:120]!r}")
                    else:
                        score_str = "N/A" if rec2.get("llm_score") is None else str(rec2.get("llm_score"))
                        print(f"[RECALL r={round_id} t={recall_t:.2f}s][#{rec2['index']}] {rec2['QA_type']} | score={score_str} | raw={rec2['pred_text'][:120]!r}")

                    if rec2["correct"]:
                        orig_t_end = recall_t - round_id * RECALL_DELAY_SEC
                        kept_for_next.append((orig_t_end, item))

                denom_r = sum(v["n"] for k, v in rr_type_stats.items() if k in {"mc_single","mc_multi","binary"})
                acc_r   = sum(v["ok"] for k, v in rr_type_stats.items() if k in {"mc_single","mc_multi","binary"})
                overall_r = (acc_r / max(1, denom_r)) if denom_r else 0.0
                print(f"Round {round_id} overall: {acc_r}/{denom_r} = {overall_r:.3f}")

                print(f"Round {round_id} accuracy by category (choice+binary only):")
                for k, v in rr_cat_stats.items():
                    acc = v["ok"] / max(1, v["n"]) if v["n"] else 0.0
                    print(f"  {k}: {v['ok']}/{v['n']} = {acc:.3f}")

                print(f"Round {round_id} accuracy by (category, subcategory) (choice+binary only):")
                for (ck, sk), v in rr_cat_sub_stats.items():
                    acc = v["ok"] / max(1, v["n"]) if v["n"] else 0.0
                    print(f"  {ck} / {sk}: {v['ok']}/{v['n']} = {acc:.3f}")

                recall_round_stats[round_id] = {
                    "type_stats": rr_type_stats,
                    "category_stats": rr_cat_stats,
                    "category_subcategory_stats": {f"{k[0]}||{k[1]}": v for k, v in rr_cat_sub_stats.items()},
                    "overall_acc_choice_binary": overall_r,
                }
                active = kept_for_next
                round_id += 1
    finally:
        try:
            video.close()
        except Exception:
            pass

    # 保存
    out_path = Path(save_dir) / "eval_predictions.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "results": results,
            "stats": {
                "initial": {
                    "type_stats": type_stats,
                    "category_stats": category_stats,
                    "category_subcategory_stats": {f"{k[0]}||{k[1]}": v for k, v in cat_sub_stats.items()},
                    "overall_acc_choice_binary": (sum(v["ok"] for k, v in type_stats.items() if k in {"mc_single","mc_multi","binary"}) / max(1, sum(v["n"] for k, v in type_stats.items() if k in {"mc_single", "mc_multi", "binary"}))) if any(k in {"mc_single","mc_multi","binary"} for k in type_stats.keys()) else 0.0,
                },
                "recall_rounds": recall_round_stats,
            }
        }, f, ensure_ascii=False, indent=2)
    print(f"Saved to: {out_path}")
    print(f"[INFO] Using VIDEO_PATH: {VIDEO_PATH}")
    print(f"[INFO] Saving to SAVE_PRED_PATH: {SAVE_PRED_PATH}")

    return {
        "video": str(video_path),
        "qa_json": str(json_path),
        "out": str(out_path),
        "num_items": sum(v["n"] for v in type_stats.values()),
        "type_stats": type_stats,
        "category_stats": category_stats,
        "category_subcategory_stats": {f"{k[0]}||{k[1]}": v for k, v in cat_sub_stats.items()},
        "recall_rounds": recall_round_stats,
    }

# =============== 入口 ===============
def main():
    os.makedirs(os.path.dirname(SAVE_PRED_PATH), exist_ok=True)
    save_dir = str(Path(SAVE_PRED_PATH).parent)
    _ = evaluate_single(VIDEO_PATH, JSON_PATH, save_dir, timeline_path=TIMELINE_JSON_PATH)

if __name__ == "__main__":
    main()
