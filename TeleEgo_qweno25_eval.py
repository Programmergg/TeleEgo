import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "5")

import re
import json
import math
import tqdm
import torch
import tempfile
import threading
import numpy as np
from PIL import Image
import soundfile as sf
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

try:
    from moviepy import VideoFileClip  # MoviePy 2.x
except Exception:
    from moviepy.editor import VideoFileClip  # MoviePy 1.x 兼容

from transformers import (
    AutoTokenizer,
    AutoProcessor,
    TextIteratorStreamer,
    GenerationConfig,  # 使用更干净的生成配置
)
from transformers import Qwen2_5_VLForConditionalGeneration

try:
    from qwen_vl_utils import process_vision_info
    _HAS_QWEN_VL_UTILS = True
except Exception:
    process_vision_info = None
    _HAS_QWEN_VL_UTILS = False

# Whisper ASR（可关）
from transformers import pipeline

# ================= 配置 =================
BASE_DIR: str = "./teleego_data"
P_NUM: str = os.getenv("P_NUM", "1")
QA_SUFFIX: str = os.getenv("QA_SUFFIX", "A")

VIDEO_PATH: str = f"{BASE_DIR}/video_merged/merged_P{P_NUM}.mp4"
JSON_PATH: str  = f"{BASE_DIR}/QAs/merged_P{P_NUM}_{QA_SUFFIX}.json"
SAVE_PRED_PATH: str = f"{BASE_DIR}/outputs/qwen25_omni/p{P_NUM}/eval_predictions_P{P_NUM}.json"
TIMELINE_JSON_PATH: str = f"{BASE_DIR}/video_merged/timeline_P{P_NUM}.json"

# --- 简单 print 方式 ---
print(f"[INFO] Using VIDEO_PATH: {VIDEO_PATH}")
print(f"[INFO] Saving to SAVE_PRED_PATH: {SAVE_PRED_PATH}")

# 模型
QWEN_MODEL_DIR = "./weights/Qwen2.5-VL-7B-Instruct"

# 推理参数（默认贪心最稳；要采样把 TEMPERATURE 改 >0）
TEMPERATURE = 0
TOP_P = 0.9
MAX_NEW_TOKENS = 1024
SEED = 42

# 召回
RECALL_DELAY_SEC = 60.0
MAX_RECALL_ROUNDS = 10

# ASR
USE_ASR = True
ASR_MODEL_NAME = "./weights/whisper-medium"
ASR_LANGUAGE   = "zh"
ASR_DEVICE     = 0 if torch.cuda.is_available() else -1

# 音频留痕
SAVE_UNIT_WAV = True
OUTPUT_AUDIO_BASENAME = "unit.wav"

# LLM 评测（开放题）
API_KEY = ""  # ⚠️ Replace with env var in production
API_VERSION = "2024-08-01-preview"
END_POINT = ""
ENGINE = "4o"  # Azure deployment name


LLM_EVAL_CONFIG = {
    "enabled": True if API_KEY else False,
    "provider": "azure",
    "api_key": API_KEY,
    "azure_endpoint": END_POINT,
    "azure_api_version": API_VERSION,
    "azure_deployment": ENGINE,
    "temperature": 0.0,
    "timeout": 30,
    "prompt": "open_ended_cn_v1",
}

# 数值稳定
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

# ================= 工具函数 =================
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
        """
        支持如下格式（大小写不敏感）：
          - D1-10:00:42
          - d1-10:00
          - day1-10:00:42
          - day1 10:00
        同时容错全角冒号、不同的短横字符。
        """
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

# =============== 按需取帧 / 按需取 1 秒音频 ===============
def open_video_clip(path: str) -> "VideoFileClip":
    clip = VideoFileClip(path)
    safe_dur = max(0.0, float(clip.duration) - 0.05)  # 防止 t==duration 取帧边界异常
    if hasattr(clip, "subclipped"):
        return clip.subclipped(0, safe_dur)  # MoviePy 2.x
    else:
        return clip.subclip(0, safe_dur)     # MoviePy 1.x

def get_image_at_time(video: "VideoFileClip", t_sec: float) -> Image.Image:
    t = min(max(0.0, float(t_sec)), max(0.0, float(video.duration) - 1e-3))
    frame = video.get_frame(t)
    return Image.fromarray(frame.astype(np.uint8)).convert("RGB")

def get_audio_1s_at_time(video: "VideoFileClip", t_start: float, sr: int = 16000) -> np.ndarray:
    """
    取 [t_start, t_start+1] 的单声道 16k 浮点音频；不足 1 秒时补零。
    """
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

# =============== Prompt/解析/评测（保持不变） ===============
def build_question_prompt(item: Dict[str, Any]) -> str:
    qtype = (item.get("QA_type") or "").lower()
    question = (item.get("question") or "").strip()
    options: List[str] = item.get("options", []) if isinstance(item.get("options"), list) else []

    if qtype == "mc_single":
        opt_str = "\n".join(options)
        instr = "请根据当前秒的视频画面与音频转写回答一个单选题。只输出选项字母（例如 A），不要输出解释。"
        return f"{instr}\n题目：{question}\n选项：\n{opt_str}\n你的答案："

    if qtype == "mc_multi":
        opt_str = "\n".join(options)
        instr = "请根据当前秒的视频画面与音频转写回答一个多选题。只输出所有正确选项字母，英文逗号分隔（如 A,B），不要输出解释。"
        return f"{instr}\n题目：{question}\n选项：\n{opt_str}\n你的答案："

    if qtype == "binary":
        instr = "请判断对错（True/False）。只输出 True 或 False，不要输出其他字符。"
        return f"{instr}\n命题：{question}\n你的答案："

    instr = "请简要作答本题。优先给出关键词或短句，尽量在20个字以内。"
    return f"{instr}\n问题：{question}\n你的答案："

def normalize_letters(s: str) -> List[str]:
    return re.findall(r"[A-Z]", (s or "").upper())

def parse_prediction(text: str, qtype: str):
    t = (text or "").strip()
    qtype = (qtype or "").lower()
    if qtype == "mc_single":
        letters = normalize_letters(t); return letters[:1] if letters else []
    if qtype == "mc_multi":
        letters = normalize_letters(t)
        seen = set(); out = []
        for x in letters:
            if x not in seen:
                seen.add(x); out.append(x)
        return out
    if qtype == "binary":
        truish = {"true", "t", "是", "对", "yes", "y", "正确"}
        falsish = {"false", "f", "否", "不对", "no", "n", "错误"}
        low = t.lower()
        if low in truish: return True
        if low in falsish: return False
        if re.search(r"\btrue\b", low): return True
        if re.search(r"\bfalse\b", low): return False
        return None
    return t

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
        if isinstance(x, bool):
            return x
        if isinstance(x, (int, float)):
            return bool(x)
        if isinstance(x, list) and x:
            return _to_bool(x[0])
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
        "api_key": API_KEY,
        "azure_endpoint": END_POINT,
        "azure_api_version": API_VERSION,
        "azure_deployment": ENGINE,
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

# =============== ASR（保持逻辑，仅输入为按需取的 1s 音频） ===============
_ASR = None
def init_asr():
    global _ASR
    if _ASR is None and USE_ASR:
        dtype = torch.float16 if (torch.cuda.is_available() and ASR_DEVICE != -1) else None
        _ASR = pipeline(
            "automatic-speech-recognition",
            model=ASR_MODEL_NAME,
            device=ASR_DEVICE,   # 0 -> cuda:0, -1 -> CPU
            torch_dtype=dtype,   # CPU 下必须是 None
        )
    return _ASR

def transcribe_audio_1s(audio_np: Optional[np.ndarray], sr: int = 16000) -> str:
    if not USE_ASR or audio_np is None or len(audio_np) == 0:
        return ""
    try:
        asr = init_asr()
        audio_arr = np.asarray(audio_np, dtype=np.float32)
        audio_arr = np.clip(np.nan_to_num(audio_arr, nan=0.0, posinf=0.0, neginf=0.0), -1.0, 1.0)
        out = asr({"array": audio_arr, "sampling_rate": sr},
                  return_timestamps=False,
                  generate_kwargs={"task": "transcribe", "language": ASR_LANGUAGE})
        text = ""
        if isinstance(out, dict):
            text = (out.get("text") or "").strip()
        elif isinstance(out, list) and out and isinstance(out[0], dict):
            text = (out[0].get("text") or "").strip()
        # 清洗控制字符并限长，避免污染提示
        if text:
            text = re.sub(r"[\x00-\x08\x0b-\x1f]", "", text)
            if len(text) > 160:
                text = text[:160] + "…"
        return text
    except Exception as e:
        print(f"[ASR] error: {e}")
        return ""

# =============== Qwen2.5-VL init（修复：trust_remote_code + right padding + im_end/eod） ===============
_QWEN = {"model": None, "tokenizer": None, "processor": None}

def init_qwen_vl(model_dir: str = QWEN_MODEL_DIR):
    # 强制 eager 注意力以提高稳定性
    try:
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_dir,
            torch_dtype=torch.float16,
            device_map="auto",
            attn_implementation="eager",
            trust_remote_code=True,  # 关键
        ).eval()
    except TypeError:
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_dir,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,  # 关键
        ).eval()

    # 使用 slow tokenizer + trust_remote_code，确保 chat template 正确
    tokenizer = AutoTokenizer.from_pretrained(
        model_dir, use_fast=False, padding_side="right", trust_remote_code=True
    )
    processor = AutoProcessor.from_pretrained(
        model_dir, trust_remote_code=True
    )
    if hasattr(processor, "tokenizer") and processor.tokenizer is not None:
        processor.tokenizer.padding_side = "right"

    # pad/eos 设定：优先 eod / <|im_end|>
    im_end_id = None
    try:
        im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
        if isinstance(im_end_id, list):
            im_end_id = im_end_id[0]
    except Exception:
        im_end_id = None

    if getattr(tokenizer, "eod_id", None) is not None:
        tokenizer.pad_token_id = tokenizer.eod_id
    elif im_end_id is not None:
        tokenizer.pad_token_id = im_end_id
    elif tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # eos: 合并 <|im_end|> 与默认 eos（若两者都存在）
    eos_id = tokenizer.eos_token_id
    if eos_id is None and im_end_id is not None:
        eos_id = im_end_id
    try:
        if eos_id is not None and im_end_id is not None and eos_id != im_end_id:
            model.generation_config.eos_token_id = [int(eos_id), int(im_end_id)]
        else:
            model.generation_config.eos_token_id = eos_id
        model.generation_config.pad_token_id = tokenizer.pad_token_id
    except Exception:
        pass

    return model, tokenizer, processor

def get_qwen():
    if _QWEN["model"] is None:
        _QWEN["model"], _QWEN["tokenizer"], _QWEN["processor"] = init_qwen_vl(QWEN_MODEL_DIR)
    return _QWEN["model"], _QWEN["tokenizer"], _QWEN["processor"]

# =============== 生成（流式打印，修复：占位符兜底 + GenerationConfig） ===============
BASE_SYS_TEXT = "你是一个多模态助手。根据当前秒的视频画面与1秒音频转写，严格按题目要求简洁作答。"

def run_streaming_round(
    model: Qwen2_5_VLForConditionalGeneration,
    tokenizer: AutoTokenizer,
    processor: AutoProcessor,
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
    save_unit_wav: bool = SAVE_UNIT_WAV,
    audio_base_path: Optional[str] = None,
    round_id: int = 0,
) -> Tuple[str, Optional[str]]:
    # 1) 可选：保存该秒原始音频
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

    # 3) 构造多模态消息
    user_text = prompt_text if not transcript else f"{prompt_text}\n\n[Audio transcript]: {transcript}"
    messages = []
    if sys_text:
        messages.append({"role": "system", "content": sys_text})
    u_content = []
    if image is not None:
        u_content.append({"type": "image", "image": image})
    u_content.append({"type": "text", "text": user_text})
    messages.append({"role": "user", "content": u_content})

    # 4) 编码：优先 utils，若占位符不匹配则 fallback
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = None
    if _HAS_QWEN_VL_UTILS and process_vision_info is not None:
        try:
            image_inputs, video_inputs = process_vision_info(messages)
            img_ph = text.count("<|image_pad|>")
            vid_ph = text.count("<|video_pad|>")
            mismatch = (
                (img_ph != (len(image_inputs) if image_inputs is not None else 0)) or
                (vid_ph != (len(video_inputs) if video_inputs is not None else 0))
            )
            if mismatch:
                print(f"[WARN] Placeholder mismatch. Fallback to simple path. "
                      f"img_ph={img_ph}, imgs={0 if image_inputs is None else len(image_inputs)}; "
                      f"vid_ph={vid_ph}, vids={0 if video_inputs is None else len(video_inputs)}")
                inputs = processor(
                    text=[text],
                    images=[image] if image is not None else None,
                    padding=True,
                    return_tensors="pt",
                )
            else:
                inputs = processor(
                    text=[text],
                    images=image_inputs,
                    videos=video_inputs,
                    padding=True,
                    return_tensors="pt",
                )
        except Exception as e:
            print(f"[WARN] process_vision_info failed: {e}. Fallback to simple path.")
            inputs = processor(
                text=[text],
                images=[image] if image is not None else None,
                padding=True,
                return_tensors="pt",
            )
    else:
        inputs = processor(
            text=[text],
            images=[image] if image is not None else None,
            padding=True,
            return_tensors="pt",
        )
    inputs = inputs.to(model.device)

    # 5) 仅采样时传温度/核采样；使用 GenerationConfig，避免无效 flag 提示
    do_sample = bool(temperature and float(temperature) > 0.0)
    gen_cfg = GenerationConfig(
        max_new_tokens=int(max_new_tokens),
        do_sample=do_sample,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=model.generation_config.eos_token_id,
        use_cache=True,
    )
    if do_sample:
        gen_cfg.temperature = float(temperature)
        gen_cfg.top_p = float(top_p)

    # 6) 流式
    streamer = TextIteratorStreamer(
        tokenizer=tokenizer,
        skip_prompt=True,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )

    th = threading.Thread(target=model.generate, kwargs={**inputs, "streamer": streamer, "generation_config": gen_cfg})
    th.start()

    print_prefix = f"[{phase} q={item_index}] "
    printed_prefix = False
    out_parts: List[str] = []
    for new_text in streamer:
        if not printed_prefix:
            print(print_prefix, end="", flush=True)
            printed_prefix = True
        out_parts.append(new_text)
        print(new_text, end="", flush=True)
    th.join()
    print()  # 换行

    return "".join(out_parts).strip(), saved_audio_path

# =============== QA 调度（按需取帧/取音频） ===============
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

    # ✅ 打开视频一次；每题“按需取帧+音频 1s”
    video = open_video_clip(video_path)
    total_seconds = int(math.ceil(video.duration))
    _ = int(math.ceil(max_t)) if math.isfinite(max_t) else total_seconds

    os.makedirs(save_dir, exist_ok=True)
    audio_base = str(Path(save_dir) / OUTPUT_AUDIO_BASENAME)

    model, tokenizer, processor = get_qwen()

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

            # ✅ 按需抓取该秒图像 + 该秒音频
            unit_image = get_image_at_time(video, t_sec=idx + 0.5)       # 帧取该秒中点更稳
            unit_audio = get_audio_1s_at_time(video, t_start=idx, sr=16000)

            # 段首介绍（只插一次，不累积）
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

                try:
                    pred_text, saved_audio = run_streaming_round(
                        model=model,
                        tokenizer=tokenizer,
                        processor=processor,
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
                        save_unit_wav=SAVE_UNIT_WAV,
                        audio_base_path=audio_base,
                    )
                except Exception as e:
                    pred_text, saved_audio = f"<ERROR: {e}>", None

                parsed = parse_prediction(pred_text, qtype)
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
                    "options": item.get("options", []) if isinstance(item.get("options"), list) else [],
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

                    # ✅ 回忆轮同样按需抓取
                    unit_image2 = get_image_at_time(video, t_sec=idx2 + 0.5)
                    unit_audio2 = get_audio_1s_at_time(video, t_start=idx2, sr=16000)

                    qtype = (item.get("QA_type") or "").lower()
                    prompt = build_question_prompt(item)

                    try:
                        pred2, saved_audio2 = run_streaming_round(
                            model=model,
                            tokenizer=tokenizer,
                            processor=processor,
                            sys_text=BASE_SYS_TEXT,  # 回忆轮不重复段首信息
                            image=unit_image2,
                            audio_1s=unit_audio2,
                            audio_sr=16000,
                            prompt_text=prompt,
                            phase="recall",
                            item_index=item.get("index", "na"),
                            temperature=TEMPERATURE,
                            top_p=TOP_P,
                            max_new_tokens=MAX_NEW_TOKENS,
                            save_unit_wav=SAVE_UNIT_WAV,
                            audio_base_path=audio_base,
                            round_id=round_id,
                        )
                    except Exception as e:
                        pred2, saved_audio2 = f"<ERROR: {e}>", None

                    parsed2 = parse_prediction(pred2, qtype)
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
                        "options": item.get("options", []) if isinstance(item.get("options"), list) else [],
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
