import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

import re
import json
import math
import glob
import torch
import librosa
import tempfile
import numpy as np
from PIL import Image
import soundfile as sf
from pathlib import Path
from collections import defaultdict, OrderedDict
from typing import List, Dict, Any, Tuple, Optional

try:
    # MoviePy 2.x
    from moviepy import VideoFileClip
except Exception:
    # MoviePy 1.x 回退
    from moviepy.video.io.VideoFileClip import VideoFileClip

from transformers import (
    AutoTokenizer,
)

# === InternVL / VideoChatOnline imports ===
from internvl.model.videochat_online import (
    VideoChatOnline_IT,
    InternVLChatConfig,
)
from internvl.train.dataset import build_transform
from internvl.train.constants import IMG_CONTEXT_TOKEN


# -----------------------------
# Config (Batch-friendly)
# -----------------------------
BASE_DIR: str = "./teleego_data"
P_NUM: str = "1"           # e.g. "5" for P5, "1" for P1
QA_SUFFIX: str = "A"       # e.g. "A" -> merged_P5_A.json

# Resolved paths
VIDEO_PATH: str = f"{BASE_DIR}/video_merged/merged_P{P_NUM}.mp4"
JSON_PATH: str  = f"{BASE_DIR}/QAs/merged_P{P_NUM}_{QA_SUFFIX}.json"
SAVE_PRED_PATH: str = f"{BASE_DIR}/outputs/videochat/p{P_NUM}/eval_predictions_P{P_NUM}.json"
TIMELINE_JSON_PATH: str = f"{BASE_DIR}/video_merged/timeline_P{P_NUM}.json"

print(f"[INFO] Using VIDEO_PATH: {VIDEO_PATH}")
print(f"[INFO] Saving to SAVE_PRED_PATH: {SAVE_PRED_PATH}")


# Model name / dir:
MODEL_NAME: str = os.environ.get("VCO_MODEL", "./weights/VideoChat-Online")

# Inference options
TEMPERATURE = 0.5
MAX_NEW_TOKENS = 1024
SEED = 42

# Recall options
RECALL_DELAY_SEC = 60.0
MAX_RECALL_ROUNDS = 10

# Optional: Whisper ASR
USE_ASR = False
ASR_MODEL_NAME = "./weights/whisper-medium"
ASR_LANGUAGE   = "zh"
ASR_DEVICE     = 0 if torch.cuda.is_available() else -1

# =========================
# LLM evaluation (open-ended only) — Azure GPT-4o
# =========================
# ⚠️ 推荐使用环境变量注入凭证；若未配置则自动禁用 LLM 评分。
API_KEY = ""  # ⚠️ Replace with env var in production
API_VER = "2024-08-01-preview"
END_POINT = ""
ENGINE = "4o"  # Azure deployment name

LLM_EVAL_CONFIG = {
    "enabled": bool(API_KEY and END_POINT),
    "provider": "azure",
    "api_key": API_KEY,
    "azure_endpoint": END_POINT,
    "azure_api_version": API_VER,
    "azure_deployment": ENGINE,
    "temperature": 0.0,
    "timeout": 30,
    "prompt": "open_ended_cn_v1",
    "base_url": None,
    "model": None,
}

# -----------------------------
# Logging helper
# -----------------------------
def log_paths():
    print("==== Path Configuration ====")
    print(f"VIDEO_PATH: {VIDEO_PATH}")
    print(f"JSON_PATH: {JSON_PATH}")
    print(f"TIMELINE_JSON_PATH: {TIMELINE_JSON_PATH}")
    print(f"SAVE_PRED_PATH: {SAVE_PRED_PATH}")
    print("============================")


# -----------------------------
# Time helpers (更鲁棒：支持小时>=24 的“跨天”)
# -----------------------------
def parse_hhmmss_flexible(ts: str) -> float:
    s = str(ts).strip()
    if not s:
        raise ValueError("empty timestamp")
    parts = s.split(":")
    if len(parts) == 2:
        h, m = int(parts[0]), int(parts[1]); sec = 0
    elif len(parts) == 3:
        h, m, sec = int(parts[0]), int(parts[1]), int(parts[2])
    else:
        raise ValueError(f"Bad timestamp: {ts}")
    if h >= 24:
        days = h // 24
        h = h % 24
        return days * 86400 + h * 3600 + m * 60 + sec
    return h * 3600 + m * 60 + sec


# -----------------------------
# Timeline mapping for merged videos
# -----------------------------
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
    def from_json(cls, timeline_json_path: str):
        with open(timeline_json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        day_prefix = (data.get("day_prefix_used") or "D")
        segs = []
        for vname, meta in (data.get("mapping_by_input_label") or {}).items():
            st_lab = meta.get("start_label")
            en_lab = meta.get("end_label")
            st_abs = cls._parse_day_label(st_lab, day_prefix)
            en_abs = cls._parse_day_label(en_lab, day_prefix)
            if st_abs is None or en_abs is None:
                continue
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
        if label is None:
            return None
        if isinstance(label, (int, float)):
            return float(label)
        s = str(label).strip()
        abs_s = self._parse_day_label(s, self.day_prefix)
        if abs_s is not None:
            for seg in self.segments:
                if seg["start_abs"] <= abs_s <= seg["end_abs"]:
                    return seg["merged_offset_start_seconds"] + (abs_s - seg["start_abs"])
            return None
        try:
            if ":" in s:
                return parse_hhmmss_flexible(s)
        except Exception:
            pass
        return None

    def find_segment_for_offset(self, merged_sec: float) -> Optional[Dict[str, Any]]:
        for seg in self.segments:
            if seg["merged_offset_start_seconds"] <= merged_sec <= seg["merged_offset_end_seconds"]:
                return seg
        return None


# -----------------------------
# Video helpers
# -----------------------------
def open_video_clip(path: str) -> "VideoFileClip":
    clip = VideoFileClip(path)
    safe_dur = max(0.0, float(clip.duration) - 0.05)
    if hasattr(clip, "subclipped"):
        return clip.subclipped(0, safe_dur)
    else:
        return clip.subclip(0, safe_dur)

def unit_index_for_time(t_end: float, total_seconds: int) -> int:
    if not math.isfinite(t_end):
        return max(0, total_seconds - 1)
    idx = int(math.ceil(max(0.0, t_end))) - 1
    return max(0, min(idx, total_seconds - 1))

def get_image_at_time(video: "VideoFileClip", t_sec: float) -> Image.Image:
    t = min(max(0.0, t_sec), max(0.0, float(video.duration) - 1e-3))
    frame = video.get_frame(t)
    return Image.fromarray(frame.astype(np.uint8)).convert("RGB")

def get_audio_1s_at_time(video: "VideoFileClip", t_start: float, sr: int = 16000) -> np.ndarray:
    t0 = max(0.0, t_start)
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
        return snd.astype(np.float32, copy=False)
    except Exception:
        return np.zeros(int(sr), dtype=np.float32)


# -----------------------------
# Prompt builders / parsing / evaluation
# -----------------------------
def build_question_prompt(item: Dict[str, Any]) -> str:
    qtype = (item.get("QA_type") or "").lower()
    question = (item.get("question") or "").strip()
    options: List[str] = item.get("options", []) if isinstance(item.get("options"), list) else []

    if qtype == "mc_single":
        opt_str = "\n".join(options)
        instr = (
            "请根据当前秒的视频画面与音频内容回答一个单选题。\n"
            "只输出选项字母（例如 A），不要输出解释。"
        )
        return f"{instr}\n题目：{question}\n选项：\n{opt_str}\n你的答案："

    if qtype == "mc_multi":
        opt_str = "\n".join(options)
        instr = (
            "请根据当前秒的视频画面与音频内容回答一个多选题。\n"
            "只输出所有正确选项字母，使用英文逗号分隔（例如 A,B），不要输出解释。"
        )
        return f"{instr}\n题目：{question}\n选项：\n{opt_str}\n你的答案："

    if qtype == "binary":
        instr = (
            "请判断对错（True/False）。\n"
            "只输出 True 或 False，不要输出其他字符。"
        )
        return f"{instr}\n命题：{question}\n你的答案："

    instr = "请简要作答本题。优先给出关键词序列或简短句子，尽量在20个字以内。"
    return f"{instr}\n问题：{question}\n你的答案："

def normalize_letters(s: str) -> List[str]:
    return re.findall(r"[A-Z]", (s or "").upper())

def parse_prediction(text: str, qtype: str):
    t = (text or "").strip()
    qtype = (qtype or "").lower()
    if qtype == "mc_single":
        letters = normalize_letters(t)
        return letters[:1] if letters else []
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

    def _to_letters(x) -> List[str]:
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
# Group by question timestamp
# -----------------------------
def collect_question_groups(qa_items: List[Dict[str, Any]], label_mapper: Optional[Any] = None) -> List[Tuple[float, List[Dict[str, Any]]]]:
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
                    t_end = parse_hhmmss_flexible(s) if ":" in s else float(s)
            except Exception:
                t_end = float("inf")
        rows.append((t_end, pos, it))
    rows.sort(key=lambda r: (r[0], r[1]))
    groups_map: Dict[float, List[Dict[str, Any]]] = OrderedDict()
    for t_end, _, it in rows:
        groups_map.setdefault(t_end, []).append(it)
    return [(t, items) for t, items in groups_map.items()]


# -----------------------------
# Seeding
# -----------------------------
def set_global_seed(seed: int = 42):
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except Exception:
        pass
    try:
        import numpy as _np
        _np.random.seed(seed)
    except Exception:
        pass
    try:
        import random as _random
        _random.seed(seed)
    except Exception:
        pass


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



def evaluate_with_llm(question: str, ground_truth: str, prediction: str, llm_config: Dict = None) -> Dict[str, any]:
    """
    稳健版：
    - 白名单合并 llm_config，防止外部污染（如 {"score":4}）
    - 校验 prompt；非法则回退默认模板
    - 先试 JSON 模式，不支持则回退普通文本模式
    - 宽松解析 score；失败则返回 0
    - 使用 client.with_options(timeout=...)，不向 create() 传 timeout
    """
    import json as _json
    import traceback

    try:
        from openai import AzureOpenAI
    except Exception:
        print("⚠️ openai / AzureOpenAI not available, skip LLM scoring.")
        return {"llm_score": 0}

    default_llm_config = {
        "enabled": bool(API_KEY and END_POINT),
        "provider": "azure",
        "api_key": API_KEY,
        "azure_endpoint": END_POINT,
        "azure_api_version": API_VER,
        "azure_deployment": ENGINE,
        "temperature": 0.0,
        "timeout": 30,
        "prompt": "open_ended_cn_v1",
    }

    # 白名单合并（只采纳认识的键）
    src = llm_config if isinstance(llm_config, dict) else {}
    cfg = {k: src.get(k, v) for k, v in default_llm_config.items()}

    if not (cfg.get("enabled") and cfg.get("api_key") and cfg.get("azure_endpoint")):
        print("ℹ️ LLM disabled or missing credentials; scoring skipped.")
        return {"llm_score": 0}

    # prompt 兜底
    prompt_name = cfg.get("prompt")
    if not isinstance(prompt_name, str) or prompt_name not in prompt_dict:
        print(f'ℹ️ Invalid prompt "{prompt_name}", fallback to "open_ended_cn_v1".')
        prompt_name = "open_ended_cn_v1"

    try:
        prompt_text = prompt_dict[prompt_name].format(
            question=str(question or ""),
            ground_truth=str(ground_truth or ""),
            prediction=str(prediction or "")
        )

        client = AzureOpenAI(
            api_key=cfg["api_key"],
            api_version=cfg["azure_api_version"],
            azure_endpoint=cfg["azure_endpoint"],
        )
        client_tmo = client.with_options(timeout=cfg.get("timeout", 30))

        # 优先 JSON 模式；不支持则回退
        content = None
        try:
            resp = client_tmo.chat.completions.create(
                model=cfg["azure_deployment"],  # 必须是你的 Azure 部署名
                messages=[{"role": "user", "content": prompt_text}],
                response_format={"type": "json_object"},
                temperature=float(cfg.get("temperature", 0.0) or 0.0),
            )
            content = resp.choices[0].message.content if resp and resp.choices else None
        except Exception as e_json:
            print(f"ℹ️ JSON mode unsupported or failed, fallback to text mode: {e_json}")
            resp = client_tmo.chat.completions.create(
                model=cfg["azure_deployment"],
                messages=[{"role": "user", "content": prompt_text}],
                temperature=float(cfg.get("temperature", 0.0) or 0.0),
            )
            content = resp.choices[0].message.content if resp and resp.choices else None

        if not content:
            print("⚠️ Empty content from LLM.")
            return {"llm_score": 0}

        # 尝试严格 JSON 解析
        try:
            data = _json.loads(content)
        except Exception:
            # 宽松提取 score
            m = re.search(r'"?score"?\s*[:=]\s*(\d+)', str(content))
            if m:
                data = {"score": int(m.group(1))}
            else:
                print(f"⚠️ JSON parse failed; content={content!r}")
                return {"llm_score": 0}

        score_raw = data.get("score", 0)
        try:
            score = int(score_raw)
        except Exception:
            score = int(str(score_raw).strip()) if str(score_raw).strip().isdigit() else 0

        return {"llm_score": score if 1 <= score <= 5 else 0}

    except Exception:
        print("⚠️ LLM evaluation error:")
        traceback.print_exc()
        return {"llm_score": 0}


# -----------------------------
# VideoChatOnline adapter
# -----------------------------
class VideoChatOnlineAdapter:
    """
    - 每题重置 system_message，避免 prompt 累积
    - 单帧 + 可选 1 秒音频转写（拼入文本）
    """
    def __init__(self, model_name: str):
        ckpt_hint = os.environ.get("VCO_MODEL", model_name)
        base_cfg  = os.environ.get("VCO_BASE_CFG", "OpenGVLab/InternVL2-4B")

        ckpt_dir = self._resolve_checkpoint_dir(ckpt_hint)
        if ckpt_dir is None:
            raise FileNotFoundError(
                f"[VideoChatOnlineAdapter] No checkpoint found under '{ckpt_hint}'. "
                f"Set env VCO_MODEL to a directory that contains *.safetensors or pytorch_model.bin."
            )

        self.config = InternVLChatConfig.from_pretrained(base_cfg, trust_remote_code=True)
        self.tokenizer = AutoTokenizer.from_pretrained(
            ckpt_dir, add_eos_token=False, trust_remote_code=True, use_fast=False
        )

        torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        kwargs = dict(
            config=self.config,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )
        attn_impl = os.environ.get("ATTN_IMPL", "")
        attn_impl = attn_impl.strip() if isinstance(attn_impl, str) else ""
        if attn_impl:
            kwargs["attn_implementation"] = attn_impl

        self.model = VideoChatOnline_IT.from_pretrained(ckpt_dir, **kwargs).eval()
        if torch.cuda.is_available():
            self.model = self.model.cuda()

        input_size = getattr(self.model.config, "force_image_size", None) \
                     or getattr(getattr(self.model.config, "vision_config", {}), "image_size", None) \
                     or 448
        self.transform = build_transform(is_train=False, input_size=int(input_size))

        self.img_ctx_token_id = self.tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
        self.model.img_context_token_id = self.img_ctx_token_id
        self.model.long_memory_bank  = int(os.environ.get("VCO_LONG_MEM", 64))
        self.model.short_memory_bank = int(os.environ.get("VCO_SHORT_MEM", 64))

        self.base_system_message = (
            "请仔细观察给定的当前1秒视频帧（以及可用的1s音频转写），"
            "依据画面与声音回答问题；若为选择题，仅输出选项字母；"
            "若为判断题，仅输出 True/False；若为开放题，尽量简短。"
        )
        self.model.system_message = self.base_system_message

    @staticmethod
    def _resolve_checkpoint_dir(path_hint: str) -> Optional[str]:
        def _has_weights(d: str) -> bool:
            if not os.path.isdir(d):
                return False
            if os.path.exists(os.path.join(d, "pytorch_model.bin")):
                return True
            if glob.glob(os.path.join(d, "*.safetensors")):
                return True
            return False

        if _has_weights(path_hint):
            return path_hint

        if os.path.isdir(path_hint):
            base = os.path.abspath(path_hint)
            for root, dirs, files in os.walk(base):
                depth = os.path.relpath(root, base).count(os.sep)
                if depth > 2:
                    dirs[:] = []
                    continue
                if _has_weights(root):
                    return root
        return None

    def _build_question_with_image(self, prompt_text: str) -> str:
        return "Frame1: <image>\n" + (prompt_text or "")

    @torch.inference_mode()
    def generate(
        self,
        image: Image.Image,
        prompt_text: str,
        extra_system_msgs: Optional[List[Dict[str, Any]]] = None,
        temperature: float = 0.5,
        max_new_tokens: int = 1024,
        stream: bool = True,
    ) -> str:
        pixel = self.transform(image).unsqueeze(0)
        pixel = pixel.to(self.model.device, dtype=(torch.bfloat16 if torch.cuda.is_available() else torch.float32))

        if extra_system_msgs:
            seg_intro = ""
            for m in extra_system_msgs:
                if isinstance(m, dict) and m.get("role") == "system":
                    seg_intro += (m.get("content") or "")
            self.model.system_message = (self.base_system_message + "\n" + seg_intro).strip()
        else:
            self.model.system_message = self.base_system_message

        question = self._build_question_with_image(prompt_text)

        generation_config = dict(
            num_beams=1,
            do_sample=bool(temperature and temperature > 0),
            temperature=float(temperature or 0.0),
            max_new_tokens=int(max_new_tokens),
            min_new_tokens=1,
        )

        # 关键修复：history=[]；不传 add_generation_prompt
        out = self.model.chat(
            self.tokenizer,
            pixel,
            question,
            generation_config,
            num_patches_list=[1],
            history=[],
            return_history=False,
            verbose=False,
        )

        if isinstance(out, (list, tuple)) and len(out) > 0:
            out = out[0]
        return str(out)


# ---- Singleton factory ----
_VC = {"adapter": None}
def get_videochat_adapter(model_name: Optional[str] = None) -> VideoChatOnlineAdapter:
    if _VC["adapter"] is None:
        hint = model_name or os.environ.get("VCO_MODEL") or MODEL_NAME
        _VC["adapter"] = VideoChatOnlineAdapter(hint)
    return _VC["adapter"]


# -----------------------------
# Optional ASR
# -----------------------------
_ASR = None
def init_asr():
    if not USE_ASR:
        return None
    from transformers import pipeline
    global _ASR
    if _ASR is None:
        _ASR = pipeline(
            "automatic-speech-recognition",
            model=ASR_MODEL_NAME,
            device=ASR_DEVICE,
        )
    return _ASR

def transcribe_audio_1s(audio_np: np.ndarray, sr: int = 16000) -> str:
    if not USE_ASR:
        return ""
    try:
        asr = init_asr()
        out = asr(audio_np.astype(np.float32, copy=False), sampling_rate=sr)
        if isinstance(out, dict):
            return (out.get("text") or "").strip()
        return ""
    except Exception as e:
        print(f"[ASR] error: {e}")
        return ""


# -----------------------------
# Streaming helpers (per-question)
# -----------------------------
def run_streaming_round(
    adapter: VideoChatOnlineAdapter,
    unit_image: Image.Image,
    unit_audio: np.ndarray,
    prompt_text: str,
    audio_base_path: str,
    item_index: Any,
    phase: str = "initial",
    round_id: int = 0,
    segment_intro: Optional[str] = None,
    temperature: float = TEMPERATURE,
) -> Tuple[str, Optional[str], str]:
    saved_audio = None
    transcript = transcribe_audio_1s(unit_audio, sr=16000)
    user_prompt = prompt_text if not transcript else f"{prompt_text}\n\n[1s Audio transcript]: {transcript}"

    extra_msgs = [{"role": "system", "content": segment_intro}] if segment_intro else None
    pred_text = adapter.generate(
        image=unit_image,
        prompt_text=user_prompt,
        extra_system_msgs=extra_msgs,
        temperature=temperature,
        max_new_tokens=MAX_NEW_TOKENS,
        stream=True,
    )
    return pred_text, saved_audio, transcript


# -----------------------------
# Core evaluation (single pair)
# -----------------------------
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

    groups = collect_question_groups(qa_items, label_mapper=(timeline_idx.label_to_merged_seconds if timeline_idx else None))
    max_t = 0.0 if not groups else max([t for t, _ in groups if math.isfinite(t)] + [0.0])

    video = open_video_clip(video_path)
    total_seconds = int(math.ceil(video.duration))
    _ = int(math.ceil(max_t)) if math.isfinite(max_t) else total_seconds

    os.makedirs(save_dir, exist_ok=True)
    audio_base = str(Path(save_dir) / "output.wav")

    results = []
    type_stats: Dict[str, Dict[str, int]] = {}
    category_stats: Dict[str, Dict[str, int]] = {}
    cat_sub_stats: Dict[Tuple[str, str], Dict[str, int]] = {}
    recall_round_stats: Dict[int, Dict[str, Any]] = {}

    correct_for_recall: List[Tuple[float, Dict[str, Any]]] = []
    announced_videos = set()

    adapter = get_videochat_adapter()

    # Initial phase
    for t_end, items in groups:
        idx = unit_index_for_time(t_end, total_seconds)
        unit_image = get_image_at_time(video, t_sec=idx + 1e-3)
        unit_audio = get_audio_1s_at_time(video, t_start=idx, sr=16000)

        seg_intro_text = None
        if timeline_idx and math.isfinite(t_end):
            seg = timeline_idx.find_segment_for_offset(t_end)
            if seg and seg["video"] not in announced_videos:
                seg_intro_text = (
                    f"【当前子视频段信息】起始时间：{seg['start_label']}；"
                    f"人物/场景描述：{seg.get('description','') or '（无）'}。"
                    f"该段在合并视频中的起点：{int(seg['merged_offset_start_seconds'])}s。"
                )
                announced_videos.add(seg["video"])

        for item in items:
            qtype = (item.get("QA_type") or "").lower()
            prompt = build_question_prompt(item)
            try:
                pred_text, saved_audio, transcript = run_streaming_round(
                    adapter=adapter,
                    unit_image=unit_image,
                    unit_audio=unit_audio,
                    prompt_text=prompt,
                    audio_base_path=audio_base,
                    item_index=item.get("index", "na"),
                    phase="initial",
                    round_id=0,
                    segment_intro=seg_intro_text,
                    temperature=TEMPERATURE,
                )
            except Exception as e:
                pred_text, saved_audio, transcript = f"<ERROR: {e}>", None, ""

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
                "audio_transcript": transcript,
            }
            if saved_audio:
                rec["audio_path"] = saved_audio
            if "overlap_tokens" in eval_res:
                rec["overlap_tokens"] = eval_res["overlap_tokens"]
            results.append(rec)

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

            if tkey in {"mc_single", "mc_multi", "binary"}:
                print(f"[t={t_end:.2f}s][#{rec.get('index')}] {tkey} | correct={rec['correct']} | pred={rec['parsed_pred']} | raw={str(rec['pred_text'])[:120]!r}")
            else:
                score_str = "N/A" if rec.get("llm_score") is None else str(rec.get("llm_score"))
                print(f"[t={t_end:.2f}s][#{rec.get('index')}] {tkey} | score={score_str} | raw={str(rec['pred_text'])[:120]!r}")

    # Initial Summary
    denom = sum(v["n"] for k, v in type_stats.items() if k in {"mc_single", "mc_multi", "binary"})
    acc_n = sum(v["ok"] for k, v in type_stats.items() if k in {"mc_single", "mc_multi", "binary"})
    overall = (acc_n / max(1, denom)) if denom else 0.0
    print("===== Summary (Initial phase) =====")
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

    # Recall phases
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
            for recall_t, item in events:
                idx2 = unit_index_for_time(recall_t, total_seconds)
                unit_image2 = get_image_at_time(video, t_sec=idx2 + 1e-3)
                unit_audio2 = get_audio_1s_at_time(video, t_start=idx2, sr=16000)

                seg_intro_text = None
                if timeline_idx and math.isfinite(recall_t):
                    seg = timeline_idx.find_segment_for_offset(recall_t)
                    if seg and seg["video"] not in announced_videos:
                        seg_intro_text = (
                            f"【当前子视频段信息】起始时间：{seg['start_label']}；"
                            f"人物/场景描述：{seg.get('description','') or '（无）'}。"
                            f"该段在合并视频中的起点：{int(seg['merged_offset_start_seconds'])}s。"
                        )
                        announced_videos.add(seg["video"])

                qtype = (item.get("QA_type") or "").lower()
                prompt = build_question_prompt(item)
                try:
                    pred2 = get_videochat_adapter().generate(
                        image=unit_image2,
                        prompt_text=prompt,
                        extra_system_msgs=([{"role": "system", "content": seg_intro_text}] if seg_intro_text else None),
                        temperature=TEMPERATURE,
                        max_new_tokens=MAX_NEW_TOKENS,
                        stream=True,
                    )
                except Exception as e:
                    pred2 = f"<ERROR: {e}>"

                parsed2 = parse_prediction(pred2, qtype)
                eval2 = evaluate_item(item, parsed2)

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
                }
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

                if rec2["correct"]:
                    orig_t_end = recall_t - round_id * RECALL_DELAY_SEC
                    kept_for_next.append((orig_t_end, item))

            denom_r = sum(v["n"] for k, v in rr_type_stats.items() if k in {"mc_single", "mc_multi", "binary"})
            acc_r   = sum(v["ok"] for k, v in rr_type_stats.items() if k in {"mc_single", "mc_multi", "binary"})
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

    # === 结果落盘到 SAVE_PRED_PATH ===
    out_path = Path(SAVE_PRED_PATH)
    os.makedirs(out_path.parent, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "results": results,
            "stats": {
                "initial": {
                    "type_stats": type_stats,
                    "category_stats": category_stats,
                    "category_subcategory_stats": {f"{k[0]}||{k[1]}": v for k, v in cat_sub_stats.items()},
                    "overall_acc_choice_binary": overall,
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
        "num_items": sum(v["n"] for k, v in type_stats.items()),
        "type_stats": type_stats,
        "category_stats": category_stats,
        "category_subcategory_stats": {f"{k[0]}||{k[1]}": v for k, v in cat_sub_stats.items()},
        "recall_rounds": recall_round_stats,
    }


# -----------------------------
# Entry point
# -----------------------------
def run_single_mode():
    os.makedirs(os.path.dirname(SAVE_PRED_PATH), exist_ok=True)
    log_paths()
    save_dir = str(Path(SAVE_PRED_PATH).parent)
    summary = evaluate_single(VIDEO_PATH, JSON_PATH, save_dir, timeline_path=TIMELINE_JSON_PATH)
    return summary

def main():
    set_global_seed(SEED)
    run_single_mode()

if __name__ == "__main__":
    main()
