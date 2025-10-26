
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

import re
import json
import math
import tqdm
import torch
import librosa
import tempfile
import numpy as np
from PIL import Image
import soundfile as sf
from pathlib import Path
from moviepy import VideoFileClip
from collections import defaultdict
from transformers import AutoModel, AutoTokenizer
from typing import List, Dict, Any, Tuple, Optional
###############################################
# Config (Batch-friendly)
###############################################
# --- Model ---
MODEL_DIR = "./weights/MiniCPM-o-2_6"

# === Modes ===
# 1) Single-file mode (backward compatible)
""
P_NUM = os.getenv("P_NUM", "0")      # 只要把这里改成 "1" / "2" / "3" ... 即可；也可用环境变量 P_NUM 覆盖
QA_SUFFIX = os.getenv("QA_SUFFIX", "A")  # 如果你的 QA 文件有后缀（如 merged_P5_A.json），也单独抽出来

BASE_DIR = "./teleego_data"

# === Paths ===
VIDEO_PATH: str = f"{BASE_DIR}/video_merged/merged_P{P_NUM}.mp4"
JSON_PATH: str  = f"{BASE_DIR}/QAs/merged_P{P_NUM}_{QA_SUFFIX}.json"
SAVE_PRED_PATH: str = f"{BASE_DIR}/outputs/minicpm/p{P_NUM}/eval_predictions_P{P_NUM}.json"
TIMELINE_JSON_PATH: str = f"{BASE_DIR}/video_merged/timeline_P{P_NUM}.json"

# --- 简单 print 方式 ---
print(f"[INFO] Using VIDEO_PATH: {VIDEO_PATH}")
print(f"[INFO] Saving to SAVE_PRED_PATH: {SAVE_PRED_PATH}")


# 2) Folder mode (generic): pair by filename stems
VIDEO_DIR: Optional[str] = None  # e.g. "/data/videos"
JSON_DIR: Optional[str]  = None  # e.g. "/data/qas"
SAVE_DIR: Optional[str]  = None  # parent dir for all outputs in folder mode

# 3) **TeleeGo dataset mode**
DATA_ROOT: Optional[str] = None  # e.g. "/path/to/teleego_data"
QAS_SUBDIR: str = "QAs"
VIDEOS_SUBDIR: str = "videos"
SAVE_ROOT: Optional[str] = None   # default to SAVE_DIR or <DATA_ROOT>/results if None
TELEEGO_STRICT_MAP: bool = True
TELEEGO_QA_PATTERNS = ["merged_{key}.json", "{key}.json"]

# Inference options
USE_TTS = True
OUTPUT_AUDIO_BASENAME = "output.wav"  # per-question unique filenames will be created
TEMPERATURE = 0.5
MAX_NEW_TOKENS = 4096
# Reproducibility
SEED = 42
# Memory / recall
RECALL_DELAY_SEC = 60.0
MAX_RECALL_ROUNDS = 10

# =========================
# LLM evaluation (open-ended only) — Azure GPT-4o
# =========================
# Get API key from env to avoid hardcoding secrets
API_KEY = ""  # ⚠️ Replace with env var in production
API_VERSION = "2024-08-01-preview"
END_POINT = ""
ENGINE = "4o"  # Azure deployment name

LLM_EVAL_CONFIG = {
    "enabled": True,                 # set False to disable LLM evaluation
    "provider": "azure",
    "api_key": API_KEY,
    "azure_endpoint": END_POINT,
    "azure_api_version": API_VERSION,
    "azure_deployment": ENGINE,
    "temperature": 0.0,
    "timeout": 30,
    "prompt": "open_ended_cn_v1",
    # OpenAI (non-Azure) compatibility fields (unused here)
    "base_url": None,
    "model": None,
}


###############################################
# Time helpers
###############################################

def parse_hhmmss(ts: str) -> float:
    """Parse "HH:MM:SS" (or "MM:SS") to seconds (float)."""
    ts = ts.strip()
    parts = ts.split(":")
    if len(parts) == 3:
        h, m, s = parts
    elif len(parts) == 2:
        h, m, s = "0", parts[0], parts[1]
    else:
        raise ValueError(f"Bad timestamp: {ts}")
    return int(h) * 3600 + int(m) * 60 + float(s)

###############################################
# Timeline helpers for merged videos (NEW)
###############################################

class TimelineIndex:
    """
    Build an index from a timeline JSON describing how multiple input clips
    were merged into a single video. Supports mapping labels like 'D1-09:31:38'
    into merged-video offset seconds.
    """
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

        # 统一一些常见符号
        s = (s.replace("：", ":")
               .replace("—", "-")
               .replace("–", "-")
               .replace("-", "-"))

        # 允许 D / d / day 作为前缀（忽略大小写），秒可省略
        # 例如: day1-10:00:42  或  d1 10:00
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
                # skip malformed
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
        """
        Map label to merged offset seconds.
        Accepts:
          - float/int (already seconds) -> returns as float
          - 'HH:MM:SS' -> returns seconds (no day; not recommended for merged)
          - 'Dk-HH:MM(:SS)' -> map into merged seconds using segments
        """
        if label is None:
            return None
        if isinstance(label, (int, float)):
            return float(label)

        s = str(label).strip()
        # Try day-prefixed format
        abs_s = self._parse_day_label(s, self.day_prefix)
        if abs_s is not None:
            for seg in self.segments:
                if seg["start_abs"] <= abs_s <= seg["end_abs"]:
                    return seg["merged_offset_start_seconds"] + (abs_s - seg["start_abs"])
            return None  # label outside all segments

        # Fallback: HH:MM:SS -> seconds (no day context)
        try:
            if ":" in s:
                return parse_hhmmss(s)
        except Exception:
            pass
        # Not parseable
        return None

    def find_segment_for_offset(self, merged_sec: float) -> Optional[Dict[str, Any]]:
        for seg in self.segments:
            if seg["merged_offset_start_seconds"] <= merged_sec <= seg["merged_offset_end_seconds"]:
                return seg
        return None

###############################################
# Video → Omni content builder (1 frame + 1s audio per second)
###############################################

def build_all_units(video_path: str) -> Tuple[List[Tuple[Image.Image, np.ndarray]], float, int]:
    """Pre-build per-second units for the whole video.
    Returns (units, duration, sr). Each unit is (PIL.Image, audio_1s_np).
    """
    video = VideoFileClip(video_path, fps_source="fps")

    # Trim tiny tail to avoid boundary read warnings
    safe_dur = max(0.0, float(video.duration) - 0.05)
    if hasattr(video, "subclipped"):
        video = video.subclipped(0, safe_dur)
    elif hasattr(video, "subclip"):
        video = video.subclip(0, safe_dur)
    else:
        raise AttributeError("VideoFileClip has neither 'subclipped' nor 'subclip'.")

    # Extract mono 16k audio once
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tf:
        wav_path = tf.name
        video.audio.write_audiofile(wav_path, codec="pcm_s16le", fps=16000)
        audio_np, sr = librosa.load(wav_path, sr=16000, mono=True)

    n_units = int(math.ceil(video.duration))
    units: List[Tuple[Image.Image, np.ndarray]] = []
    for i in range(n_units):
        t = min(i + 1, max(0.0, video.duration - 1e-3))
        frame = video.get_frame(t)
        image = Image.fromarray(frame.astype(np.uint8))
        audio = audio_np[sr * i : sr * (i + 1)]
        units.append((image, audio))

    return units, float(video.duration), sr


def unit_index_for_time(t_end: float, total_units: int) -> int:
    """Map a question timestamp (seconds) to the *single* unit index to use."""
    if total_units <= 0:
        return 0
    if not math.isfinite(t_end):
        return total_units - 1
    idx = int(math.ceil(max(0.0, t_end))) - 1  # unit covering (idx, idx+1]
    return max(0, min(idx, total_units - 1))


def build_single_unit_contents(units: List[Tuple[Image.Image, np.ndarray]], idx: int, flatten: bool = True):
    """Pack only ONE unit (frame+1s audio) at index idx into omni input contents."""
    idx = max(0, min(idx, len(units) - 1))
    image, audio = units[idx]
    if flatten:
        return ["<unit>", image, audio]
    else:
        return [["<unit>", image, audio]]

###############################################
# Prompt builders by QA type
###############################################

def build_question_prompt(item: Dict[str, Any]) -> str:
    qtype = (item.get("QA_type") or "").lower()
    question = item.get("question", "").strip()
    options: List[str] = item.get("options", [])

    if qtype == "mc_single":
        opt_str = "\n".join(options)
        instr = (
            "请根据提供的视频与音频内容回答一个单选题。\n"
            "只输出选项字母（例如 A），不要输出解释。\n"
        )
        return f"{instr}\n题目：{question}\n选项：\n{opt_str}\n你的答案："

    if qtype == "mc_multi":
        opt_str = "\n".join(options)
        instr = (
            "请根据提供的视频与音频内容回答一个多选题。\n"
            "只输出所有正确选项字母，使用英文逗号分隔（例如 A,B），不要输出解释。\n"
        )
        return f"{instr}\n题目：{question}\n选项：\n{opt_str}\n你的答案："

    if qtype == "binary":
        instr = (
            "请判断对错（True/False）。\n"
            "只输出 True 或 False，不要输出其他字符。\n"
        )
        return f"{instr}\n命题：{question}\n你的答案："

    instr = (
        "请简要作答本题。优先给出关键词序列或简短句子。\n"
        "在20个字以内为佳。\n"
    )
    return f"{instr}\n问题：{question}\n你的答案："

###############################################
# Parsing and evaluation helpers
###############################################

def normalize_letters(s: str) -> List[str]:
    letters = re.findall(r"[A-Z]", s.upper())
    return letters


def parse_prediction(text: str, qtype: str):
    t = (text or "").strip()
    qtype = qtype.lower()
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
        if re.search(r"\btrue\b", low):
            return True
        if re.search(r"\bfalse\b", low):
            return False
        return None
    return t


def evaluate_item(gt: Dict[str, Any], pred) -> Dict[str, Any]:
    """
    更鲁棒的判分：
    - 选择题：支持 GT 为 ["A"] / "A" / "A,B" / ["A","B"] 等
    - 判断题：支持 GT 为 True/False / 1/0 / "true"/"false"/"是"/"否" / [True] 等
    - 开放题：把 GT 转成字符串后做 token overlap
    """
    qtype = (gt.get("QA_type") or "").lower()
    ans = gt.get("answer", {}) or {}
    raw = ans.get("value", None)

    def _to_letters(x) -> list[str]:
        # 将各种形式统一转成大写字母列表
        if isinstance(x, list):
            s = ",".join(map(str, x))
        else:
            s = "" if x is None else str(x)
        return re.findall(r"[A-Z]", s.upper())

    def _to_bool(x) -> bool:
        # 将各种形式统一转 bool
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

    # --- 选择题 ---
    if qtype in {"mc_single", "mc_multi"}:
        gt_letters = _to_letters(raw)
        if qtype == "mc_single":
            pred_letters = [str(x).upper() for x in (pred or [])]
            result["correct"] = (len(pred_letters) == 1 and pred_letters[0] in gt_letters)
            result["metric"] = "accuracy"
            return result
        # mc_multi
        pred_set = set([str(x).upper() for x in (pred or [])])
        gt_set = set(gt_letters)
        result["correct"] = (len(gt_set) > 0 and pred_set == gt_set)
        result["metric"] = "exact_set_match"
        return result

    # --- 判断题 ---
    if qtype == "binary":
        gt_bool = _to_bool(raw)
        result["correct"] = (pred is not None and bool(pred) == gt_bool)
        result["metric"] = "accuracy"
        return result

    # --- 开放题 ---
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

###############################################
# QA scheduling by evidence -> questionstamp
###############################################

def collect_questionstamps(qa_items: List[Dict[str, Any]], label_mapper: Optional[Any] = None) -> List[Tuple[float, List[Dict[str, Any]]]]:
    """
    根据 evidence.timestep.end 提取问题发生时间并排序。
    若提供 label_mapper（如 TimelineIndex.label_to_merged_seconds），
    会优先将类似 'D1-09:31:38' 的标签映射到合并视频的秒。
    返回 [(t_end_seconds, [item]), ...]
    """
    rows: List[Tuple[float, int, Dict[str, Any]]] = []  # (t_end, pos, item)

    for pos, it in enumerate(qa_items):
        ts = (it.get("evidence") or {}).get("timestep") or {}
        end_raw = ts.get("end") or ts.get("End") or ts.get("to")
        t_end = None
        # Timeline-based mapping first
        if label_mapper is not None:
            try:
                t_end = label_mapper(end_raw)
            except Exception:
                t_end = None
        # Fallbacks
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

###############################################
# Seeding helpers


def _lower_safe(x):
    """Robust lower(): None -> '', non-str -> str() -> lower()."""
    if x is None:
        return ""
    return str(x).strip().lower()

###############################################

def set_global_seed(seed: int = 42):
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch
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

###############################################
# LLM judge prompt + caller (for open_ended)
###############################################

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



def evaluate_with_llm(
    question: str,
    ground_truth: str,
    prediction: str,
    llm_config: Dict = None
) -> Dict[str, any]:
    """
    Use Azure OpenAI GPT-4o to evaluate open-ended answers.
    Return {"llm_score": <int 1..5>} (0 if disabled/error).
    """
    import json as _json
    import traceback

    # default Azure config (auto-fill)
    default_llm_config = {
        "enabled": True,
        "provider": "azure",
        "api_key": API_KEY,
        "azure_endpoint": END_POINT,
        "azure_api_version": API_VERSION,
        "azure_deployment": ENGINE,
        "temperature": 0.0,
        "timeout": 30,
        "prompt": "open_ended_cn_v1",
    }
    if llm_config is None:
        llm_config = {}
    for k, v in default_llm_config.items():
        llm_config.setdefault(k, v)

    if not llm_config.get("enabled", False):
        return {"llm_score": 0}

    try:
        import openai
        try:
            from openai import AzureOpenAI
        except Exception:
            AzureOpenAI = None

        prompt_name = llm_config["prompt"]
        assert prompt_name in prompt_dict, (
            f"Prompt '{prompt_name}' not found in prompt_dict. "
            f"Available: {list(prompt_dict.keys())}"
        )
        prompt = prompt_dict[prompt_name].format(
            question=str(question or ""),
            ground_truth=str(ground_truth or ""),
            prediction=str(prediction or "")
        )

        provider = str(llm_config.get("provider", "azure")).lower()

        if provider == "azure":
            required = ["api_key", "azure_endpoint", "azure_api_version", "azure_deployment"]
            missing = [k for k in required if not llm_config.get(k)]
            if missing:
                return {"llm_score": 0}

            if AzureOpenAI is None:
                return {"llm_score": 0}

            client = AzureOpenAI(
                api_key=llm_config["api_key"],
                api_version=llm_config["azure_api_version"],
                azure_endpoint=llm_config["azure_endpoint"],
            )

            response = client.chat.completions.create(
                model=llm_config["azure_deployment"],
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=llm_config.get("temperature", 0.0),
                timeout=llm_config.get("timeout", 30),
            )
        else:
            client = openai.OpenAI(api_key=llm_config.get("api_key", "dummy"))
            response = client.chat.completions.create(
                model=llm_config.get("model", "gpt-4o"),
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=llm_config.get("temperature", 0.0),
                timeout=llm_config.get("timeout", 30),
            )

        content = response.choices[0].message.content
        data = _json.loads(content)

        score_raw = data.get("score", 0)  # expect 1..5
        try:
            score = int(score_raw)
        except Exception:
            try:
                score = int(str(score_raw).strip())
            except Exception:
                score = 0

        if 1 <= score <= 5:
            return {"llm_score": score}
        else:
            return {"llm_score": 0}

    except AssertionError:
        raise
    except Exception:
        print("⚠️ LLM evaluation error:", traceback.format_exc())
        return {"llm_score": 0}

###############################################
# Streaming helpers
###############################################

def run_streaming_round(
    model,
    tokenizer,
    sys_msg,
    unit_contents_flat: List[Any],
    prompt_text: str,
    session_id: str,
    generate_audio: bool,
    temperature: float,
    audio_base_path: str,
    item_index: Any,
    phase: str = "initial",
    round_id: int = 0,
    extra_system_msgs: Optional[List[Dict[str, Any]]] = None,  # NEW: segment intro
):
    """
    Streaming per-question:
      1) reset_session
      2) streaming_prefill(system [+ extra_system_msgs])
      3) streaming_prefill(<unit>)
      4) streaming_prefill(<question> prompt)
      5) streaming_generate for text/audio
    Returns (pred_text, saved_audio_path_or_None)
    """
    # 1) reset
    try:
        model.reset_session()
    except Exception:
        try:
            model.reset()
        except Exception:
            pass

    # 2) system prefill
    _ = model.streaming_prefill(session_id=session_id, msgs=[sys_msg], tokenizer=tokenizer)

    # 2.1) extra system msgs (segment intro)
    if extra_system_msgs:
        _ = model.streaming_prefill(session_id=session_id, msgs=extra_system_msgs, tokenizer=tokenizer)

    # 3) unit content
    _ = model.streaming_prefill(session_id=session_id, msgs=[{"role": "user", "content": unit_contents_flat}], tokenizer=tokenizer)

    # 4) question prompt
    _ = model.streaming_prefill(session_id=session_id, msgs=[{"role": "user", "content": ["<question>", prompt_text]}], tokenizer=tokenizer)

    # 5) generate
    res = model.streaming_generate(
        session_id=session_id,
        tokenizer=tokenizer,
        temperature=temperature,
        generate_audio=generate_audio
    )

    audios = []
    text = ""
    sampling_rate = None
    if generate_audio:
        for r in res:
            # 兼容对象/字典两种返回格式
            if isinstance(r, dict):
                audio_wav = r.get("audio_wav")
                txt = r.get("text")
                sr = r.get("sampling_rate")
            else:
                audio_wav = getattr(r, "audio_wav", None)
                txt = getattr(r, "text", None)
                sr = getattr(r, "sampling_rate", None)

            if audio_wav is not None:
                audios.append(audio_wav)
            if txt:
                text += txt
            if sr:
                sampling_rate = sr

        saved_audio = None
        if audios:
            arr = np.concatenate(audios, axis=0)
            base, ext = os.path.splitext(audio_base_path or "output.wav")
            uniq = f"{base}_{phase}_q{item_index}"
            if round_id:
                uniq += f"_r{round_id}"
            uniq += ext if ext else ".wav"
            sf.write(uniq, arr, samplerate=sampling_rate or 16000)
            saved_audio = uniq
        return text, saved_audio
    else:
        for r in res:
            if isinstance(r, dict):
                text += r.get("text", "")
            else:
                txt = getattr(r, "text", None)
                if txt:
                    text += txt
        return text, None

###############################################
# Core evaluation (single pair)
###############################################

def evaluate_single(video_path: str, json_path: str, save_dir: str, model, tokenizer, sys_msg, timeline_path: Optional[str] = None) -> Dict[str, Any]:
    """Runs the end-to-end evaluation on one (video, QA-json) pair and saves results under save_dir.
    Returns a dict with summary info for aggregation.
    """
    # set seeds for reproducibility
    set_global_seed(SEED)
    assert os.path.exists(json_path), f"QA json not found: {json_path}"
    assert os.path.exists(video_path), f"Video not found: {video_path}"

    # (NEW) Optional timeline index
    timeline_idx = None
    if timeline_path:
        try:
            timeline_idx = TimelineIndex.from_json(timeline_path)
        except Exception as _e:
            print(f"⚠️ Failed to load timeline: {timeline_path} -> {_e}")

    with open(json_path, "r", encoding="utf-8") as f:
        qa_items: List[Dict[str, Any]] = json.load(f)

    # 1) Group questions by evidence end timestamp (support timeline mapping)
    groups = collect_questionstamps(qa_items, label_mapper=(timeline_idx.label_to_merged_seconds if timeline_idx else None))
    max_t = 0.0 if not groups else max([t for t, _ in groups if math.isfinite(t)] + [0.0])

    # 2) Build units
    units, duration, sr = build_all_units(video_path)
    _ = int(math.ceil(max_t)) if math.isfinite(max_t) else len(units)

    os.makedirs(save_dir, exist_ok=True)
    audio_base = str(Path(save_dir) / OUTPUT_AUDIO_BASENAME)

    results = []
    # === stats containers
    type_stats: Dict[str, Dict[str, int]] = {}          # by QA_type (initial phase)
    category_stats: Dict[str, Dict[str, int]] = {}      # by category (initial phase)
    cat_sub_stats: Dict[Tuple[str, str], Dict[str, int]] = {}  # by (category, subcategory) (initial phase)

    # recall per-round stats: for each round -> dict of stats
    recall_round_stats: Dict[int, Dict[str, Any]] = {}

    correct_for_recall: List[Tuple[float, Dict[str, Any]]] = []  # (t_end, item)

    # for segment intro messages (insert once per child video)
    announced_videos = set()

    # 3) Iterate groups in chronological order
    for t_end, items in tqdm.tqdm(groups, desc="QA by questionstamp"):
        idx = unit_index_for_time(t_end, len(units))
        contents = build_single_unit_contents(units, idx, flatten=True)

        # (NEW) extra system prompt at the start of each segment (once)
        extra_msgs = None
        if timeline_idx and math.isfinite(t_end):
            seg = timeline_idx.find_segment_for_offset(t_end)
            if seg and seg["video"] not in announced_videos:
                intro_txt = (
                    f"【当前子视频段信息】起始时间：{seg['start_label']}；"
                    f"人物/场景描述：{seg.get('description','') or '（无）'}。"
                    f"该段在合并视频中的起点：{int(seg['merged_offset_start_seconds'])}s。"
                )
                extra_msgs = [{"role": "system", "content": intro_txt}]
                announced_videos.add(seg["video"])

        for item in items:
            qtype = (item.get("QA_type") or "").lower()
            prompt = build_question_prompt(item)

            # streaming round
            session_id = f"q-{item.get('index','na')}-initial"
            try:
                pred_text, saved_audio = run_streaming_round(
                    model=model,
                    tokenizer=tokenizer,
                    sys_msg=sys_msg,
                    unit_contents_flat=list(contents),
                    prompt_text=prompt,
                    session_id=session_id,
                    generate_audio=USE_TTS,
                    temperature=TEMPERATURE,
                    audio_base_path=audio_base,
                    item_index=item.get("index", "na"),
                    phase="initial",
                    round_id=0,
                    extra_system_msgs=extra_msgs,  # NEW: segment intro
                )
            except Exception as e:
                pred_text, saved_audio = f"<ERROR: {e}>", None

            parsed = parse_prediction(pred_text, qtype)
            eval_res = evaluate_item(item, parsed)

            # open_ended: LLM score 1..5
            llm_score = 0
            if qtype not in {"mc_single", "mc_multi", "binary"} and LLM_EVAL_CONFIG.get("enabled", False):
                gt_vals = (item.get("answer", {}) or {}).get("value", [])
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
                "QA_type": item.get("QA_type") or "open_ended",
                "question": item.get("question"),
                "options": item.get("options", []),
                "gold": item.get("answer", {}).get("value", []),
                "questionstamp": None if not math.isfinite(t_end) else t_end,
                "pred_text": pred_text,
                "parsed_pred": parsed,
                "correct": eval_res.get("correct", False),
                "metric": eval_res.get("metric", ""),
                "llm_score": llm_score if qtype not in {"mc_single","mc_multi","binary"} else None,
            }
            if saved_audio:
                rec["audio_path"] = saved_audio
            if "overlap_tokens" in eval_res:
                rec["overlap_tokens"] = eval_res["overlap_tokens"]
            results.append(rec)

            # Per-type stats (count correctness only for non-open-ended)
            tkey = _lower_safe(rec.get("QA_type"))
            s = type_stats.setdefault(tkey, {"n": 0, "ok": 0})
            s["n"] += 1
            s["ok"] += int(rec["correct"]) if rec["metric"] != "token_overlap>=25%" else 0

            # === NEW: category / (category, subcategory) stats (initial phase)
            if rec["metric"] != "token_overlap>=25%":
                cat = str(rec.get("category") or "unknown")
                sub = str(rec.get("subcategory") or "unknown")
                s1 = category_stats.setdefault(cat, {"n": 0, "ok": 0})
                s1["n"] += 1; s1["ok"] += int(rec["correct"])
                s2 = cat_sub_stats.setdefault((cat, sub), {"n": 0, "ok": 0})
                s2["n"] += 1; s2["ok"] += int(rec["correct"])

            # enqueue recall if correct and timestamp is finite
            if rec["correct"] and math.isfinite(t_end):
                correct_for_recall.append((t_end, item))

            # Logging: open_ended prints score; others print correct
            if tkey in {"mc_single", "mc_multi", "binary"}:
                print(f"[t={t_end:.2f}s][#{rec['index']}] {tkey} | correct={rec['correct']} | pred={rec['parsed_pred']} | raw={rec['pred_text'][:120]!r}")
            else:
                score_str = "N/A" if rec.get("llm_score") is None else str(rec.get("llm_score"))
                print(f"[t={t_end:.2f}s][#{rec['index']}] {tkey} | score={score_str} | raw={rec['pred_text'][:120]!r}")

    # Summary for initial phase
    print("===== Summary (Initial phase) =====")
    denom = sum(v["n"] for k, v in type_stats.items() if k in {"mc_single", "mc_multi", "binary"})
    acc_n = sum(v["ok"] for k, v in type_stats.items())
    overall = (acc_n / max(1, denom)) if denom else 0.0
    for k, v in type_stats.items():
        if k in {"mc_single", "mc_multi", "binary"}:
            acc = v["ok"] / max(1, v["n"])
            print(f"  {k}: {v['ok']}/{v['n']} = {acc:.3f}")
        else:
            print(f"  {k}: {v['n']} samples (text metric)")
    print(f"Overall (choice+binary only): {acc_n}/{denom} = {overall:.3f}")

    print("===== Accuracy by category (Initial, choice+binary only) =====")
    for k, v in category_stats.items():
        acc = v["ok"] / max(1, v["n"])
        print(f"  {k}: {v['ok']}/{v['n']} = {acc:.3f}")

    print("===== Accuracy by (category, subcategory) (Initial, choice+binary only) =====")
    for (ck, sk), v in cat_sub_stats.items():
        acc = v["ok"] / max(1, v["n"])
        print(f"  {ck} / {sk}: {v['ok']}/{v['n']} = {acc:.3f}")

    # Open-ended LLM average score (initial phase only)
    open_llm = [r for r in results if (r.get("QA_type","" ).lower() not in {"mc_single","mc_multi","binary"}) and (r.get("llm_score") is not None)]
    if open_llm:
        scores = [max(0, int(r.get("llm_score", 0))) for r in open_llm]
        avg = sum(scores) / max(1, len(scores))
        print(f"Open-ended (LLM) avg score (initial): {avg:.3f} over {len(scores)} items")

    print(f"[INFO] Using VIDEO_PATH: {VIDEO_PATH}")
    print(f"[INFO] Saving to SAVE_PRED_PATH: {SAVE_PRED_PATH}")
    
    # 4.5) RECALL PHASE
    if RECALL_DELAY_SEC > 0 and correct_for_recall:
        active: List[Tuple[float, Dict[str, Any]]] = list(correct_for_recall)  # (t_end, item)
        round_id = 1
        while active and round_id <= MAX_RECALL_ROUNDS:
            # per-round containers
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
                idx = unit_index_for_time(recall_t, len(units))
                contents = build_single_unit_contents(units, idx, flatten=True)

                # segment intro for recall when entering a new segment
                extra_msgs = None
                if timeline_idx and math.isfinite(recall_t):
                    seg = timeline_idx.find_segment_for_offset(recall_t)
                    if seg and seg["video"] not in announced_videos:
                        intro_txt = (
                            f"【当前子视频段信息】起始时间：{seg['start_label']}；"
                            f"人物/场景描述：{seg.get('description','') or '（无）'}。"
                            f"该段在合并视频中的起点：{int(seg['merged_offset_start_seconds'])}s。"
                        )
                        extra_msgs = [{"role": "system", "content": intro_txt}]
                        announced_videos.add(seg["video"])

                qtype = (item.get("QA_type") or "").lower()
                prompt = build_question_prompt(item)

                session_id = f"q-{item.get('index','na')}-recall-{round_id}"
                try:
                    pred_text, saved_audio = run_streaming_round(
                        model=model,
                        tokenizer=tokenizer,
                        sys_msg=sys_msg,
                        unit_contents_flat=list(contents),
                        prompt_text=prompt,
                        session_id=session_id,
                        generate_audio=USE_TTS,
                        temperature=TEMPERATURE,
                        audio_base_path=audio_base,
                        item_index=item.get("index", "na"),
                        phase="recall",
                        round_id=round_id,
                        extra_system_msgs=extra_msgs,  # NEW: segment intro for recall
                    )
                except Exception as e:
                    pred_text, saved_audio = f"<ERROR: {e}>", None

                parsed = parse_prediction(pred_text, qtype)
                eval_res = evaluate_item(item, parsed)

                # open_ended: LLM score 1..5
                llm_score = 0
                if qtype not in {"mc_single", "mc_multi", "binary"} and LLM_EVAL_CONFIG.get("enabled", False):
                    gt_vals = (item.get("answer", {}) or {}).get("value", [])
                    ground_truth = "" if not gt_vals else str(gt_vals[0])
                    llm_out = evaluate_with_llm(
                        question=item.get("question", ""),
                        ground_truth=ground_truth,
                        prediction=pred_text,
                        llm_config=LLM_EVAL_CONFIG
                    )
                    llm_score = int(llm_out.get("llm_score", 0))

                rec2 = {
                    "phase": "recall",
                    "round": round_id,
                    "index": item.get("index"),
                    "category": item.get("category"),
                    "subcategory": item.get("subcategory"),
                    "QA_type": item.get("QA_type") or "open_ended",
                    "question": item.get("question"),
                    "options": item.get("options", []),
                    "gold": item.get("answer", {}).get("value", []),
                    "questionstamp": recall_t,
                    "recall_delay": round_id * RECALL_DELAY_SEC,
                    "pred_text": pred_text,
                    "parsed_pred": parsed,
                    "correct": eval_res.get("correct", False),
                    "metric": eval_res.get("metric", ""),
                    "llm_score": llm_score if qtype not in {"mc_single","mc_multi","binary"} else None,
                }
                if saved_audio:
                    rec2["audio_path"] = saved_audio
                if "overlap_tokens" in eval_res:
                    rec2["overlap_tokens"] = eval_res["overlap_tokens"]
                results.append(rec2)

                # Per-round stats (choice/binary only)
                qtype_key = _lower_safe(rec2.get("QA_type"))
                tstats = rr_type_stats.setdefault(qtype_key, {"n": 0, "ok": 0})
                tstats["n"] += 1
                if rec2["metric"] != "token_overlap>=25%":
                    tstats["ok"] += int(rec2["correct"])
                    # Category & subcategory (NEW for recall)
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

            # per-round summaries
            denom_r = sum(v["n"] for v in rr_type_stats.values())
            acc_r   = sum(v["ok"] for v in rr_type_stats.values())
            overall_r = (acc_r / max(1, denom_r)) if denom_r else 0.0
            print(f"Round {round_id} per-type:")
            for qt, v in rr_type_stats.items():
                acc_t = v["ok"] / max(1, v["n"]) if v["n"] else 0.0
                print(f"  {qt}: {v['ok']}/{v['n']} = {acc_t:.3f}")
            print(f"Round {round_id} overall: {acc_r}/{denom_r} = {overall_r:.3f}")

            print(f"Round {round_id} accuracy by category (choice+binary only):")
            for k, v in rr_cat_stats.items():
                acc = v["ok"] / max(1, v["n"])
                print(f"  {k}: {v['ok']}/{v['n']} = {acc:.3f}")

            print(f"Round {round_id} accuracy by (category, subcategory) (choice+binary only):")
            for (ck, sk), v in rr_cat_sub_stats.items():
                acc = v["ok"] / max(1, v["n"])
                print(f"  {ck} / {sk}: {v['ok']}/{v['n']} = {acc:.3f}")

            # store into recall_round_stats (NEW structure)
            recall_round_stats[round_id] = {
                "type_stats": rr_type_stats,
                "category_stats": rr_cat_stats,
                "category_subcategory_stats": {f"{k[0]}||{k[1]}": v for k, v in rr_cat_sub_stats.items()},
                "overall_acc_choice_binary": overall_r,
            }

            active = kept_for_next
            round_id += 1

    # 5) Save predictions for this pair
    out_path = Path(save_dir) / "eval_predictions.json"
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
                "recall_rounds": recall_round_stats,  # NEW: per-round stats with category/subcategory
            }
        }, f, ensure_ascii=False, indent=2)
    print(f"Saved to: {out_path}")
    print(f"[INFO] Using VIDEO_PATH: {VIDEO_PATH}")
    print(f"[INFO] Saving to SAVE_PRED_PATH: {SAVE_PRED_PATH}")
    
    # return summary for aggregation
    return {
        "video": str(video_path),
        "qa_json": str(json_path),
        "out": str(out_path),
        "choice_binary_overall_acc": overall,
        "num_items": sum(v["n"] for v in type_stats.values()),
        "type_stats": type_stats,
        "category_stats": category_stats,
        "category_subcategory_stats": {f"{k[0]}||{k[1]}": v for k, v in cat_sub_stats.items()},
        "recall_rounds": recall_round_stats,
    }

###############################################
# Batch pairing helpers
###############################################

VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".avi", ".mpg", ".mpeg", ".m4v"}


def find_pairs(video_dir: str, json_dir: str) -> List[Tuple[Path, Path]]:
    """Find (video, qa_json) pairs by matching filename stems. Case-insensitive.
    e.g. video 'scene_01.mp4' pairs with 'scene_01.json'.
    """
    print("video_dir:", video_dir)
    vdir = Path(video_dir)
    jdir = Path(json_dir)
    videos = [p for p in vdir.rglob("*") if p.suffix.lower() in VIDEO_EXTS]
    jsons  = {p.stem.lower(): p for p in jdir.rglob("*.json")}

    pairs: List[Tuple[Path, Path]] = []
    for v in sorted(videos):
        key = v.stem.lower()
        if key in jsons:
            pairs.append((v, jsons[key]))
        else:
            print(f"⚠️ No QA json found for video: {v}")
    return pairs

# === TeleeGo-specific pairing ===
# merged_P1.json  -> videos/P1/*.mp4  (one-to-many)
# merged_<KEY>.json maps to videos/<KEY> directory.

def find_teleego_groups(data_root: str, qas_subdir: str = "QAs", videos_subdir: str = "videos"):
    root = Path(data_root)
    qdir = root / qas_subdir
    vdir = root / videos_subdir

    # Map QA key -> path by accepted filename patterns
    qa_map: Dict[str, Path] = {}
    for q in qdir.glob("*.json"):
        stem = q.stem.lower()
        # Accept 'merged_p1' or 'p1' as key
        key = stem
        if stem.startswith("merged_"):
            key = stem.replace("merged_", "", 1)
        qa_map[key] = q

    # For each videos/<KEY> directory, find its QA json
    groups = []  # list of tuples: (key, qa_path, [video_paths])
    for dirent in sorted(vdir.iterdir()):
        if not dirent.is_dir():
            continue
        key = dirent.name.lower()
        # choose QA path using patterns
        qa_path: Optional[Path] = None
        if key in qa_map:
            qa_path = qa_map[key]
        else:
            # Fallback: search
            match = None
            for q in qdir.glob("*.json"):
                s = q.stem.lower()
                if s == key or s == f"merged_{key}":
                    match = q; break
            qa_path = match
        if not qa_path or not qa_path.exists():
            print(f"⚠️ No QA json found for videos folder '{dirent.name}'. Expect 'merged_{dirent.name}.json' under {qdir}.")
            continue
        videos = [p for p in dirent.rglob("*") if p.suffix.lower() in VIDEO_EXTS]
        if not videos:
            print(f"⚠️ No videos under: {dirent}")
            continue
        groups.append((dirent.name, qa_path, videos))
    return groups


def normalize_stem(p: Path) -> str:
    return p.stem.lower()


def split_qa_by_video(qa_items: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    """Attempt to split QA items per video using common fields.
    Returns mapping: normalized_video_stem -> items.
    If no per-video info is found, returns an empty dict.
    Heuristics: look for keys like 'video', 'video_name', 'source', 'file',
    or under 'evidence': {'video': ..., 'source': ...}.
    """
    by_video: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

    def extract_name(it: Dict[str, Any]) -> Optional[str]:
        cand = (
            it.get("video") or it.get("video_name") or it.get("source") or it.get("file")
            or (it.get("evidence") or {}).get("video")
            or (it.get("evidence") or {}).get("source")
        )
        if not cand:
            return None
        s = str(cand)
        # accept basename or stem-like values
        name = os.path.splitext(os.path.basename(s))[0].strip()
        return name.lower() if name else None

    any_hit = False
    for it in qa_items:
        nm = extract_name(it)
        if nm:
            any_hit = True
            by_video[nm].append(it)
    return by_video if any_hit else {}

###############################################
# Entry points
###############################################

def run_single_mode():
    assert VIDEO_PATH and JSON_PATH and SAVE_PRED_PATH, "Single-file mode requires VIDEO_PATH, JSON_PATH, SAVE_PRED_PATH"

    # 3) Load model/tokenizer once
    model = AutoModel.from_pretrained(
        MODEL_DIR,
        trust_remote_code=True,
        attn_implementation="sdpa",
        torch_dtype=torch.bfloat16,
    )
    model = model.eval().cuda()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)
    try:
        model.init_tts()
    except Exception:
        pass
    sys_msg = model.get_sys_prompt(mode='omni', language='zh')

    save_dir = str(Path(SAVE_PRED_PATH).parent)
    summary = evaluate_single(VIDEO_PATH, JSON_PATH, save_dir, model, tokenizer, sys_msg, timeline_path=TIMELINE_JSON_PATH)
    return summary


def run_batch_mode():
    assert VIDEO_DIR and JSON_DIR and SAVE_DIR, "Folder mode requires VIDEO_DIR, JSON_DIR, SAVE_DIR"

    pairs = find_pairs(VIDEO_DIR, JSON_DIR)
    if not pairs:
        raise RuntimeError("No (video, json) pairs found. Ensure filenames match by stem.")

    # 3) Load model/tokenizer once
    model = AutoModel.from_pretrained(
        MODEL_DIR,
        trust_remote_code=True,
        attn_implementation="sdpa",
        torch_dtype=torch.bfloat16,
    )
    model = model.eval().cuda()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)
    try:
        model.init_tts()
    except Exception:
        pass
    sys_msg = model.get_sys_prompt(mode='omni', language='zh')

    agg: List[Dict[str, Any]] = []
    for vid_path, qa_path in tqdm.tqdm(pairs, desc="Pairs"):
        vid_stem = vid_path.stem
        pair_save_dir = str(Path(SAVE_DIR) / vid_stem)
        print(f"===== Processing: {vid_stem} =====")
        summ = evaluate_single(str(vid_path), str(qa_path), pair_save_dir, model, tokenizer, sys_msg, timeline_path=TIMELINE_JSON_PATH)
        agg.append(summ)

    # Save aggregate summary
    os.makedirs(SAVE_DIR, exist_ok=True)
    aggregate_path = Path(SAVE_DIR) / "aggregate_results.json"
    with open(aggregate_path, "w", encoding="utf-8") as f:
        json.dump({"pairs": agg}, f, ensure_ascii=False, indent=2)
    print(f"Aggregate saved to: {aggregate_path}")
    return {"pairs": agg, "aggregate_path": str(aggregate_path)}


def run_teleego_mode():
    assert DATA_ROOT, "TeleeGo mode requires DATA_ROOT"
    save_root = Path(SAVE_ROOT or (Path(DATA_ROOT) / "results"))
    groups = find_teleego_groups(DATA_ROOT, QAS_SUBDIR, VIDEOS_SUBDIR)
    if not groups:
        raise RuntimeError("No TeleeGo groups found. Expect QAs/merged_*.json and matching videos/<KEY>/.")

    # Load model/tokenizer once
    model = AutoModel.from_pretrained(
        MODEL_DIR,
        trust_remote_code=True,
        attn_implementation="sdpa",
        torch_dtype=torch.bfloat16,
    )
    model = model.eval().cuda()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)
    try:
        model.init_tts()
    except Exception:
        pass
    sys_msg = model.get_sys_prompt(mode='omni', language='zh')

    agg: List[Dict[str, Any]] = []

    for key, qa_path, videos in groups:
        print(f"===== Group {key}: {len(videos)} videos =====")
        with open(qa_path, "r", encoding="utf-8") as f:
            qa_items: List[Dict[str, Any]] = json.load(f)
        # TeleeGo strict: use the same QA items for every video in this folder
        for vpath in tqdm.tqdm(videos, desc=f"{key} videos"):
            pair_dir = save_root / key / vpath.stem
            os.makedirs(pair_dir, exist_ok=True)
            tmp_qa = pair_dir / "_tmp_video_qa.json"
            with open(tmp_qa, "w", encoding="utf-8") as f:
                json.dump(qa_items, f, ensure_ascii=False, indent=2)
            summ = evaluate_single(str(vpath), str(tmp_qa), str(pair_dir), model, tokenizer, sys_msg, timeline_path=TIMELINE_JSON_PATH)
            agg.append(summ)

    # Save aggregate summary across all groups
    os.makedirs(save_root, exist_ok=True)
    aggregate_path = save_root / "aggregate_results.json"
    with open(aggregate_path, "w", encoding="utf-8") as f:
        json.dump({"pairs": agg}, f, ensure_ascii=False, indent=2)
    print(f"Aggregate saved to: {aggregate_path}")
    # --- 简单 print 方式 ---
    print(f"[INFO] Using VIDEO_PATH: {VIDEO_PATH}")
    print(f"[INFO] Saving to SAVE_PRED_PATH: {SAVE_PRED_PATH}")
    
    return {"pairs": agg, "aggregate_path": str(aggregate_path)}


def main():
    """Mode selection priority:
    1) TeleeGo dataset mode if DATA_ROOT is set
    2) Generic folder mode if VIDEO_DIR & JSON_DIR set
    3) Single pair mode otherwise
    """
    if DATA_ROOT:
        run_teleego_mode()
        return
    folder_mode = bool(VIDEO_DIR and JSON_DIR)
    if folder_mode:
        run_batch_mode()
    else:
        run_single_mode()


if __name__ == "__main__":
    main()
