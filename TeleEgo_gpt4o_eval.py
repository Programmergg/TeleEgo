import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import re
import io
import tqdm
import json
import math
import base64
import librosa
import tempfile
import numpy as np
from PIL import Image
from pathlib import Path
from moviepy import VideoFileClip
from typing import List, Dict, Any, Tuple, Optional

# =================== Config ===================
AZURE_OPENAI_API_KEY      = os.environ.get("AZURE_OPENAI_API_KEY", "")
AZURE_OPENAI_ENDPOINT     = os.environ.get("AZURE_OPENAI_ENDPOINT", "")
AZURE_OPENAI_API_VERSION  = os.environ.get("AZURE_OPENAI_API_VERSION", "2024-08-01-preview")
AZURE_OPENAI_DEPLOYMENT   = os.environ.get("AZURE_OPENAI_DEPLOYMENT", "4o")

P_NUM = os.getenv("P_NUM", "1")      # 只要把这里改成 "1" / "2" / "3" ... 即可；也可用环境变量 P_NUM 覆盖
QA_SUFFIX = os.getenv("QA_SUFFIX", "A")  # 如果你的 QA 文件有后缀（如 merged_P5_A.json），也单独抽出来

BASE_DIR = "./teleego_data"

# === Paths ===
VIDEO_PATH: str = f"{BASE_DIR}/video_merged/merged_P{P_NUM}.mp4"
JSON_PATH: str  = f"{BASE_DIR}/QAs/merged_P{P_NUM}_{QA_SUFFIX}.json"
SAVE_PRED_PATH: str = f"{BASE_DIR}/outputs/gpt4o/p{P_NUM}/eval_predictions_P{P_NUM}.json"
TIMELINE_JSON_PATH: str = f"{BASE_DIR}/video_merged/timeline_P{P_NUM}.json"

# --- 简单 print 方式 ---
print(f"[INFO] Using VIDEO_PATH: {VIDEO_PATH}")
print(f"[INFO] Saving to SAVE_PRED_PATH: {SAVE_PRED_PATH}")


VIDEO_DIR: Optional[str] = None
JSON_DIR: Optional[str]  = None
SAVE_DIR: Optional[str]  = None

DATA_ROOT: Optional[str] = None
QAS_SUBDIR: str = "QAs"
VIDEOS_SUBDIR: str = "videos"
SAVE_ROOT: Optional[str] = None

USE_TTS = False
OUTPUT_AUDIO_BASENAME = "output.wav"
TEMPERATURE = 0.2
MAX_NEW_TOKENS = 2048
SEED = 42
RECALL_DELAY_SEC = 60.0
MAX_RECALL_ROUNDS = 10

# =================== Helpers ===================
def _lower_safe(x: Optional[str]) -> str:
    if x is None:
        return ""
    return str(x).strip().lower()

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
                return parse_hhmmss(s)
        except Exception:
            pass
        return None

    def find_segment_for_offset(self, merged_sec: float) -> Optional[Dict[str, Any]]:
        for seg in self.segments:
            if seg["merged_offset_start_seconds"] <= merged_sec <= seg["merged_offset_end_seconds"]:
                return seg
        return None

def build_all_units(video_path: str):
    video = VideoFileClip(video_path, fps_source="fps")
    safe_dur = max(0.0, float(video.duration) - 0.05)
    if hasattr(video, "subclipped"):
        video = video.subclipped(0, safe_dur)
    elif hasattr(video, "subclip"):
        video = video.subclip(0, safe_dur)
    else:
        raise AttributeError("VideoFileClip has neither 'subclipped' nor 'subclip'.")
    
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tf:
        wav_path = tf.name
        video.audio.write_audiofile(wav_path, codec="pcm_s16le", fps=16000)
        audio_np, sr = librosa.load(wav_path, sr=16000, mono=True)
    n_units = int(math.ceil(video.duration))
    units = []
    for i in range(n_units):
        t = min(i + 1, max(0.0, video.duration - 1e-3))
        frame = video.get_frame(t)
        image = Image.fromarray(frame.astype(np.uint8))
        audio = audio_np[sr * i : sr * (i + 1)]
        units.append((image, audio))
    return units, float(video.duration), sr

def unit_index_for_time(t_end: float, total_units: int) -> int:
    if total_units <= 0:
        return 0
    if not math.isfinite(t_end):
        return total_units - 1
    idx = int(math.ceil(max(0.0, t_end))) - 1
    return max(0, min(idx, total_units - 1))

def build_single_unit_contents(units, idx):
    idx = max(0, min(idx, len(units) - 1))
    image, audio = units[idx]
    return image, audio

def build_question_prompt(item: Dict[str, Any]) -> str:
    qtype = _lower_safe(item.get("QA_type"))
    question = str(item.get("question", "")).strip()
    options: List[str] = item.get("options", [])

    if qtype == "mc_single":
        opt_str = "\n".join(options)
        instr = "请根据提供的图像帧与（可用时的）音频内容回答一个单选题。\n只输出选项字母（例如 A），不要输出解释。\n"
        return f"{instr}\n题目：{question}\n选项：\n{opt_str}\n你的答案："
    if qtype == "mc_multi":
        opt_str = "\n".join(options)
        instr = "请根据提供的图像帧与（可用时的）音频内容回答一个多选题。\n只输出所有正确选项字母，使用英文逗号分隔（例如 A,B），不要输出解释。\n"
        return f"{instr}\n题目：{question}\n选项：\n{opt_str}\n你的答案："
    if qtype == "binary":
        instr = "请判断对错（True/False）。\n只输出 True 或 False，不要输出其他字符。\n"
        return f"{instr}\n命题：{question}\n你的答案："
    instr = "请简要作答本题。优先给出关键词序列或简短句子。\n在20个字以内为佳。\n"
    return f"{instr}\n问题：{question}\n你的答案："

def normalize_letters(s: str) -> List[str]:
    letters = re.findall(r"[A-Z]", s.upper())
    return letters

def parse_prediction(text: str, qtype: str):
    t = (text or "").strip()
    qtype = _lower_safe(qtype)
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


def collect_questionstamps(qa_items: List[Dict[str, Any]], label_mapper: Optional[Any] = None):
    rows = []
    for pos, it in enumerate(qa_items):
        ts = (it.get("evidence") or {}).get("timestep") or {}
        end_raw = ts.get("end") or ts.get("End") or ts.get("to")
        t_end = None
        if label_mapper is not None:
            try: t_end = label_mapper(end_raw)
            except Exception: t_end = None
        if t_end is None:
            try:
                if end_raw is None: t_end = float("inf")
                elif isinstance(end_raw, (int, float)): t_end = float(end_raw)
                else:
                    s = str(end_raw).strip()
                    t_end = parse_hhmmss(s) if ":" in s else float(s)
            except Exception:
                t_end = float("inf")
        rows.append((t_end, pos, it))
    rows.sort(key=lambda r: (r[0], r[1]))
    grouped = [(t_end, [it]) for (t_end, _, it) in rows]
    return grouped

def set_global_seed(seed: int = 42):
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as _np; _np.random.seed(seed)
    except Exception: pass
    try:
        import random as _random; _random.seed(seed)
    except Exception: pass

def pil_to_base64_png(image: Image.Image) -> str:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{b64}"

def call_gpt4o(messages, temperature=TEMPERATURE, max_tokens=MAX_NEW_TOKENS) -> str:
    from openai import AzureOpenAI
    client = AzureOpenAI(
        api_key=AZURE_OPENAI_API_KEY,
        api_version=AZURE_OPENAI_API_VERSION,
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
    )
    resp = client.chat.completions.create(
        model=AZURE_OPENAI_DEPLOYMENT,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return (resp.choices[0].message.content or "").strip()

def run_gpt4o_round(image: Image.Image, prompt_text: str, extra_system_msgs: Optional[List[Dict[str, Any]]] = None):
    base_sys = {"role": "system", "content": "你是一个多模态助手。你会结合提供的图像帧和文本指令作答；严格遵循输出格式要求。"}
    msgs = [base_sys]
    if extra_system_msgs:
        msgs.extend(extra_system_msgs)
    img_b64 = pil_to_base64_png(image)
    user_content = [
        {"type": "text", "text": "以下是一帧来自视频的图像。请结合题目指令作答。"},
        {"type": "image_url", "image_url": {"url": img_b64}},
        {"type": "text", "text": prompt_text},
    ]
    msgs.append({"role": "user", "content": user_content})
    try:
        text = call_gpt4o(msgs)
    except Exception as e:
        text = f"<ERROR: {e}>"
    return text, None

def evaluate_single(video_path: str, json_path: str, save_dir: str, timeline_path: Optional[str] = None) -> Dict[str, Any]:
    set_global_seed(SEED)
    assert os.path.exists(json_path), f"QA json not found: {json_path}"
    assert os.path.exists(video_path), f"Video not found: {video_path}"
    timeline_idx = None
    if timeline_path:
        try: timeline_idx = TimelineIndex.from_json(timeline_path)
        except Exception as _e: print(f"⚠️ Failed to load timeline: {timeline_path} -> {_e}")
    with open(json_path, "r", encoding="utf-8") as f:
        qa_items: List[Dict[str, Any]] = json.load(f)

    groups = collect_questionstamps(qa_items, label_mapper=(timeline_idx.label_to_merged_seconds if timeline_idx else None))
    max_t = 0.0 if not groups else max([t for t, _ in groups if math.isfinite(t)] + [0.0])
    units, duration, sr = build_all_units(video_path)
    _ = int(math.ceil(max_t)) if math.isfinite(max_t) else len(units)

    os.makedirs(save_dir, exist_ok=True)
    audio_base = str(Path(save_dir) / OUTPUT_AUDIO_BASENAME)

    results = []
    type_stats: Dict[str, Dict[str, int]] = {}
    category_stats: Dict[str, Dict[str, int]] = {}
    cat_sub_stats: Dict[Tuple[str, str], Dict[str, int]] = {}
    correct_for_recall: List[Tuple[float, Dict[str, Any]]] = []

    announced_videos = set()

    for t_end, items in tqdm.tqdm(groups, desc="QA by questionstamp"):
        idx = unit_index_for_time(t_end, len(units))
        image, audio_np = build_single_unit_contents(units, idx)

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
            qtype = _lower_safe(item.get("QA_type"))
            prompt = build_question_prompt(item)

            try:
                pred_text, saved_audio = run_gpt4o_round(image=image, prompt_text=prompt, extra_system_msgs=extra_msgs)
            except Exception as e:
                pred_text, saved_audio = f"<ERROR: {e}>", None

            parsed = parse_prediction(pred_text, qtype)
            eval_res = evaluate_item(item, parsed)

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
                "llm_score": None,
            }
            if "overlap_tokens" in eval_res:
                rec["overlap_tokens"] = eval_res["overlap_tokens"]
            results.append(rec)

            tkey = _lower_safe(rec.get("QA_type"))
            s = type_stats.setdefault(tkey, {"n": 0, "ok": 0})
            s["n"] += 1
            s["ok"] += int(rec["correct"]) if rec["metric"] != "token_overlap>=25%" else 0

            if rec["metric"] != "token_overlap>=25%":
                cat = str(rec.get("category") or "unknown")
                sub = str(rec.get("subcategory") or "unknown")
                s1 = category_stats.setdefault(cat, {"n": 0, "ok": 0})
                s1["n"] += 1; s1["ok"] += int(rec["correct"])
                s2 = cat_sub_stats.setdefault((cat, sub), {"n": 0, "ok": 0})
                s2["n"] += 1; s2["ok"] += int(rec["correct"])

            if rec["correct"] and math.isfinite(t_end):
                correct_for_recall.append((t_end, item))

            if tkey in {"mc_single", "mc_multi", "binary"}:
                print(f"[t={t_end:.2f}s][#{rec['index']}] {tkey} | correct={rec['correct']} | pred={rec['parsed_pred']} | raw={rec['pred_text'][:120]!r}")
            else:
                print(f"[t={t_end:.2f}s][#{rec['index']}] open_ended | raw={rec['pred_text'][:120]!r}")

    print("===== Summary (initial) =====")
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

    print("===== Accuracy by category (initial, choice+binary only) =====")
    for k, v in category_stats.items():
        acc = v["ok"] / max(1, v["n"])
        print(f"  {k}: {v['ok']}/{v['n']} = {acc:.3f}")

    print("===== Accuracy by (category, subcategory) (initial, choice+binary only) =====")
    for (ck, sk), v in cat_sub_stats.items():
        acc = v["ok"] / max(1, v["n"])
        print(f"  {ck} / {sk}: {v['ok']}/{v['n']} = {acc:.3f}")

    # Recall rounds (type + category + subcategory)
    recall_round_type_stats: Dict[int, Dict[str, Dict[str, int]]] = {}
    recall_round_cat_stats: Dict[int, Dict[str, Dict[str, int]]] = {}
    recall_round_cat_sub_stats: Dict[int, Dict[str, Dict[str, int]]] = {}

    if RECALL_DELAY_SEC > 0 and correct_for_recall:
        active: List[Tuple[float, Dict[str, Any]]] = list(correct_for_recall)
        round_id = 1
        while active and round_id <= MAX_RECALL_ROUNDS:
            events: List[Tuple[float, Dict[str, Any]]] = []
            for t_end, item in active:
                recall_t = t_end + round_id * RECALL_DELAY_SEC
                events.append((recall_t, item))
            events.sort(key=lambda x: x[0])

            kept_for_next: List[Tuple[float, Dict[str, Any]]] = []
            print(f"===== Recall round {round_id} (+{round_id*RECALL_DELAY_SEC}s) =====")

            typemap = recall_round_type_stats.setdefault(round_id, {})
            catmap = recall_round_cat_stats.setdefault(round_id, {})
            catsubmap = recall_round_cat_sub_stats.setdefault(round_id, {})

            for recall_t, item in tqdm.tqdm(events, desc=f"Recall r{round_id}"):
                idx = unit_index_for_time(recall_t, len(units))
                image, audio_np = build_single_unit_contents(units, idx)

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

                qtype = _lower_safe(item.get("QA_type"))
                prompt = build_question_prompt(item)

                try:
                    pred_text, saved_audio = run_gpt4o_round(image=image, prompt_text=prompt, extra_system_msgs=extra_msgs)
                except Exception as e:
                    pred_text, saved_audio = f"<ERROR: {e}>", None

                parsed = parse_prediction(pred_text, qtype)
                eval_res = evaluate_item(item, parsed)

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
                    "llm_score": None,
                }
                if "overlap_tokens" in eval_res:
                    rec2["overlap_tokens"] = eval_res["overlap_tokens"]
                results.append(rec2)

                qtype_key = _lower_safe(rec2.get("QA_type"))
                tstats = typemap.setdefault(qtype_key, {"n": 0, "ok": 0})
                tstats["n"] += 1
                if rec2["metric"] != "token_overlap>=25%":
                    tstats["ok"] += int(rec2["correct"])
                    cat = str(rec2.get("category") or "unknown")
                    sub = str(rec2.get("subcategory") or "unknown")
                    cstat = catmap.setdefault(cat, {"n": 0, "ok": 0})
                    cstat["n"] += 1; cstat["ok"] += int(rec2["correct"])
                    csstat = catsubmap.setdefault(f"{cat}||{sub}", {"n": 0, "ok": 0})
                    csstat["n"] += 1; csstat["ok"] += int(rec2["correct"])

                if _lower_safe(rec2.get("QA_type")) in {"mc_single","mc_multi","binary"}:
                    print(f"[RECALL r={round_id} t={recall_t:.2f}s][#{rec2['index']}] {rec2['QA_type']} | correct={rec2['correct']} | pred={rec2['parsed_pred']} | raw={rec2['pred_text'][:120]!r}")
                else:
                    print(f"[RECALL r={round_id} t={recall_t:.2f}s][#{rec2['index']}] open_ended | raw={rec2['pred_text'][:120]!r}")

                if rec2["correct"]:
                    orig_t_end = recall_t - round_id * RECALL_DELAY_SEC
                    kept_for_next.append((orig_t_end, item))

            active = kept_for_next
            round_id += 1

    out_path = Path(save_dir) / "eval_predictions.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "results": results,
            "stats": {
                "initial": {
                    "type_stats": type_stats,
                    "category_stats": category_stats,
                    "category_subcategory_stats": {f"{k[0]}||{k[1]}": v for k, v in cat_sub_stats.items()},
                    "overall_acc_choice_binary": (sum(v["ok"] for v in type_stats.values()) / max(1, sum(v["n"] for k, v in type_stats.items() if k in {"mc_single","mc_multi","binary"}))),
                },
                "recall_rounds": {
                    "per_round_type_stats": recall_round_type_stats,
                    "per_round_category_stats": recall_round_cat_stats,
                    "per_round_category_subcategory_stats": recall_round_cat_sub_stats,
                }
            }
        }, f, ensure_ascii=False, indent=2)
    print(f"Saved to: {out_path}")
    return {"video": str(video_path), "qa_json": str(json_path), "out": str(out_path)}

VIDEO_EXTS = {".mp4",".mov",".mkv",".avi",".mpg",".mpeg",".m4v"}

def find_pairs(video_dir: str, json_dir: str):
    vdir = Path(video_dir); jdir = Path(json_dir)
    videos = [p for p in vdir.rglob("*") if p.suffix.lower() in VIDEO_EXTS]
    jsons  = {p.stem.lower(): p for p in jdir.rglob("*.json")}
    pairs = []
    for v in sorted(videos):
        key = v.stem.lower()
        if key in jsons: pairs.append((v, jsons[key]))
    return pairs

def run_single_mode():
    save_dir = str(Path(SAVE_PRED_PATH).parent)
    return evaluate_single(VIDEO_PATH, JSON_PATH, save_dir, timeline_path=TIMELINE_JSON_PATH)

def run_batch_mode():
    pairs = find_pairs(VIDEO_DIR, JSON_DIR)
    agg = []
    for vid_path, qa_path in tqdm.tqdm(pairs, desc="Pairs"):
        vid_stem = vid_path.stem
        pair_save_dir = str(Path(SAVE_DIR) / vid_stem)
        agg.append(evaluate_single(str(vid_path), str(qa_path), pair_save_dir, timeline_path=TIMELINE_JSON_PATH))
    os.makedirs(SAVE_DIR, exist_ok=True)
    aggregate_path = Path(SAVE_DIR) / "aggregate_results.json"
    with open(aggregate_path, "w", encoding="utf-8") as f:
        json.dump({"pairs": agg}, f, ensure_ascii=False, indent=2)
    print(f"Aggregate saved to: {aggregate_path}")
    return {"pairs": agg, "aggregate_path": str(aggregate_path)}

def run_teleego_mode():
    assert DATA_ROOT, "TeleeGo mode requires DATA_ROOT"
    save_root = Path(SAVE_ROOT or (Path(DATA_ROOT) / "results"))
    # (Implementation omitted: mirror batch semantics if needed.)
    raise NotImplementedError

def main():
    if DATA_ROOT:
        run_teleego_mode(); return
    folder_mode = bool(VIDEO_DIR and JSON_DIR)
    if folder_mode: run_batch_mode()
    else: run_single_mode()

if __name__ == "__main__":
    main()
