<div align="center">
  <h1>
    TeleEgo: <br> 
    Benchmarking Egocentric AI Assistants in the Wild
  </h1>

  <!-- 项目徽章 -->
  <p>
    <a href="https://huggingface.co/datasets/David0219/TeleEgo">
      <img alt="Hugging Face" src="https://img.shields.io/badge/HuggingFace-Dataset-orange">
    </a>
    <a href="https://arxiv.org/abs/2510.23981">
      <img alt="arXiv" src="https://img.shields.io/badge/ArXiv-2510.23981-b31b1b.svg">
    </a>
    <a href="https://programmergg.github.io/jrliu.github.io/">
      <img alt="Page" src="https://img.shields.io/badge/Project Page-Link-green">
    </a>
  </p>

  <img src="assets/teaser.png" alt="Teaser" style="width:80%; max-width:700px;">
  
  📢 **Note**：This project is still under active development, and the benchmark will be continuously maintained.  
</div>



## 📌 Introduction

**TeleEgo** is a comprehensive **omni benchmark** designed for **multi-person, multi-scene, multi-task, and multimodal long-term memory reasoning** in egocentric video streams.
It reflects realistic personal assistant scenarios where continuous egocentric video data is collected across hours or even days, requiring models to maintain and reason over **memory, understanding, and cross-memory reasoning**. **Omni** here means that TeleEgo covers the full spectrum of **roles, scenes, tasks, modalities, and memory horizons**, offering all-round evaluation for egocentric AI assistants.

**TeleEgo provides:**

- 🧠 **Omni-scale, diverse egocentric data** from 5 roles across 4 daily scenarios.
- 🎤 **Multi-modal annotations**: video, narration, and speech transcripts.
- ❓ **Fine-grained QA benchmark**: 3 cognitive dimensions, 12 subcategories.


---

## 📊 Dataset Overview

- **Participants**: 5 (balanced gender)
- **Scenarios**:
  - Work & Study
  - Lifestyle & Routines
  - Social Activities
  - Outings & Culture
- **Recording**: 3 days/participant (~14.4 hours each)
- **Modalities**:
  - Egocentric video streams
  - Speech & conversations
  - Narration and event descriptions

---

## 🧪 Benchmark Tasks

TeleEgo-QA evaluates models along **three main dimensions**:

1. **Memory**
   - Short-term / Long-term / Ultra-long Memory
   - Entity Tracking
   - Temporal Comparison & Interval

2. **Understanding**
   - Causal Understanding
   - Intent Inference
   - Multi-step Reasoning
   - Cross-modal Understanding

3. **Cross-Memory Reasoning**
   - Cross-temporal Causality
   - Cross-entity Relation
   - Temporal Chain Understanding

Each QA instance includes:

- Question type: Single-choice, Multi-choice, Binary, Open-ended

---

<!-- ## Key Advantages over Existing Benchmarks

* **Compared with EgoLife**: TeleEgo offers **omni-scenario coverage** (not restricted to a single shared environment), broader task diversity, fine-grained memory categories, multi-task trajectories, and difficulty levels.
* **Compared with M3-Agent / HourVideo**: TeleEgo is explicitly **omni-task and omni-modal**, focusing on **diagnostic memory evaluation**, cross-event reasoning, and multimodal grounding in **real-life egocentric settings**.

--- -->

## 🗂️ Repository Structure

```
TeleEgo/
│
├── teleego_data/                # Dataset samples / metadata (link provided separately)
├── weights/                     # Pre-trained weights (MiniCPM-o, Qwen2.5-Omni, ...)
├── TeleEgo_gemini25_pro_eval.py # Evaluation scripts
├── TeleEgo_gpt4o_eval.py        # Evaluation scripts
├── TeleEgo_minicpm_eval.py      # Evaluation scripts
├── TeleEgo_qwen25_eval.py       # Evaluation scripts
├── TeleEgo_qweno25_eval.py      # Evaluation scripts
├── TeleEgo_videochat_eval.py    # Evaluation scripts
└── README.md                    # This file
```

## 🚀 Usage

### 📥 Dataset Access

Due to privacy and licensing constraints, please request access here:
👉 [**Dataset Access Form**](https://huggingface.co/datasets/David0219/TeleEgo)

### 🧪 Running Evaluations

```bash
python TeleEgo_gpt4o_eval.py
```

---

<!-- ## Baselines
![Baseline 1](assets/res1.png)
![Baseline 2](assets/res2.png)
---

## 🤝 Collaborators

Thanks to these amazing people for contributing to the project:

<a href="https://github.com/rebeccaeexu">
  <img src="https://avatars.githubusercontent.com/rebeccaeexu" width="60px" style="border-radius:50%" />
</a>
<a href="https://github.com/DavisWANG0">
  <img src="https://avatars.githubusercontent.com/DavisWANG0" width="60px" style="border-radius:50%" />
</a>
<a href="https://github.com/H-oliday">
  <img src="https://avatars.githubusercontent.com/H-oliday" width="60px" style="border-radius:50%" />
</a>
<a href="https://github.com/Xiaolong-RRL">
  <img src="https://avatars.githubusercontent.com/Xiaolong-RRL" width="60px" style="border-radius:50%" />
</a>
<a href="https://github.com/Programmergg">
  <img src="https://avatars.githubusercontent.com/Programmergg" width="60px" style="border-radius:50%" />
</a>
<a href="https://github.com/yiheng-wang-duke">
  <img src="https://avatars.githubusercontent.com/yiheng-wang-duke" width="60px" style="border-radius:50%" />
</a>
<a href="https://github.com/cocowy1">
  <img src="https://avatars.githubusercontent.com/cocowy1" width="60px" style="border-radius:50%" />
</a>
<a href="https://github.com/chxy95">
  <img src="https://avatars.githubusercontent.com/chxy95" width="60px" style="border-radius:50%" />
</a> -->


## 📜 Citation

If you find our **TeleEgo** in your research, please cite:

```bibtex
@misc{yan2025teleegobenchmarkingegocentricai,
      title={TeleEgo: Benchmarking Egocentric AI Assistants in the Wild}, 
      author={Jiaqi Yan and Ruilong Ren and Jingren Liu and Shuning Xu and Ling Wang and Yiheng Wang and Yun Wang and Long Zhang and Xiangyu Chen and Changzhi Sun and Jixiang Luo and Dell Zhang and Hao Sun and Chi Zhang and Xuelong Li},
      year={2025},
      eprint={2510.23981},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2510.23981}, 
}
```

## 🪪 License

This project is licensed under the **MIT License**.
Dataset usage is restricted under a **research-only license**.

---

<!-- ## References

* EgoLife: Towards Egocentric Life Assistant [\[arXiv:2503.03803\]](https://arxiv.org/abs/2503.03803)
* M3-Agent: Seeing, Listening, Remembering, and Reasoning [\[arXiv:2508.09736\]](https://arxiv.org/abs/2508.09736)
* HourVideo: 1-Hour Video-Language Understanding [\[arXiv:2411.04998\]](https://arxiv.org/abs/2411.04998) -->


## 📬 Contact

If you have any questions, please feel free to reach out: chxy95@gmail.com.

---

<div align="center">

<strong>✨ TeleEgo is an Omni benchmark, a step toward building AI assistants with true long-term memory and memory reasoning in real-world wearable scenarios. ✨</strong>

</div>

<!-- <br/> -->

<div align="center" style="margin-top: 10px;">
  <img src="assets/TeleAI.jpg" alt="TeleAI Logo" width="120px" />
  &nbsp;&nbsp;&nbsp;
  <img src="assets/TeleEgo.png" alt="TeleEgo Logo" width="120px" />
</div>
