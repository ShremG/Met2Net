# Met2Net: A Decoupled Two-Stage Spatio-Temporal Forecasting Model for Complex Meteorological Systems

🏆 **ICCV 2025 Accepted Paper**  · 🛰️ Spatio-Temporal Forecasting · ⏳ Two-Stage Learning

[![arXiv](https://img.shields.io/badge/arXiv-2507.17189-b31b1b.svg)](https://arxiv.org/pdf/2507.17189)
[![HF Daily Paper](https://img.shields.io/badge/HuggingFace-DailyPaper-yellow)](https://huggingface.co/papers/2507.17189)
[![License](https://img.shields.io/badge/license-Apache--2.0-green.svg)](LICENSE)
[![Stars](https://img.shields.io/github/stars/ShremG/Met2Net?style=social)](https://github.com/ShremG/Met2Net/stargazers)

---
## 🆕 Updates

- **[June 26, 2025]** 🎉 Met2Net accepted to **ICCV 2025**! Paper available on arXiv: [2507.17189](https://arxiv.org/pdf/2507.17189)
---
## 📂 Datasets & Models Overview

We release both the processed datasets and pretrained model weights on Hugging Face for full reproducibility:

| Dataset Name     | Variables                | Shape (C×H×W) | Seq (Input→Output) | Samples (Train/Test) | Dataset Repo | Model Repo |
|------------------|--------------------------|----------------|--------------------|-----------------------|--------------|------------|
| **ERA5-Cropped** | T2M, U10, V10, MSL       | 4×128×128      | 12 → 12            | 43,801 / 8,737        | [📂 HF Dataset](https://huggingface.co/datasets/guaishou1/Met2Net) | [🧠 HF Model](https://huggingface.co/guaishou1/Met2Net) |
| **MvMm-FNIST**   | 3 synthetic channels     | 3×64×64        | 10 → 10            | 10,000 / 10,000       | [📂 HF Dataset](https://huggingface.co/datasets/guaishou1/Met2Net)     | [🧠 HF Model](https://huggingface.co/guaishou1/Met2Net) |


---

## 🚀 Getting Started

### 1. Clone and setup environment

```bash
git clone https://github.com/ShremG/Met2Net.git
cd Met2Net
bash create_env.sh
```

### 2. Run training
```bash
bash run.sh
```


---
## 📄 Citation

If you find this project useful in your research, please cite:

```bibtex
@inproceedings{li2025met2net,
  title     = {Met2Net: A Decoupled Two-Stage Spatio-Temporal Forecasting Model for Complex Meteorological Systems},
  author    = {Shaohan Li and Hao Yang and Min Chen and Xiaolin Qin},
  booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
  year      = {2025},
  eprint    = {2507.17189},
  archivePrefix = {arXiv},
  primaryClass = {cs.CV}
}
```

---

## 📬 Contact

For questions, collaborations, or implementation issues, please open a GitHub issue or contact the corresponding author.

---

## 🙏 Acknowledgements

This work is built upon foundations from [OpenSTL](https://github.com/chengtan9907/OpenSTL).  
We thank the open-source community for enabling reproducible weather forecasting research.

---