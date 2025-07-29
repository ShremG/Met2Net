# Met2Net: A Decoupled Two-Stage Spatio-Temporal Forecasting Model for Complex Meteorological Systems

🏆 **ICCV 2025 Accepted Paper**  · 🛰️ Spatio-Temporal Forecasting · ⏳ Two-Stage Learning

[![arXiv](https://img.shields.io/badge/arXiv-2507.17189-b31b1b.svg)](https://arxiv.org/pdf/2507.17189)
[![License](https://img.shields.io/badge/license-Apache--2.0-green.svg)](LICENSE)
[![Stars](https://img.shields.io/github/stars/ShremG/Met2Net?style=social)](https://github.com/ShremG/Met2Net/stargazers)

---
## 🆕 Updates

- **[June 26, 2025]** 🎉 Met2Net accepted to **ICCV 2025**! Paper available on arXiv: [2507.17189](https://arxiv.org/pdf/2507.17189)
---
## 🌐 Overview

**Met2Net** is a representation-decoupled spatio-temporal forecasting framework tailored for complex meteorological systems. It introduces a **two-stage pipeline**:

- **Stage 1:** Learn variable-specific representations using independent encoder-decoder pairs.
- **Stage 2:** Enable cross-variable translation via a lightweight Transformer-based Translator to model interactions.

This approach effectively separates feature learning from interaction modeling, improving both accuracy and generalization.

---

## 📊 Benchmarks

Met2Net is evaluated on standard weather forecasting datasets including ERA5 and WeatherBench, covering multi-variable 3D gridded fields over time.

| Dataset      | Variables                  | Resolution    | Horizon        | Download         |
|--------------|----------------------------|---------------|----------------|------------------|
| ERA5-Cropped | T2M, UV10, R, TCC          | 128×128       | 24h, 48h, 72h   | [Link](#)        |
| WeatherBench | T850, Z500, U10, V10       | 5.625°        | 24h – 120h     | [Link](#)        |

> 📌 For detailed training and evaluation settings, please refer to [`configs/`](configs/)

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