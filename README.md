# Temporal Memory for Robust Navigation under Distribution Shift

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23256938.svg)](https://doi.org/10.5281/zenodo.23256938)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3119/)
[![arXiv](https://img.shields.io/badge/arXiv-XXXX.XXXXX-b31b1b.svg)](https://arxiv.org/abs/XXXX.XXXXX)

A controlled, CPU-friendly benchmark for studying the role of **temporal memory** in **robust mobile robot navigation under distribution shift**.

This repository contains the complete code, trained models, evaluation scripts, and paper source for our study comparing a memoryless MLP policy (PPO) against a recurrent LSTM policy (RecurrentPPO) on a goal-occlusion navigation task.

---

## 🔑 Key Result

We evaluate both policies across three test conditions of increasing difficulty. Under a severe, unseen distribution shift, the recurrent LSTM policy degrades by only **8 percentage points** from its training distribution, while the memoryless MLP collapses by **30 points**.

| Policy | No DR | Mild DR | Severe DR |
|--------|-------|---------|-----------|
| MLP (PPO) | 20.0% | 35.0% | 5.0% |
| **LSTM (RecurrentPPO)** | **33.0%** | 29.0% | 2.0% |

See the paper for full analysis and discussion of limitations.

---

## 📄 Paper

- **PDF**: [`paper/main.pdf`](paper/main.pdf)
- **LaTeX source**: [`paper/main.tex`](paper/main.tex)
- **arXiv**: [arXiv:XXXX.XXXXX](https://arxiv.org/abs/XXXX.XXXXX)

If you use this work, please cite it (see [Citation](#-citation)).

---

## ⚙️ Installation

**Requirements:** Python 3.11 (Python 3.12 may work, but 3.13/3.14 are not currently supported by PyTorch).

```bash
git clone https://github.com/YOUR_USERNAME/sim2real-navigation.git
cd sim2real-navigation

python -m venv venv
# Windows:
venv\Scripts\activate
# Linux / macOS:
source venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
```

---

## 🚀 Quick Start

### Train both policies

```bash
# Memoryless MLP baseline (1M timesteps, ~40 minutes on CPU)
python train.py --policy mlp --no-dr --timesteps 1000000

# Recurrent LSTM policy (1.5M timesteps, ~5 hours on CPU)
python train.py --policy lstm --timesteps 1500000
```

### Evaluate under all three conditions

```bash
# MLP
python evaluate.py --model models/ppo_mlp_nodr --vecnorm models/ppo_mlp_nodr_vecnorm.pkl --episodes 100 --out results/eval_mlp_nodr.csv
python evaluate.py --model models/ppo_mlp_nodr --vecnorm models/ppo_mlp_nodr_vecnorm.pkl --episodes 100 --domain-randomization --out results/eval_mlp_dr.csv
python evaluate.py --model models/ppo_mlp_nodr --vecnorm models/ppo_mlp_nodr_vecnorm.pkl --episodes 100 --severe --out results/eval_mlp_severe.csv

# LSTM (same three commands, different model path)
python evaluate.py --model models/ppo_lstm_dr --vecnorm models/ppo_lstm_dr_vecnorm.pkl --episodes 100 --out results/eval_lstm_nodr.csv
python evaluate.py --model models/ppo_lstm_dr --vecnorm models/ppo_lstm_dr_vecnorm.pkl --episodes 100 --domain-randomization --out results/eval_lstm_dr.csv
python evaluate.py --model models/ppo_lstm_dr --vecnorm models/ppo_lstm_dr_vecnorm.pkl --episodes 100 --severe --out results/eval_lstm_severe.csv
```

### Reproduce the paper figures

```bash
python export_tb.py      # TensorBoard logs → paper_data/*.csv
python make_figures.py   # paper_data/*.csv → figures/*.pdf and *.png
```

---

## 📊 Environment

The task is a 2D navigation problem where a differential-drive robot must reach a goal in an **8×8 m** arena with **5 circular obstacles** (radius 0.4 m).

**Observation (23-dim):**
- Robot state: `(x, y, cosθ, sinθ)` — 4 dims
- Goal info: `(dx, dy, dist)` — 3 dims
- LiDAR: 16 beams, max range 4 m — 16 dims

**Action:** continuous `[linear_vel, angular_vel]` in `[-1, 1]`.

**Key design feature — goal occlusion.** When any obstacle blocks the line of sight between the robot and the goal, the goal information is zeroed out. The policy must therefore remember the last visible goal direction to keep navigating. This is what makes temporal memory a genuinely useful capability rather than a decorative addition.

---

## 🧠 Policies

| | MLP Baseline | LSTM Policy |
|---|---|---|
| Algorithm | PPO (SB3) | RecurrentPPO (sb3-contrib) |
| Memory | None (single frame) | Hidden state across timesteps |
| Hidden layers | 2 × 128 | 2 × 64 + LSTM(128) |
| Training steps | 1,000,000 | 1,500,000 |
| Learning rate | 3e-4 | 1e-4 |
| Rollout length | 2048 | 2048 |

---

## 🧪 Evaluation Conditions

| Condition | Obstacles | Obstacle speed | LiDAR noise | Action latency | Goal noise |
|---|---|---|---|---|---|
| **No DR** | 5 (static) | 0.0 m/s | 0 | 0 | 0 |
| **Mild DR** | 5 | up to 0.4 m/s | up to 0.08 | up to 4 steps | up to 0.15 |
| **Severe DR** | 10 | up to 1.5 m/s | up to 0.18 | up to 8 steps | up to 0.35 |

Severe DR is never seen during training — it is a true distribution shift.

---

## 📈 Full Results

### Success rate (%)

| Policy | No DR | Mild DR | Severe DR |
|---|---|---|---|
| MLP | 20.0 | 35.0 | 5.0 |
| LSTM | 33.0 | 29.0 | 2.0 |

### Degradation from training distribution (mild) to severe

| Policy | Δ | Relative drop |
|---|---|---|
| MLP | **−30.0 pts** | −86% |
| LSTM | **−8.0 pts** | −93% |

The recurrent policy degrades by roughly one-quarter as many percentage points as the memoryless baseline.

### Temporal consistency (higher = smoother)

| Policy | No DR | Mild DR | Severe DR |
|---|---|---|---|
| MLP | 0.75 | 0.69 | 0.70 |
| LSTM | 0.70 | 0.74 | **0.78** |

### Collision rate under severe DR

| Policy | Collision rate |
|---|---|
| MLP | 90% |
| LSTM | 97% |

Both policies fail the severe condition primarily by colliding with obstacles.

---

## 📂 Project Structure

```
sim2real-navigation/
├── config.yaml              Hyperparameters and environment settings
├── env_nav.py               Navigation environment with goal occlusion
├── policies.py              LSTM feature extractor for SB3
├── metrics.py               Episode-level evaluation metrics
├── train.py                 Training (PPO / RecurrentPPO)
├── evaluate.py              Evaluation with DR condition flags
├── export_tb.py             TensorBoard scalars → CSV
├── make_figures.py          Reproduce paper figures
├── requirements.txt         Python dependencies
├── CITATION.cff             Citation metadata
├── LICENSE                  MIT license
├── .zenodo.json             Zenodo metadata
├── paper/                   LaTeX source and compiled PDF
├── figures/                 PNGs used in this README
└── results/                 Evaluation CSVs
```

---

# Paper Source

This folder contains the LaTeX source and compiled PDF of the paper
associated with this repository.

- `main.tex` — LaTeX source
- `main.pdf` — compiled PDF
- `figures/` — figure PDFs used by `main.tex`

## How to compile

**On Overleaf:**
1. Create a new project → "Upload Project"
2. Zip the entire `paper/` folder and upload
3. Overleaf will compile automatically

**Locally (requires pdflatex):**
```bash
cd paper
pdflatex main.tex
pdflatex main.tex   # run twice for cross-references

## 📚 Citation

If you use this code, benchmark, or results in your work, please cite:

```bibtex
@misc{khan2026temporal,
  title        = {Temporal Memory for Robust Navigation under Distribution Shift:
                  A Controlled Benchmark of Recurrent and Memoryless Policies},
  author       = {Khan, Sultan Ali},
  year         = {2026},
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.23256938},
  url          = {https://github.com/sultanalikhan7543/sim2real-navigation}
  orcid        = {https://orcid.org/0009-0004-7403-6484}
}
```

---

## 🛠️ Reproducibility Notes

- All results reported in the paper were produced on a single CPU (Windows 11, Python 3.11, PyTorch 2.x CPU build).
- Training is deterministic given a fixed seed; each run above uses the default SB3 seeding behavior.
- To replicate the exact numbers in the paper, delete `models/` and `logs/` before each run.
- The severe DR condition was added *after* training; it is evaluated by passing `--severe` to `evaluate.py`.

---

## ⚠️ Known Limitations

- **Single seed per policy.** RL is highly seed-sensitive; the results here are indicative but not statistically conclusive.
- **Low absolute performance.** Even the best condition reaches 35% success. The goal-occlusion feature makes this task significantly harder than standard mapless navigation benchmarks.
- **No physical robot validation.** All experiments are in simulation.

These limitations are discussed in detail in the paper. Multi-seed replication is ongoing.

---

## 🤝 Contributing

Issues and pull requests are welcome. If you reproduce or extend these results, please open an issue or send a pull request with a link to your work.

---

## 📜 License

This project is released under the MIT License. See [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgements

Built on top of [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3) and [sb3-contrib](https://github.com/Stable-Baselines3/sb3-contrib). Environment and training loops are written in pure NumPy and PyTorch without external simulators.