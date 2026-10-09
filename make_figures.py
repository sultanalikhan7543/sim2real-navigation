"""Generate publication-quality figures from exported TensorBoard CSVs
   and evaluation CSVs. Outputs PDF (for LaTeX) and PNG (for GitHub)."""

import os
import glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.size": 10,
    "font.family": "serif",
    "axes.labelsize": 10,
    "axes.titlesize": 10,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "figure.dpi": 150,
    "savefig.bbox": "tight",
    "axes.grid": True,
    "grid.alpha": 0.3,
})

OUT = "figures"
os.makedirs(OUT, exist_ok=True)
DATA = "paper_data"


def smooth(x, w=15):
    if len(x) < w:
        return x
    return pd.Series(x).rolling(w, min_periods=1).mean().to_numpy()


def find_run(prefix):
    """Find the latest run matching the prefix inside paper_data."""
    matches = sorted(glob.glob(os.path.join(DATA, f"{prefix}*__rollout_ep_rew_mean.csv")))
    if not matches:
        raise FileNotFoundError(f"No run found for {prefix}")
    return matches[-1]


def extract_run_name(path):
    base = os.path.basename(path)
    return base.split("__")[0]


# ------------------------------------------------------------------
# Figure 1: learning curves (reward)
# ------------------------------------------------------------------
mlp_run = extract_run_name(find_run("PPO_"))
lstm_candidates = [p for p in glob.glob(os.path.join(DATA, "*__rollout_ep_rew_mean.csv"))
                   if "RecurrentPPO" in p]
if not lstm_candidates:
    raise FileNotFoundError("No RecurrentPPO run found")
lstm_run = extract_run_name(sorted(lstm_candidates)[-1])

mlp = pd.read_csv(os.path.join(DATA, f"{mlp_run}__rollout_ep_rew_mean.csv"))
lstm = pd.read_csv(os.path.join(DATA, f"{lstm_run}__rollout_ep_rew_mean.csv"))

fig, ax = plt.subplots(figsize=(5.5, 3.2))
ax.plot(mlp["step"] / 1e6, smooth(mlp["value"]), label="MLP (PPO)", color="#444444", lw=1.8)
ax.plot(lstm["step"] / 1e6, smooth(lstm["value"]), label="LSTM (RecurrentPPO)", color="#0072B2", lw=1.8)
ax.set_xlabel("Timesteps (millions)")
ax.set_ylabel("Mean episode reward")
ax.set_title("Learning curves")
ax.legend(loc="lower right", frameon=False)
fig.savefig(os.path.join(OUT, "fig1_learning_curves.pdf"))
fig.savefig(os.path.join(OUT, "fig1_learning_curves.png"))
plt.close(fig)

# ------------------------------------------------------------------
# Figure 2: success rate curves
# ------------------------------------------------------------------
mlp_s = pd.read_csv(os.path.join(DATA, f"{mlp_run}__rollout_success_rate.csv"))
lstm_s = pd.read_csv(os.path.join(DATA, f"{lstm_run}__rollout_success_rate.csv"))

fig, ax = plt.subplots(figsize=(5.5, 3.2))
ax.plot(mlp_s["step"] / 1e6, smooth(mlp_s["value"], 25), label="MLP (PPO)", color="#444444", lw=1.8)
ax.plot(lstm_s["step"] / 1e6, smooth(lstm_s["value"], 25), label="LSTM (RecurrentPPO)", color="#0072B2", lw=1.8)
ax.set_xlabel("Timesteps (millions)")
ax.set_ylabel("Success rate (training rollouts)")
ax.set_title("Success rate during training")
ax.set_ylim(0, 0.6)
ax.legend(loc="lower right", frameon=False)
fig.savefig(os.path.join(OUT, "fig2_success_curves.pdf"))
fig.savefig(os.path.join(OUT, "fig2_success_curves.png"))
plt.close(fig)

# ------------------------------------------------------------------
# Figure 3: robustness bar chart (headline figure)
# ------------------------------------------------------------------
# Hard-coded from evaluation CSVs — update if you re-run evaluation
conditions = ["No DR\n(in-dist.)", "Mild DR\n(training)", "Severe DR\n(unseen)"]
mlp_vals = [20.0, 35.0, 5.0]
lstm_vals = [33.0, 29.0, 2.0]

x = np.arange(len(conditions))
w = 0.36

fig, ax = plt.subplots(figsize=(5.5, 3.2))
b1 = ax.bar(x - w / 2, mlp_vals, w, label="MLP (PPO)", color="#888888", edgecolor="black", linewidth=0.5)
b2 = ax.bar(x + w / 2, lstm_vals, w, label="LSTM (RecurrentPPO)", color="#0072B2", edgecolor="black", linewidth=0.5)

for bars in (b1, b2):
    for b in bars:
        h = b.get_height()
        ax.annotate(f"{h:.1f}", (b.get_x() + b.get_width() / 2, h),
                    ha="center", va="bottom", fontsize=8)

ax.set_xticks(x)
ax.set_xticklabels(conditions)
ax.set_ylabel("Success rate (%)")
ax.set_title("Robustness under distribution shift")
ax.set_ylim(0, 45)
ax.legend(frameon=False)
fig.savefig(os.path.join(OUT, "fig3_robustness.pdf"))
fig.savefig(os.path.join(OUT, "fig3_robustness.png"))
plt.close(fig)

print("Wrote figures to", OUT)