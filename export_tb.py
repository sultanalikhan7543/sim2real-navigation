"""Export TensorBoard scalars to CSV files for plotting."""
import os
import glob
import pandas as pd
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

LOG_DIR = "logs"
OUT_DIR = "paper_data"
os.makedirs(OUT_DIR, exist_ok=True)

SCALARS = ["rollout/ep_rew_mean", "rollout/success_rate", "rollout/ep_len_mean"]

for run_dir in sorted(glob.glob(os.path.join(LOG_DIR, "*"))):
    if not os.path.isdir(run_dir):
        continue
    name = os.path.basename(run_dir)
    print(f"Processing {name}...")
    ea = EventAccumulator(run_dir, size_guidance={"scalars": 0})
    ea.Reload()
    available = ea.Tags().get("scalars", [])
    for tag in SCALARS:
        if tag not in available:
            continue
        events = ea.Scalars(tag)
        df = pd.DataFrame([(e.step, e.value) for e in events],
                          columns=["step", "value"])
        safe_tag = tag.replace("/", "_")
        out = os.path.join(OUT_DIR, f"{name}__{safe_tag}.csv")
        df.to_csv(out, index=False)
        print(f"  wrote {out}")
print("Done.")