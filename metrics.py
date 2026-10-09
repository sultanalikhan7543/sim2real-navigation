"""Tracks navigation metrics for evaluation and paper figures."""

import numpy as np
import pandas as pd


class Metrics:
    def __init__(self):
        self.episodes = []

    def add(self, success, collision, steps, total_reward,
            path_length, temporal_consistency, final_dist):
        self.episodes.append({
            "success": int(bool(success)),
            "collision": int(bool(collision)),
            "steps": int(steps),
            "total_reward": float(total_reward),
            "path_length": float(path_length),
            "temporal_consistency": float(temporal_consistency),
            "final_distance": float(final_dist),
        })

    def summary(self):
        if not self.episodes:
            return {}
        df = pd.DataFrame(self.episodes)
        return {
            "episodes": len(df),
            "success_rate": float(df["success"].mean()),
            "collision_rate": float(df["collision"].mean()),
            "avg_steps": float(df["steps"].mean()),
            "avg_reward": float(df["total_reward"].mean()),
            "avg_path_length": float(df["path_length"].mean()),
            "avg_temporal_consistency": float(df["temporal_consistency"].mean()),
            "avg_final_distance": float(df["final_distance"].mean()),
        }

    def to_csv(self, path):
        pd.DataFrame(self.episodes).to_csv(path, index=False)


def temporal_consistency(actions):
    """Lower mean action-change -> higher consistency score in (0, 1]."""
    if len(actions) < 2:
        return 1.0
    a = np.asarray(actions)
    diffs = np.linalg.norm(np.diff(a, axis=0), axis=1)
    mean_diff = float(np.mean(diffs))
    return 1.0 / (1.0 + mean_diff)