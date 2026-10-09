"""
Evaluate a trained PPO or RecurrentPPO policy.
Reads metadata JSON to determine architecture automatically.
Supports --domain-randomization (mild) and --severe (unseen harder shift).
"""

import argparse
import json
import os
import sys
import yaml
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor

from env_nav import NavigationEnv, HistoryWrapper
from metrics import Metrics, temporal_consistency


def make_env(config, domain_randomization, severe=False):
    def _init():
        env = NavigationEnv(
            world_size=config["env"]["world_size"],
            max_steps=config["env"]["max_steps"],
            lidar_beams=config["env"]["lidar_beams"],
            lidar_range=config["env"]["lidar_range"],
            goal_threshold=config["env"]["goal_threshold"],
            collision_threshold=config["env"]["collision_threshold"],
            num_obstacles=10 if severe else config["env"]["num_obstacles"],
            domain_randomization=domain_randomization,
            severe_dr=severe,
        )
        env = HistoryWrapper(env, history_len=1)
        return Monitor(env)

    return DummyVecEnv([_init])


def run(args):
    print("Loading config...", flush=True)
    with open(args.config) as f:
        config = yaml.safe_load(f)

    meta_path = args.model + "_meta.json"
    with open(meta_path) as f:
        meta = json.load(f)
    algorithm = meta.get("algorithm", "ppo")
    base_obs_dim = meta["base_obs_dim"]
    print(f"Metadata: algorithm={algorithm}, base_obs_dim={base_obs_dim}", flush=True)

    if args.severe:
        print("Test profile: SEVERE distribution shift (unseen)", flush=True)
    elif args.domain_randomization:
        print("Test profile: mild DR (training distribution)", flush=True)
    else:
        print("Test profile: no DR (in-distribution)", flush=True)

    print("Building env...", flush=True)
    env = make_env(config,
                   domain_randomization=args.domain_randomization,
                   severe=args.severe)
    env = VecNormalize.load(args.vecnorm, env)
    env.training = False
    env.norm_reward = False

    print(f"Loading model from {args.model}...", flush=True)
    if algorithm == "recurrent_ppo":
        from sb3_contrib import RecurrentPPO
        model = RecurrentPPO.load(args.model, env=env, device="cpu")
    else:
        model = PPO.load(args.model, env=env, device="cpu")

    metrics = Metrics()
    print(f"Running {args.episodes} episodes...", flush=True)

    for ep in range(args.episodes):
        obs = env.reset()
        done = False
        positions, actions, rewards = [], [], []
        info = {}

        if algorithm == "recurrent_ppo":
            lstm_states = None
            episode_starts = np.ones((env.num_envs,), dtype=bool)

        while not done:
            if algorithm == "recurrent_ppo":
                action, lstm_states = model.predict(
                    obs,
                    state=lstm_states,
                    episode_start=episode_starts,
                    deterministic=True,
                )
                obs, reward, done_arr, info_arr = env.step(action)
                episode_starts = np.array([bool(done_arr[0])], dtype=bool)
            else:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done_arr, info_arr = env.step(action)

            done = bool(done_arr[0])
            info = info_arr[0]

            raw = obs[0, -base_obs_dim:]
            positions.append(raw[:2].copy())
            actions.append(action[0].copy())
            rewards.append(float(reward[0]))

        if len(positions) > 1:
            path_len = float(np.sum(np.linalg.norm(
                np.diff(np.array(positions), axis=0), axis=1)))
        else:
            path_len = 0.0

        metrics.add(
            success=info.get("is_success", False),
            collision=info.get("is_collision", False),
            steps=info.get("step_count", 0),
            total_reward=float(np.sum(rewards)),
            path_length=path_len,
            temporal_consistency=temporal_consistency(actions),
            final_dist=info.get("distance_to_goal", float("inf")),
        )

        if (ep + 1) % 10 == 0:
            print(f"  episode {ep + 1}/{args.episodes} done", flush=True)

    summary = metrics.summary()
    print("\n=== Evaluation Summary ===", flush=True)
    for k, v in summary.items():
        if isinstance(v, float):
            print(f"{k:28s}: {v:.4f}", flush=True)
        else:
            print(f"{k:28s}: {v}", flush=True)

    metrics.to_csv(args.out)
    print(f"\nSaved raw episode data to {args.out}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--vecnorm", required=True)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--domain-randomization", action="store_true",
                        help="mild DR (matches training distribution)")
    parser.add_argument("--severe", action="store_true",
                        help="severe distribution shift (unseen, harder)")
    parser.add_argument("--out", default="eval_results.csv")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"ERROR: {type(e).__name__}: {e}", flush=True)
        import traceback
        traceback.print_exc()
        sys.exit(1)