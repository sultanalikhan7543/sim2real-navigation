"""
Train PPO (MLP) or RecurrentPPO (LSTM) on the occluded-goal navigation env.
- MLP: single frame, feedforward, no memory
- LSTM: single frame per step, recurrent hidden state learned across time
Saves metadata JSON alongside each model for automatic evaluation.
"""

import os
import json
import argparse
import yaml
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback

from env_nav import NavigationEnv, HistoryWrapper


def make_env(config, domain_randomization):
    def _init():
        env = NavigationEnv(
            world_size=config["env"]["world_size"],
            max_steps=config["env"]["max_steps"],
            lidar_beams=config["env"]["lidar_beams"],
            lidar_range=config["env"]["lidar_range"],
            goal_threshold=config["env"]["goal_threshold"],
            collision_threshold=config["env"]["collision_threshold"],
            num_obstacles=config["env"]["num_obstacles"],
            domain_randomization=domain_randomization,
        )
        # history_len=1 for both: LSTM handles sequence internally via RecurrentPPO,
        # MLP sees one frame only. This is the clean memory-vs-no-memory comparison.
        env = HistoryWrapper(env, history_len=1)
        return Monitor(env)

    return DummyVecEnv([_init])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--policy", default="mlp", choices=["mlp", "lstm"])
    parser.add_argument("--no-dr", action="store_true")
    parser.add_argument("--timesteps", type=int, default=None)
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    total_timesteps = args.timesteps or config["training"]["total_timesteps"]
    dr = not args.no_dr
    history_len = 1  # both policies use single frames

    os.makedirs(config["training"]["log_dir"], exist_ok=True)
    os.makedirs(config["training"]["model_dir"], exist_ok=True)
    os.makedirs(os.path.join(config["training"]["model_dir"], "checkpoints"),
                exist_ok=True)

    train_env = make_env(config, domain_randomization=dr)
    train_env = VecNormalize(train_env, norm_obs=True, norm_reward=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    base_obs_dim = 7 + config["env"]["lidar_beams"]

    print(f"Training on: {device}")
    print(f"Policy: {args.policy} | Domain randomization: {dr}")
    print(f"Base obs dim: {base_obs_dim}")
    print(f"Total timesteps: {total_timesteps}")

    if args.policy == "lstm":
        from sb3_contrib import RecurrentPPO

        # RecurrentPPO handles hidden state internally. Lower LR for stability.
        model = RecurrentPPO(
            "MlpLstmPolicy",
            train_env,
            learning_rate=1.0e-4,
            n_steps=2048,
            batch_size=256,
            gamma=config["training"]["gamma"],
            gae_lambda=config["training"]["gae_lambda"],
            n_epochs=10,
            clip_range=0.2,
            ent_coef=0.001,
            max_grad_norm=0.5,
            policy_kwargs=dict(
                lstm_hidden_size=128,
                n_lstm_layers=1,
                shared_lstm=False,
                enable_critic_lstm=True,
                net_arch=dict(pi=[64, 64], vf=[64, 64]),
            ),
            tensorboard_log=config["training"]["log_dir"],
            device=device,
            verbose=1,
        )
    else:
        model = PPO(
            "MlpPolicy",
            train_env,
            learning_rate=config["training"]["learning_rate"],
            n_steps=config["training"]["n_steps"],
            batch_size=config["training"]["batch_size"],
            gamma=config["training"]["gamma"],
            gae_lambda=config["training"]["gae_lambda"],
            policy_kwargs=dict(net_arch=dict(pi=[128, 128], vf=[128, 128])),
            tensorboard_log=config["training"]["log_dir"],
            device=device,
            verbose=1,
        )

    tag = f"ppo_{args.policy}{'_dr' if dr else '_nodr'}"
    ckpt = CheckpointCallback(
        save_freq=20000,
        save_path=os.path.join(config["training"]["model_dir"], "checkpoints"),
        name_prefix=tag,
    )

    model.learn(total_timesteps=total_timesteps, callback=ckpt, progress_bar=False)

    model_path = os.path.join(config["training"]["model_dir"], tag)
    model.save(model_path)
    train_env.save(model_path + "_vecnorm.pkl")

    meta = {
        "policy": args.policy,
        "algorithm": "recurrent_ppo" if args.policy == "lstm" else "ppo",
        "domain_randomization": dr,
        "history_len": history_len,
        "lidar_beams": config["env"]["lidar_beams"],
        "base_obs_dim": base_obs_dim,
        "timesteps": total_timesteps,
    }
    with open(model_path + "_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nSaved model to {model_path}.zip")
    print(f"Saved metadata to {model_path}_meta.json")


if __name__ == "__main__":
    main()