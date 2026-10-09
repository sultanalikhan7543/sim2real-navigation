"""
Recurrent feature extractor for Stable-Baselines3.
Reshapes the flat stacked-history observation into (batch, history_len, obs_dim)
and processes it with an LSTM.
"""

import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class LSTMFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=64, history_len=5, hidden_dim=64):
        super().__init__(observation_space, features_dim)
        total_dim = observation_space.shape[0]
        assert total_dim % history_len == 0, (
            f"obs dim {total_dim} not divisible by history_len {history_len}"
        )
        obs_dim = total_dim // history_len

        self.history_len = history_len
        self.obs_dim = obs_dim

        self.lstm = nn.LSTM(input_size=obs_dim, hidden_size=hidden_dim,
                            num_layers=1, batch_first=True)
        self.proj = nn.Sequential(
            nn.Linear(hidden_dim, features_dim),
            nn.ReLU(),
        )

    def forward(self, obs):
        # obs: (batch, history_len * obs_dim)
        batch = obs.shape[0]
        seq = obs.view(batch, self.history_len, self.obs_dim)
        out, _ = self.lstm(seq)
        # Take the last timestep
        return self.proj(out[:, -1, :])