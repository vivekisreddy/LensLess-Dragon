import torch as th
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class SmallCNN(BaseFeaturesExtractor):
    """
    Lightweight CNN feature extractor to replace the default NatureCNN in Stable-Baselines3.
    Model size reduced from ~1.6GB to <100MB.
    """
    def __init__(self, observation_space, features_dim=128):
        super(SmallCNN, self).__init__(observation_space, features_dim)
        n_input_channels = observation_space.shape[0]  # Number of channels (stacked frames)

        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 16, kernel_size=5, stride=4, padding=2),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4))
        )

        conv_out_size = 64 * 4 * 4

        self.linear = nn.Sequential(
            nn.Linear(conv_out_size, features_dim),
            nn.ReLU()
        )

    def forward(self, observations: th.Tensor) -> th.Tensor:
        x = self.cnn(observations)
        x = th.flatten(x, 1)
        return self.linear(x)
