import torch
import torch.nn as nn
import torch.nn.functional as F


class DQN(nn.Module):
    def __init__(self, n_observation, n_action) -> None:
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observation, 128)
        self.layer2 = nn.Linear(128, 32)
        self.output = nn.Linear(32, n_action)

    def forward(self, x):

        # print(x.shape)
        x = F.relu(self.layer1(x))
        # print(x.shape)
        x = F.relu(self.layer2(x))
        # print(x.shape)
        out = F.relu(self.output(x))
        return out

