import torch
import torch.nn as nn
import torch.nn.functional as F


class RewardModel(nn.Module):
    def __init__(self, n_observation, n_action) -> None:
        super(RewardModel, self).__init__()
        self.layer1 = nn.Linear(n_observation + n_action, 128)
        self.layer2 = nn.Linear(128, 32)
        self.output = nn.Linear(32, 1)

    def forward(self, obs, act):

        x = torch.cat((obs, act))
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        out = F.tanh(self.output(x))
        return out


