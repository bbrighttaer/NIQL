
import torch.nn as nn
import torch.nn.functional as F


class SimpleCommNet(nn.Module):

    def __init__(self, input_dim, hdim, com_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hdim)
        self.fc2 = nn.Linear(hdim, hdim)
        self.fc3 = nn.Linear(hdim, com_dim)

    def forward(self, obs):
        x = F.relu(self.fc1(obs))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
