import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# convert an AxBxN np array to AxBxLx(N-L)


class Net(nn.Module):
    def __init__(self, hidden=1024):
        super().__init__()
        self.input = torch.nn.Linear(128, hidden)

        self.output = torch.nn.Linear(hidden, 32)

    def forward(self, x):
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float)
        x = F.relu(self.input(x))
        return torch.sigmoid(self.output(x))
