import torch
import numpy as np
from torch.optim import Adam
import torch.nn as nn


class Generator(nn.Module):

    def __init__(self, input_length: int):
        super(Generator, self).__init__()
        self.dense_layer = nn.Linear(int(input_length)*4, 256)
        self.hidden1 = nn.Linear(256, 512)
        self.hidden2 = nn.Linear(512, 512)
        self.output = nn.Linear(512, int(input_length))
        self.activation = nn.LeakyReLU()

    def forward(self, x):
        x = self.activation(self.dense_layer(x))
        x = self.activation(self.hidden1(x))
        x = self.activation(self.hidden2(x))
        return torch.sigmoid(self.output(x))


class Discriminator(nn.Module):
    def __init__(self, input_length: int):
        super(Discriminator, self).__init__()
        self.dense_layer = nn.Linear(int(input_length)*4, 256)
        self.hidden1 = nn.Linear(256, 512)
        self.output = nn.Linear(512, int(input_length))
        self.activation = nn.LeakyReLU()

    def forward(self, x):
        x = self.activation(self.dense_layer(x))
        x = self.activation(self.hidden1(x))
        return torch.sigmoid(self.output(x))
