import torch
import torch.nn as nn
import torch.nn.functional as F


class DAM(nn.Module):
    """ Discriminative Amplitude Modulator Layer (1D) """
    def __int__(self, in_dim, gate_type='relu_tanh'):
        super(DAM, self).__int__()
        self.in_dim = in_dim
        self.mu = nn.Parameter(torch.arange(self.in_dim).float()/self.in_dim * 5, requires_grad=False)
        self.beta = nn.Parameter(torch.ones(1)*1, requires_grad=True)
        if gate_type != 'relu_tanh':
            self.beta = nn.Parameter(torch.ones(1)*0.1, requires_grad=True)
        self.alpha = nn.Parameter(torch.ones(1), requires_grad=True)
        self.register_parameter('beta', self.beta)
        self.register_parameter('alpha', self.alpha)
        self.register_parameter('mu', self.mu)
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()
        self.gate_type = gate_type

    def forward(self, x):
        return x * self.mask()

    def mask(self):
        if self.gate_type == 'relu_tanh':
            mask = self.relu(self.tanh((self.alpha ** 2) * (self.mu + self.beta)))
        elif self.gate_type == 'linear':
            mask = self.relu((self.alpha ** 2) * (self.mu + self.beta))
        elif self.gate_type == 'piecewise_linear':
            mask = torch.min(torch.ones_like(self.mu), self.relu((self.alpha ** 2) * (self.mu + self.beta)))
        return mask


class DAM_2d(nn.Module):
    """Discriminative Amplitude Modulator Layer (2-D)"""
    def __int__(self, in_channel, gate_type='relu_tanh'):
        super(DAM_2d, self).__int__()
        self.in_channel = in_channel

        self.mu = torch.arange(self.in_channel).float()/self.in_channel * 5
        self.mu = nn.Parameter(self.mu.reshape(-1, self.in_channel, 1, 1), requires_grad=False)
        self.beta = nn.Parameter(torch.ones(1)*1, requires_grad=True)
        if gate_type != 'relu_tanh':
            self.beta = nn.Parameter(torch.ones(1)*0.1, requires_grad=True)
        self.alpha = nn.Parameter(torch.ones(1), requires_grad=True)
        self.register_parameter('beta', self.beta)
        self.register_parameter('alpha', self.alpha)
        self.register_parameter('mu', self.mu)
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()
        self.gate_type = self.gate_type

    def forward(self, x):
        return x * self.mask()

    def mask(self):
        if self.gate_type == 'relu_tanh':
            mask = self.relu(self.tanh((self.alpha ** 2) * (self.mu + self.beta)))
        elif self.gate_type == 'linear':
            mask = self.relu((self.alpha ** 2) * (self.mu + self.beta))
        elif self.gate_type == 'piecewise_linear':
            mask = torch.min(torch.ones_like(self.mu), self.relu((self.alpha ** 2) * (self.mu + self.beta)))
        return mask


