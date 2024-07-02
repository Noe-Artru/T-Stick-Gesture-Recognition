import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import sys

# Define LSTM classifier model
class LSTMClassifier(nn.Module):
    def __init__(self, num_sensors, hidden_size, num_gestures, device='cpu'):
        super().__init__()
        self.num_sensors = num_sensors
        self.hidden_size = hidden_size
        self.num_layers = 1 # See later for modifications, requires change in foward pass
        self.num_gestures = num_gestures


        self.lstm = nn.LSTM(
            input_size=self.num_sensors,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True
        ).to(device)
        self.fc = nn.Linear(self.hidden_size, self.num_gestures).to(device)
        self.sigmoid = nn.Sigmoid().to(device)

    def forward(self, x):
        batch_size = x.size(0)
        hidden = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device).requires_grad_()
        cell = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device).requires_grad_()

        _, (hidden, _) = self.lstm(x, (hidden, cell))
        out = self.fc(hidden[0]) # First dim -> num layers (1 for now, thus hidden[0])
        out = out.flatten()
        out = self.sigmoid(out)
        return out

