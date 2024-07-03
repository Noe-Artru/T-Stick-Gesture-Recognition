import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import sys

# Define LSTM classifier model
class LSTMClassifier2(nn.Module):
    def __init__(self, num_sensors, hidden_size, window_size, num_gestures, device='cpu'):
        super().__init__()
        self.num_sensors = num_sensors
        self.hidden_size = hidden_size
        self.num_layers = 1 # See later for modifications, requires change in foward pass
        self.num_gestures = num_gestures
        self.window_size = window_size

        # Avoid Dropout on LSTM layer as LSTM are not good at learning multiple things at once

        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=self.num_sensors,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True
        ).to(device)

        # GRU layer
        self.gru = nn.GRU(
            input_size=self.num_sensors,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True
        ).to(device)

        # Simple RNN layer
        self.rnn = nn.RNN(
            input_size=self.num_sensors,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True
        ).to(device)
        
        # Fully connected Layer
        self.fc = nn.Linear(self.hidden_size, self.num_gestures).to(device)
        # Dropout 
        self.dropout = nn.Dropout(0.3).to(device)
        # Activation
        self.sigmoid = nn.Sigmoid().to(device)

    def forward(self, x):
        batch_size = x.size(0)
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device).requires_grad_()
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device).requires_grad_()

        _, (hidden, _) = self.lstm(x, (h0, c0))

        _, hidden = self.gru(x, hidden) # Stacking a GRU for more complexity ? Should I run in it parallel with the new hidden state ?

        _, hidden = self.rnn(x, hidden)
        
        out = self.fc(hidden[0]) # First dim -> num layers (1 for now, thus hidden[0])
        out = self.dropout(out)
        out = out.flatten()
        out = self.sigmoid(out)
        return out