"""The classifier. Copy this file into Colab for training; the app imports the same file."""
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class CausalLSTM(nn.Module):
    def __init__(self, input_dim=768, hidden_dim=128):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, x, lengths):
        """Training: x is [batch, turns, 768]; returns one logit per turn."""
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        out, _ = self.lstm(packed)
        out, _ = pad_packed_sequence(out, batch_first=True)
        return self.head(out).squeeze(-1)

    def step(self, turn_emb, state=None):
        """Live use: feed ONE turn [1, 1, 768]; returns fraud probability and new state."""
        out, state = self.lstm(turn_emb, state)
        return torch.sigmoid(self.head(out)).view(-1), state
