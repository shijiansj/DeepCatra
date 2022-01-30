import torch
import torch.nn as nn
from lstm_preprocess import encoding

opcode_dict = encoding()
class LSTM_net(nn.Module):
    def __init__(self):
        super(LSTM_net, self).__init__()
        self.embedding = nn.Embedding(len(opcode_dict),128)
        self.lstm = nn.LSTM(input_size=128, hidden_size=256, num_layers=2, dropout=0.3, bidirectional=True, batch_first=True)
        self.fc1 = nn.Sequential(nn.Linear(512, 64),
                                 nn.Tanh())
        self.fc2 = nn.Sequential(nn.Linear(64, 32),
                                 nn.Tanh())
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.embedding(x)
        x, _ = self.lstm(x)
        x = self.fc1(x[:, -1, :])
        x = self.fc2(x)
        x = self.tanh(torch.mean(x,0))
        return x

