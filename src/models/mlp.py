import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, num_classes = 10):
        super(MLP, self).__init__()
        self.feature = nn.Sequential(
            nn.Linear(3*28*28, 100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Linear(100, 16),
            nn.ReLU()
        )
        self.fc = nn.Linear(16, num_classes)


    def forward(self, x):
        x = x.view(x.size(0), -1) / 255
        x = self.feature(x)
        x = self.fc(x)
        return x