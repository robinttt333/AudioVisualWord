import torch.nn as nn
from resnet import resnet18
import torch


class GRU(nn.Module):
    def __init__(self):
        super(GRU, self).__init__()
        self.gru = nn.GRU(256, 512, 2, bias=False,
                          batch_first=True, bidirectional=True)
        self.fc = nn.Linear(1024, 500)

    def forward(self, x):
        h0 = torch.zeros(2*2, x.size(2), 512)
        x = x.permute(2, 0, 1)
        output, hn = self.gru(x, h0)
        x = self.fc(output)
        x = x.transpose(0, 1)
        return x


class TemporalCNN(nn.Module):
    def __init__(self):
        super(TemporalCNN, self).__init__()
        self.backend_conv1 = nn.Sequential(
            nn.Conv1d(256, 512, 5, 2, 0, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(True),
            nn.MaxPool1d(2, 2),
            nn.Conv1d(512, 1024, 5, 2, 0, bias=False),
            nn.BatchNorm1d(1024),
            nn.ReLU(True),)
        self.backend_conv2 = nn.Sequential(
            nn.Linear(1024, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(True),
            nn.Linear(256, 500))

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.backend_conv1(x)
        x = torch.mean(x, 2)
        x = self.backend_conv2(x)
        return x


class Lipreader(nn.Module):
    def __init__(self, stage=1):
        super(Lipreader, self).__init__()
        self.Fronted1D = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=80, stride=4, padding=38, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(True)
        )
        self.resnet18 = resnet18()
        self.Backend = TemporalCNN() if stage == 1 else GRU()

    def forward(self, x):
        x = x.view(-1, 1, x.size(1))
        x = self.Fronted1D(x)
        x = self.resnet18(x)
        x = x.view(-1, 29, 256)
        x = self.Backend(x)
        return x
