import torch.nn as nn
from resnet import resnet18
import torch
from operator import mul
import functools
import math
import re


class GRU(nn.Module):
    def __init__(self):
        super(GRU, self).__init__()
        self.gru = nn.GRU(256, 512, 2, bias=False,
                          batch_first=True, bidirectional=True)
        self.fc = nn.Linear(1024, 500)

    def forward(self, x):
        h0 = torch.zeros(2*2, x.size(0), 512)
        output, hn = self.gru(x, h0)
        x = self.fc(output)
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
        self.convolution1d = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=80, stride=4, padding=38, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(True)
        )
        self.resnet18 = resnet18()
        self.Backend = TemporalCNN() if stage == 1 else GRU()

        def weights_init(m):
            classname = m.__class__.__name__
            if classname in ["Conv1d", "Conv2d", "Conv3d"]:
                n = functools.reduce(mul, m.kernel_size) * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif re.search("BatchNorm[123]d", classname):
                m.weight.data.fill_(1.0)
                if m.bias is not None:
                    m.bias.data.fill_(0)
            elif re.search("Linear", classname):
                if m.bias is not None:
                    m.bias.data.fill_(0)

        self.apply(weights_init)

    def forward(self, x):
        x = x.view(-1, 1, x.size(1))
        x = self.convolution1d(x)
        x = self.resnet18(x)
        x = x.view(-1, 29, 256)
        x = self.Backend(x)
        return x
