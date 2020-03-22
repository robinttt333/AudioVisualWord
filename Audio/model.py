import torch.nn as nn
from resnet import resnet18


class Lipreader(nn.Module):
    def __init__(self):
        super(Lipreader, self).__init__()
        self.Fronted1D = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=80, stride=4, padding=38, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(True)
        )
        self.resnet18 = resnet18()

    def forward(self, x):
        x = x.view(-1, 1, x.size(1))
        x = self.Fronted1D(x)
        x = self.resnet18(x)
        x = x.view(-1, 29, 256)
        return x
