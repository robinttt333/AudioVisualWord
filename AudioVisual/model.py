import torch.nn as nn
import torch
from audioModel import Lipreader as AudioModel
from videoModel import Lipreader as VideoModel


class GRU(nn.Module):
    def __init__(self):
        super(GRU, self).__init__()
        self.gru = nn.GRU(1000, 2048, 2, bias=False,
                          batch_first=True, bidirectional=True)
        self.fc = nn.Linear(2048*2, 500)

    def forward(self, x):
        h0 = torch.zeros(2*2, x.size(1), 2048)
        x = x.permute(1, 0, 2)
        output, hn = self.gru(x, h0)
        x = self.fc(output)
        x = x.transpose(0, 1)
        return x


class Lipreader(nn.Module):
    def __init__(self):
        super(Lipreader, self).__init__()
        self.audioModel = AudioModel()
        self.videoModel = VideoModel()
        self.gru = GRU()

    def forward(self, x):
        audioOutput = self.audioModel(x[0])
        videoOutput = self.videoModel(x[1])
        x = torch.cat((audioOutput, videoOutput), dim=2)
        x = self.gru(x)
        return x
