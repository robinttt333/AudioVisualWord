import torch.nn as nn
from torchvision.models import resnet
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
        x = self.backend_conv1(x)
        x = self.backend_conv2(torch.mean(x, 2))
        return x


class Resnet(nn.Module):
    def __init__(self):
        super(Resnet, self).__init__()
        self.model = resnet.resnet34(False, num_classes=256)
        """By default the pytorch resnet models take in 3 channels but we have 64 so we need to modify
        the first layer input to 64 instead of 3"""
        self.model.conv1 = nn.Conv2d(64, 64, kernel_size=7, stride=2, padding=3,
                                     bias=False)

    def forward(self, input):
        """ input shape is batch * channels * frames * height * width. We need to convert this
        5d tensor to a 4d tensor. So we first change its shape to batch * frames * channels * height * width. Then,
        since the input needs to be a 4d tensor we concatenate the batch and frames dimensions into 1 dimension by doing
        transposed.reshape(-1,64,28,28).
        Finally we reshape the O/P back to batch * frames * 256. This 256 will be an encoded representation
        corresponding to each frame of the video.
        """
        height, width = input.shape[3], input.shape[4]
        input = input.transpose(1, 2)
        vector4d = input.reshape(-1,
                                 64, height, width)
        output = self.model(vector4d)
        output = output.view(-1, 29, 256)
        return output


class Lipreader(nn.Module):
    def __init__(self):
        super(Lipreader, self).__init__()
        stage = 1
        self.FrontEnd = nn.Sequential(
            nn.Conv3d(1, 64, kernel_size=(5, 7, 7), stride=(
                1, 2, 2), padding=(2, 3, 3), bias=False),
            nn.BatchNorm3d(64),
            nn.ReLU(True),
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(
                1, 2, 2), padding=(0, 1, 1))
        )
        self.resnet34 = Resnet()
        self.Backend = TemporalCNN() if stage == 1 else GRU()

    def forward(self, x):
        x = self.FrontEnd(x)
        x = self.resnet34(x)
        x = x.transpose(1, 2)
        x = self.Backend(x)
        return x
