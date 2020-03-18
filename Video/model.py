import torch.nn as nn
from torchvision.models import resnet


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
        self.FrontEnd = nn.Sequential(
            nn.Conv3d(1, 64, kernel_size=(5, 7, 7), stride=(
                1, 2, 2), padding=(2, 3, 3), bias=False),
            nn.BatchNorm3d(64),
            nn.ReLU(True),
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(
                1, 2, 2), padding=(0, 1, 1))
        )
        self.resnet = Resnet()

    def forward(self, x):
        return self.resnet(self.FrontEnd(x))
