from torch.utils.data import Dataset
import os
import glob
import numpy as np
import torchvision.transforms as transforms
import cv2
import config
import torch


class VideoDataset(Dataset):
    def __init__(self, mode):
        self.mode = mode
        self.samples = []
        self.init()
        self.size = len(self.samples)

    def __getitem__(self, idx):
        video = self.loadFileAndApplyTransforms(idx)
        return self.samples[idx][0], video

    def __len__(self):
        return self.size

    def init(self):
        path = config.data["path"]
        ct = 0
        labels = os.listdir(os.path.join(os.curdir, path))
        for i, label in enumerate(labels):
            videosPath = os.path.join(path, label, self.mode)
            videos = os.listdir(videosPath)
            for video in videos:
                self.samples.append((ct, os.path.join(videosPath, video)))
                ct += 1

    def loadFileAndApplyTransforms(self, idx):
        video = np.load(self.samples[idx][1])['data']
        if self.mode == "train":
            tfs = [
                transforms.ToPILImage(),
                transforms.Grayscale(num_output_channels=1),
                transforms.RandomCrop(88),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.413621], [0.1688])
            ]
        else:
            tfs = [
                transforms.ToPILImage(),
                transforms.Grayscale(num_output_channels=1),
                transforms.CenterCrop(88),
                transforms.ToTensor(),
                transforms.Normalize([0.413621], [0.1688])
            ]
        video = [transforms.Compose(tfs)(frame) for frame in video]
        video = torch.stack(video, axis=0)
        return video
