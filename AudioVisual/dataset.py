from torch.utils.data import Dataset
import os
import glob
import numpy as np
import torchvision.transforms as transforms
import cv2
import config
import torch
import librosa
import numpy as np
import random


class AudioVideoDataset(Dataset):
    def __init__(self, mode):
        self.mode = mode
        self.samples = []
        self.init()
        self.size = len(self.samples)
        self.SNRlevels = [-5, 0, 5, 10, 15, 20]

    def __getitem__(self, idx):
        video = self.loadFileAndApplyTransforms(idx)
        audio = np.load(self.samples[idx][2])['data']
        if self.mode == "train":
            audio = self.inject_noise_sample(
                audio, random.choice(self.SNRlevels))
        return self.samples[idx][0], self.normalize(audio), video

    def __len__(self):
        return self.size

    def init(self):
        videoPath = config.data["processedPathVideo"]
        audioPath = config.data["processedPathAudio"]
        ct = 0
        labels = os.listdir(os.path.join(os.curdir, videoPath))
        for i, label in enumerate(labels):
            videosPath = os.path.join(videoPath, label, self.mode)
            audiosPath = os.path.join(audioPath, label, self.mode)
            videos = os.listdir(videosPath)
            audios = os.listdir(audiosPath)
            for idx in range(len(videos)):
                self.samples.append((ct, os.path.join(
                    videosPath, videos[idx]), os.path.join(audiosPath, audios[idx])))
                ct += 1

    def normalize(self, audio):
        std = audio.std()
        std = std if std is not 0 else 1
        mean = audio.mean()
        audio = (audio - mean) / std
        return audio

    def inject_noise_sample(self, sample, SNR, noise='blabber.wav'):
        """https://github.com/iariav/End-to-End-VAD/blob/b95d08da9bdb8711693d4d551473db640e36d69d/utils/NoiseInjection.py"""
        path = os.path.join(os.getcwd(), noise)
        noise, _ = librosa.load(path, sr=16000)
        noise = noise[:len(sample)]
        sample_std = np.std(sample)
        noise_std = np.std(noise)
        new_noise_std = sample_std / (10 ** (SNR / 20))
        noise /= noise_std
        noise *= new_noise_std
        return sample + noise

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
                transforms.ToTensor(),  # divides by 255 by default
                transforms.Normalize([0.413621], [0.1688])
            ]
        video = [transforms.Compose(tfs)(frame) for frame in video]
        video = torch.stack(video, axis=0)
        return video
