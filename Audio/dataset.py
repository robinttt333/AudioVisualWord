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


class AudioDataset(Dataset):
    def __init__(self, mode):
        self.mode = mode
        self.samples = []
        self.init()
        self.size = len(self.samples)
        self.SNRlevels = [-5, 0, 5, 10, 15, 20]

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

    def normalize(self, audio):
        std = audio.std()
        std = std if std is not 0 else 1
        mean = audio.mean()
        audio = (audio - mean) / std
        return audio

    def __getitem__(self, idx):
        audio = np.load(self.samples[idx][1])['data']
        if self.mode == "train":
            audio = self.inject_noise_sample(
                audio, random.choice(self.SNRlevels))
        return self.samples[idx][0], self.normalize(audio)

    def __len__(self):
        return self.size

    def init(self):
        path = config.data["processedPath"]
        ct = 0
        labels = os.listdir(os.path.join(os.curdir, path))
        for i, label in enumerate(labels):
            audiosPath = os.path.join(path, label, self.mode)
            audios = os.listdir(audiosPath)
            for audio in audios:
                self.samples.append((ct, os.path.join(audiosPath, audio)))
                ct += 1
