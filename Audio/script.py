import config
import os
import glob
import cv2
import numpy as np
import librosa
import warnings

warnings.simplefilter("ignore")
path = config.data["path"]
path = os.path.join(os.getcwd(), path)
outputPath = os.path.join(os.getcwd(), "data")
if not os.path.exists(outputPath):
    os.mkdir(outputPath)

labels = os.listdir(path)
for label in labels:
    if not os.path.exists(os.path.join(outputPath, label)):
        os.mkdir(os.path.join(outputPath, label))
        os.mkdir(os.path.join(outputPath, label, "train"))
        os.mkdir(os.path.join(outputPath, label, "val"))

    modes = ["train", "val"]
    for mode in modes:
        dir = os.path.join(path, label, mode)
        files = os.listdir(dir)

        for file in files:
            if file.endswith('.mp4'):
                path_to_save = os.path.join(
                    outputPath, label, mode, file.split(".")[0])
                preprocessedVideo, sr = librosa.load(
                    os.path.join(dir, file), sr=16000)
                preprocessedVideo = preprocessedVideo[-19456:]
                np.savez(path_to_save, data=preprocessedVideo)


for file in files:
    """The path to save needs to be corrected based on the data"""
    path_to_save = os.curdir
    np.savez(path_to_save, data=file)
