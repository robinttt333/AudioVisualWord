import config
import os
import glob
import cv2
import numpy as np


def extractFramesAndCrop(file):
    frames = []
    cap = cv2.VideoCapture(file)
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
        else:
            break
    cap.release()

    frames = np.array(frames)
    return frames[..., ::-1][:, 115:211, 79:175]


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
                preprocessedVideo = extractFramesAndCrop(
                    os.path.join(dir, file))
                np.savez(path_to_save, data=preprocessedVideo)
