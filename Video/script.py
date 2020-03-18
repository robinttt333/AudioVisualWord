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
files = glob.glob(os.path.join("../data", '*', '*', '*.mp4'))
for file in files:

    preprocessedVideo = extractFramesAndCrop(file)
    """The path to save needs to be corrected based on the data"""
    path_to_save = os.curdir
    np.savez(path_to_save, data=preprocessedVideo)
