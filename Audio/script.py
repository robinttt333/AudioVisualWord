import config
import os
import glob
import cv2
import numpy as np
import librosa

path = config.data["path"]
files = glob.glob(os.path.join("../data", '*', '*', '*.mp4'))
for file in files:
    file, sr = librosa.load(file, sr=16000)
    """The path to save needs to be corrected based on the data"""
    path_to_save = os.curdir
    np.savez(path_to_save, data=file)
