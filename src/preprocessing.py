#Importing required Libraries
import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import shutil
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator

class BasicPreprocessing:
    def __init__(self):
        #setting up paths and config
        self.data_dir= None
        self.train_dir="data/train"
        self.val_dir="data/val"
        self.test_dir="data/test"
        self.img_size=(128,128)
        self.batch_size=32
        self.class_names=["with_mask","without_mask"]
