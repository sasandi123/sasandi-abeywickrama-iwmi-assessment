#Importing required Libraries
import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Additional Libraries
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, BatchNormalization,
    Dropout, Flatten, Dense, GlobalAveragePooling2D
)
from tensorflow.keras.callbacks import (
    ModelCheckpoint,EarlyStopping,ReduceLROnPlateau,
)
from tensorflow.keras.optimizers import Adam

class ModelDevelopment:

    def __init__(self):
        #Image size should match what preprocessing uses
        self.img_size=(128,128,3)
        self.num_classes=1 #binary - mask vs no mask
        self.model=None
        self.history=None

