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

    def build_model(self):
        #building CNN from scratch
        #3 conv blocks with increasing filters, then dense head

        model = Sequential([

            #----- Block 1 -----
            #learning low level features like edges and colors
            Conv2D(32, (3, 3), activation="relu", padding="same", input_shape=self.img_size),
            BatchNormalization(),
            Conv2D(32, (3, 3), activation="relu", padding="same"),
            BatchNormalization(),
            MaxPooling2D(pool_size=(2, 2)),
            Dropout(0.25),

            #--- Block 2 ---
            #going deeper - learning more complex patterns
            Conv2D(64, (3, 3), activation="relu", padding="same"),
            BatchNormalization(),
            Conv2D(64,(3,3), activation="relu", padding="same"),
            BatchNormalization(),
            MaxPooling2D(pool_size=(2, 2)),
            Dropout(0.25),

            #--- Block 3 ---
            #highest level spatial features before flatten
            Conv2D(128, (3, 3), activation="relu", padding="same"),
            BatchNormalization(),
            Conv2D(128, (3,3), activation="relu", padding="same"),
            BatchNormalization(),
            MaxPooling2D(pool_size=(2, 2)),
            Dropout(0.30),

            #--- Fully Connected Head ---
            GlobalAveragePooling2D(),
            Dense(256, activation="relu"),
            BatchNormalization(),
            Dropout(0.40),
            Dense(128, activation="relu"),
            Dropout(0.30),

            #output - sigmoid for binary classification
            Dense(1, activation="sigmoid")

        ])

        model.summary()
        self.model = model
        return model



