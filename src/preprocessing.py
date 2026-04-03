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

    def import_dataset(self):
        # reading all image paths and labels from the dataset folder
        image_paths =[]
        labels=[]
        dataset_path="dataset"

        for label in self.class_names:
            class_folder=os.path.join(dataset_path,label)
            if not os.path.join(dataset_path,label):
                print(f"[WARNING] Folder not found: {class_folder}")
                continue
            for fname in os.listdir(class_folder):
                if fname.lower().endswith((".jpg",".jpeg",".png")):
                    image_paths.append(os.path.join(class_folder,fname))
                    labels.append(label)
        df=pd.DataFrame({"image_path":image_paths,"label":labels})
        print(f"Total images loaded: {len(df)}")
        print(df["label"].value_counts())
        return df

