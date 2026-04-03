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
        self.train_dir = "../data/train"
        self.val_dir = "../data/val"
        self.test_dir = "../data/test"
        self.img_size=(128,128)
        self.batch_size=32
        self.class_names=["with_mask","without_mask"]

    def import_dataset(self):
        # reading all image paths and labels from the dataset folder
        image_paths =[]
        labels=[]
        dataset_path = "../dataset/Data Science - Dataset/data"

        for label in self.class_names:
            class_folder=os.path.join(dataset_path,label)
            if not os.path.exists(class_folder):
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

    def split_and_copy_dataset(self, df):
        #70/15/15 split
        train_df, temp_df = train_test_split(df,test_size=0.30,stratify=df["label"], random_state=42)
        val_df, test_df = train_test_split(temp_df, test_size=0.50, stratify=temp_df["label"], random_state=42)
        print(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")
        for split_name, split_df in [("train", train_df), ("val", val_df), ("test", test_df)]:
            for _, row in split_df.iterrows():
                label = row["label"]
                dest_folder = os.path.join(f"../data/{split_name}", label)
                os.makedirs(dest_folder, exist_ok=True)
                shutil.copy(row["image_path"], dest_folder)
        return train_df, val_df, test_df

    def check_class_distribution(self, df):
        #checking for class imbalance before training
        counts = df["label"].value_counts()
        plt.figure(figsize=(6,4))
        counts.plot(kind="bar", color=["steelblue", "coral"])
        plt.title("Class Distribution in Dataset")
        plt.xlabel("Class")
        plt.ylabel("Number of Images")
        plt.xticks(rotation=0)
        plt.tight_layout()
        os.makedirs("../results", exist_ok=True)
        plt.savefig("../results/class_distribution.png")
        plt.show()
        print(counts)

    def get_data_generators(self):
        # augmentation only on training data - val/test only gets normalized
        train_datagen = ImageDataGenerator(
            rescale=1.0/255.0,
            rotation_range=20,
            width_shift_range=0.15,
            height_shift_range=0.15,
            shear_range=0.1,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode="nearest"
        )
        val_test_datagen = ImageDataGenerator(rescale=1.0/255.0)

        train_generator = train_datagen.flow_from_directory(
            self.train_dir,
            target_size=self.img_size,
            batch_size=self.batch_size,
            class_mode="binary",
            shuffle=True
        )
        val_generator = val_test_datagen.flow_from_directory(
            self.val_dir,
            target_size=self.img_size,
            batch_size=self.batch_size,
            class_mode="binary",
            shuffle=False
        )
        test_generator = val_test_datagen.flow_from_directory(
            self.test_dir,
            target_size=self.img_size,
            batch_size=self.batch_size,
            class_mode="binary",
            shuffle=False
        )
        return train_generator, val_generator, test_generator

    def visualize_samples(self,df, num_samples=6):
        #visually verifying images look correct before training
        fig, axes = plt.subplots(2,3, figsize=(10,7))
        axes=axes.flatten()

        sample_df = df.sample(n=num_samples, random_state=7)

        for i,(_,row) in enumerate(sample_df.iterrows()):
            img=cv2.imread(row["image_path"])
            img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
            img=cv2.resize(img,self.img_size)
            axes[i].imshow(img)
            axes[i].set_title(row["label"])
            axes[i].axis('off')

        plt.suptitle("Sample Images from Dataset", fontsize=14)
        plt.tight_layout()
        os.makedirs("../results", exist_ok=True)
        plt.savefig("../results/sample_images.png")
        plt.show()
        print("Sample visualizations saved")

#Implementing the main() function
def main():
    print("IWMI Data Science Internship Assessment, I'm not a data scientist")
    prep = BasicPreprocessing()
    df=prep.import_dataset()
    prep.check_class_distribution(df)
    prep.visualize_samples(df)
    prep.split_and_copy_dataset(df)
    print("Preprocessing done. Data splits ready")

main()