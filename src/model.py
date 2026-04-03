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

    def compile_model(self):
        #adam with slightly low lr to be safe at the start
        self.model.compile(
            optimizer=Adam(learning_rate=1e-3),
            loss="binary_crossentropy",
            metrics=["accuracy"]
        )
        print("Model compiled")

    def get_callbacks(self):
        #saving best model, reducing lr when stuck, stopping early if no progress
        os.makedirs("models", exist_ok=True)

        checkpoint = ModelCheckpoint(
            filepath="models/best_model.keras",
            monitor="val_accuracy",
            save_best_only=True,
            mode="max",
            verbose=1
        )

        lr_scheduler = ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=4,
            min_lr=1e-6,
            verbose=1
        )

        early_stop=EarlyStopping(
            monitor="val_loss",
            patience=10,
            restore_best_weights=True,
            verbose=1
        )

        return[checkpoint, lr_scheduler, early_stop]

    def train_model(self,train_generator, val_generator, epochs=40):
        if self.model is None:
            print("Model not built yet. Call build_model() first.")
            return

        callbacks = self.get_callbacks()

        print("Starting training...")
        self.history = self.model.fit(
            train_generator,
            validation_data=val_generator,
            epochs=epochs,
            callbacks=callbacks,
            verbose=1
        )
        print("Training complete.")
        return self.history

    def plot_training_curves(self):
        #plotting loss and accuracy over epochs to see if model trained well
        if self.history is None:
            print("No history found. Train first.")
            return

        fig, (ax1,ax2) = plt.subplots(1,2,figsize=(14,5))

        ax1.plot(self.history.history['accuracy'], label="Train Accuracy", color="blue")
        ax1.plot(self.history.history['val_accuracy'], label="Validation Accuracy", color="orange")
        ax1.set_title("Training vs Validation Accuracy")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Accuracy")
        ax1.legend()
        ax1.grid(True)

        ax2.plot(self.history.history['loss'], label="Train Loss", color="blue")
        ax2.plot(self.history.history['val_loss'], label="Validation Loss", color="orange")
        ax2.set_title("Training vs Validation Loss")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Loss")
        ax2.legend()
        ax2.grid(True)

        plt.suptitle("Model Training Curves", fontsize=14)
        plt.tight_layout()

        os.makedirs("results", exist_ok=True)
        plt.savefig("results/training_curves.png",dpi=150)
        plt.show()
        print("Training curves saved.")

def main():
    print("IWMI Data Science Internship Assessment, I'm not a data scientist")

    from preprocessing import BasicPreprocessing
    prep = BasicPreprocessing()
    df = prep.import_dataset()
    prep.split_and_copy_dataset(df)
    train_gen, val_gen, _ = prep.get_data_generators()

    dev = ModelDevelopment()
    dev.build_model()
    dev.compile_model()
    dev.train_model(train_gen, val_gen, epochs=40)
    dev.plot_training_curves()


main()





