# Importing required libraries
import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Any additional libraries go under here
from tensorflow.keras.models import load_model
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, roc_curve, accuracy_score,
    precision_score, recall_score, f1_score
)
import seaborn as sns


class BasicInference:

    def __init__(self):
        self.model = None
        self.img_size = (128, 128)
        self.class_names = ["with_mask", "without_mask"]
        # haar cascade for face detection
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )

    def load_trained_model(self, model_path="../models/best_model.keras"):
        # loading saved model from training
        if not os.path.exists(model_path):
            print(f"Model not found at {model_path}. Train first.")
            return
        self.model = load_model(model_path)
        print(f"Model loaded from {model_path}")