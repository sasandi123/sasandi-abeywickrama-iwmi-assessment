# Importing required libraries
import os
import cv2
import numpy as np
import streamlit as st
from PIL import Image
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

st.set_page_config(
    page_title="Mask Detector | IWMI Assessment",
    layout="wide"
)

MODEL_PATH = os.path.join(os.path.dirname(__file__), "../models/best_model.keras")
IMG_SIZE = (128, 128)
CLASS_NAMES = ["With Mask", "Without Mask"]


@st.cache_resource
def load_trained_model():
    # caching so model doesn't reload on every interaction
    if not os.path.exists(MODEL_PATH):
        return None
    model = load_model(MODEL_PATH)
    return model


def predict(model, image_array):
    # resize, normalize, add batch dim then predict
    img_resized = cv2.resize(image_array, IMG_SIZE)
    img_normalized = img_resized / 255.0
    img_input = np.expand_dims(img_normalized, axis=0)

    raw_score = model.predict(img_input, verbose=0)[0][0]

    confidence_with_mask = round((1 - raw_score) * 100, 2)
    confidence_without_mask = round(raw_score * 100, 2)
    predicted_class = "Without Mask" if raw_score >= 0.5 else "With Mask"

    return predicted_class, confidence_with_mask, confidence_without_mask, raw_score

