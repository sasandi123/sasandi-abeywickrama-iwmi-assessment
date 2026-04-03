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


with st.sidebar:
    st.title("Model Information")
    st.markdown("---")
    st.subheader("Architecture Summary")
    st.markdown("""
    **Type:** Custom CNN (built from scratch)

    **Input Size:** 128 x 128 x 3

    **Classes:** With Mask / Without Mask

    **Layers:**
    - 3 Conv Blocks (32 to 64 to 128 filters)
    - BatchNormalization after each Conv
    - MaxPooling (2x2) per block
    - Dropout (0.25 to 0.40)
    - GlobalAveragePooling2D
    - Dense(256) -> Dense(128) -> Dense(1)
    - Sigmoid output
    """)
    st.markdown("---")
    st.subheader("Achieved Accuracy")
    st.markdown("""
    | Metric | Value |
    |--------|-------|
    | Training Accuracy | 99.13% |
    | Validation Accuracy | 98.76% |
    | Test Accuracy | update after inference |
    | AUC-ROC | update after inference |
    """)
    st.caption("Run inference.py for exact test metrics.")
    st.markdown("---")
    st.caption("IWMI Data Science Internship Assessment")
