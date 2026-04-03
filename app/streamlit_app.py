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
    | Test Accuracy | 99% |
    | AUC-ROC | 0.9971 |
    """)
    st.markdown("---")
    st.caption("IWMI Data Science Internship Assessment")

    st.title("Face Mask Detection")
    st.markdown("Upload an image (.jpg, .jpeg, or .png) to classify whether the person is wearing a mask.")
    st.markdown("---")

    model = load_trained_model()

    if model is None:
        st.error(
            "Trained model not found at models/best_model.keras. "
            "Please run src/model.py first to train and save the model."
        )
        st.stop()

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        pil_image = Image.open(uploaded_file).convert("RGB")
        image_array = np.array(pil_image)

        col1, col2 = st.columns([1, 1])

        with col1:
            st.subheader("Uploaded Image")
            st.image(pil_image, use_container_width=True)

        with col2:
            st.subheader("Prediction Result")

            with st.spinner("Classifying..."):
                pred_class, conf_with, conf_without, raw_score = predict(model, image_array)

            if pred_class == "With Mask":
                st.success(f"Prediction: {pred_class}")
                st.metric("Confidence", f"{conf_with}%")
            else:
                st.error(f"Prediction: {pred_class}")
                st.metric("Confidence", f"{conf_without}%")

            st.markdown("---")
            st.markdown("**Confidence Breakdown:**")
            st.write(f"- With Mask: {conf_with}%")
            st.write(f"- Without Mask: {conf_without}%")
            st.write(f"- Raw score: {raw_score:.4f}")
