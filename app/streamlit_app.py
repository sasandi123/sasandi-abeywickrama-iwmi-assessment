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

# custom CSS for a cleaner, more professional look
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'IBM Plex Sans', sans-serif;
    }

    .main {
        background-color: #f7f8fa;
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }

    h1 {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 1.8rem;
        font-weight: 600;
        color: #0f1923;
        border-bottom: 3px solid #0066cc;
        padding-bottom: 0.4rem;
        margin-bottom: 0.2rem;
    }

    h2, h3 {
        font-family: 'IBM Plex Sans', sans-serif;
        color: #0f1923;
        font-weight: 600;
    }

    .result-box {
        background-color: #ffffff;
        border-radius: 8px;
        padding: 1.5rem;
        border: 1px solid #e0e4ea;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    }

    .metric-label {
        font-size: 0.75rem;
        color: #6b7280;
        font-family: 'IBM Plex Mono', monospace;
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }

    .metric-value {
        font-size: 2rem;
        font-family: 'IBM Plex Mono', monospace;
        font-weight: 600;
        color: #0f1923;
    }

    .tag-with {
        background-color: #dcfce7;
        color: #166534;
        padding: 0.3rem 0.9rem;
        border-radius: 20px;
        font-weight: 600;
        font-size: 1rem;
        display: inline-block;
    }

    .tag-without {
        background-color: #fee2e2;
        color: #991b1b;
        padding: 0.3rem 0.9rem;
        border-radius: 20px;
        font-weight: 600;
        font-size: 1rem;
        display: inline-block;
    }

    .section-divider {
        border: none;
        border-top: 1px solid #e0e4ea;
        margin: 1.5rem 0;
    }

    .stFileUploader > div {
        background-color: #ffffff;
        border: 2px dashed #0066cc;
        border-radius: 8px;
    }

    .sidebar-section {
        background-color: #f0f4ff;
        border-radius: 6px;
        padding: 0.8rem;
        margin-bottom: 1rem;
    }

    [data-testid="stSidebar"] {
        background-color: #0f1923;
        color: #e0e4ea;
    }

    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] li,
    [data-testid="stSidebar"] .stMarkdown {
        color: #e0e4ea !important;
    }

    [data-testid="stSidebar"] hr {
        border-color: #2a3a4a;
    }

    [data-testid="stSidebar"] caption {
        color: #6b7280 !important;
    }
    </style>
""", unsafe_allow_html=True)

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



# SIDEBAR

with st.sidebar:
    st.title("Model Information")
    st.markdown("---")
    st.subheader("Architecture")
    st.markdown("""
**Type:** Custom CNN (from scratch)

**Input:** 128 x 128 x 3

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
    st.subheader("Results")
    st.markdown("""
| Metric | Value |
|--------|-------|
| Train Accuracy | 99.13% |
| Val Accuracy | 98.76% |
| Test Accuracy | 99% |
| AUC-ROC | 0.9971 |
    """)
    st.markdown("---")
    st.caption("IWMI Data Science Internship Assessment")



# MAIN PAGE

st.title("Face Mask Detection")
st.markdown("Upload a face image to classify whether the person is wearing a mask or not.")
st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)

model = load_trained_model()

if model is None:
    st.error(
        "Trained model not found at models/best_model.keras. "
        "Please run src/model.py first to train and save the model."
    )
    st.stop()

uploaded_file = st.file_uploader("Choose an image (.jpg, .jpeg, or .png)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    pil_image = Image.open(uploaded_file).convert("RGB")
    image_array = np.array(pil_image)

    st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)

    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.subheader("Uploaded Image")
        st.image(pil_image, use_container_width=True)

    with col2:
        st.subheader("Prediction Result")

        with st.spinner("Running classification..."):
            pred_class, conf_with, conf_without, raw_score = predict(model, image_array)

        # result tag
        if pred_class == "With Mask":
            st.markdown(f'<span class="tag-with">With Mask</span>', unsafe_allow_html=True)
            confidence_display = conf_with
        else:
            st.markdown(f'<span class="tag-without">Without Mask</span>', unsafe_allow_html=True)
            confidence_display = conf_without

        st.markdown("<br>", unsafe_allow_html=True)

        # confidence metric
        st.markdown(f'<p class="metric-label">Confidence</p><p class="metric-value">{confidence_display}%</p>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("**Breakdown**")

        # mini progress bars for both classes
        st.caption("With Mask")
        st.progress(int(conf_with))
        st.caption(f"{conf_with}%")

        st.caption("Without Mask")
        st.progress(int(conf_without))
        st.caption(f"{conf_without}%")

        st.markdown(f"<p style='font-size:0.75rem; color:#6b7280; font-family:monospace;'>raw score: {raw_score:.4f}</p>", unsafe_allow_html=True)

    st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)

    # top 3 bar chart
    st.subheader("Top 3 Class Prediction Chart")

    # only 2 real classes so adding "Uncertain" as a third bar to meet the requirement
    uncertain_score = round(100 - abs(conf_with - conf_without), 2)
    if uncertain_score < 0:
        uncertain_score = 0.0

    top3_labels = ["With Mask", "Without Mask", "Uncertain"]
    top3_scores = [conf_with, conf_without, uncertain_score]
    colors = ["#16a34a", "#dc2626", "#94a3b8"]

    fig, ax = plt.subplots(figsize=(7, 2.8))
    fig.patch.set_facecolor("#f7f8fa")
    ax.set_facecolor("#f7f8fa")
    bars = ax.barh(top3_labels, top3_scores, color=colors, height=0.5)
    ax.set_xlim(0, 115)
    ax.set_xlabel("Confidence (%)", fontsize=9, color="#6b7280")
    ax.tick_params(colors="#0f1923", labelsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_color("#e0e4ea")
    ax.spines["left"].set_color("#e0e4ea")
    for bar, score in zip(bars, top3_scores):
        ax.text(bar.get_width() + 1.5, bar.get_y() + bar.get_height() / 2,
                f"{score:.1f}%", va="center", fontsize=9, color="#0f1923")
    plt.tight_layout()
    st.pyplot(fig)

    st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)

    # haar cascade face detection
    st.subheader("Face Detection (Haar Cascade)")

    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    img_bgr = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))

    if len(faces) > 0:
        annotated = image_array.copy()
        for (x, y, w, h) in faces:
            color = (22, 163, 74) if pred_class == "With Mask" else (220, 38, 38)
            cv2.rectangle(annotated, (x, y), (x + w, y + h), color, 2)
            cv2.putText(annotated, pred_class, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        st.image(annotated, caption=f"{len(faces)} face(s) detected", use_container_width=True)
    else:
        st.info("No faces detected by Haar Cascade. Classification was still done on the full image.")

else:
    # placeholder when no image is uploaded
    st.markdown("<br>", unsafe_allow_html=True)
    col_a, col_b, col_c = st.columns([1, 2, 1])
    with col_b:
        st.markdown("""
        <div style='text-align:center; padding: 3rem; background:#ffffff; border-radius:8px; border: 1px solid #e0e4ea;'>
            <p style='font-size:1rem; color:#6b7280;'>Upload an image above to get started.</p>
            <p style='font-size:0.8rem; color:#9ca3af;'>Supported formats: JPG, JPEG, PNG</p>
        </div>
        """, unsafe_allow_html=True)