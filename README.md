# Face Mask Detection - IWMI Data Science Internship Assessment

This is my submission for the IWMI Data Science Internship technical assessment. The goal was to build a custom CNN model from scratch to classify whether a person in an image is wearing a face mask or not, and deploy it as a Streamlit web app.

---

## What I built

- A custom CNN (no pretrained models) trained on the provided dataset
- Data preprocessing pipeline with augmentation
- Model evaluation with confusion matrix, ROC curve, and classification report
- Face detection using OpenCV Haar Cascade
- A Streamlit web app where you can upload an image and get a prediction

---

## Project Structure

```
sasandi-abeywickrama-iwmi-assessment/
├── src/
│   ├── preprocessing.py       # data loading, splitting, augmentation
│   ├── model.py               # CNN architecture and training
│   └── inference.py           # evaluation and face detection
├── app/
│   └── streamlit_app.py       # web application
├── models/
│   └── best_model.keras       # saved model weights
├── results/
│   ├── training_curves.png
│   ├── confusion_matrix.png
│   ├── roc_curve.png
│   ├── sample_images.png
│   ├── class_distribution.png
│   └── analysis_report.txt
├── dataset/                   # not pushed - too large, download link below
├── README.md
├── requirements.txt
└── .gitignore
```

---

## How to Run

**1. Clone the repo**
```bash
git clone https://github.com/sasandi123/sasandi-abeywickrama-iwmi-assessment.git
cd sasandi-abeywickrama-iwmi-assessment
```

**2. Create a virtual environment with Python 3.11**
```bash
py -3.11 -m venv .venv
.venv\Scripts\activate
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

**4. Download the dataset**

Download from [this link](https://drive.google.com/file/d/1Dw0DGHwdmiblqzk8u1LeMzMCqo87sJhN/view?usp=sharing) and extract it.

**5. Run preprocessing**
```bash
cd src
python preprocessing.py
```

**6. Train the model**
```bash
python model.py
```
This saves the best model to `models/best_model.keras` and saves training plots to `results/`.

**7. Run evaluation**
```bash
python inference.py
```

**8. Launch the web app**
```bash
cd ..
streamlit run app/streamlit_app.py
```
Then open `http://localhost:8501` in your browser.

---

## Model Summary

Built a custom CNN with 3 convolutional blocks from scratch. No transfer learning used.

- Input: 128x128 RGB image
- 3 Conv blocks (32 → 64 → 128 filters), each with BatchNorm, MaxPooling, and Dropout
- GlobalAveragePooling → Dense(256) → Dense(128) → Dense(1, sigmoid)
- Optimizer: Adam with ReduceLROnPlateau
- Loss: Binary Crossentropy

---

## Results

| Metric | Value |
|--------|-------|
| Training Accuracy | 99.13% |
| Validation Accuracy | 98.76% |
| Test Accuracy | 99% |
| F1-Score | 0.99 |
| AUC-ROC | 0.9971 |

---

## Demo

[Screen Recording Demo](https://drive.google.com/file/d/14PxvrKtntYb6b2ZNoRdZjI3HChI_Vr2J/view?usp=sharing)

---

*Sasandi Abeywickrama — IWMI Data Science Internship Assessment*