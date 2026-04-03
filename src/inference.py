# Importing required libraries
import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Additional libraries
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

    def detect_images(self):
        # detecting faces using haar cascade then classifying each face
        image_path = input("Enter the path to the image: ").strip()

        if not os.path.exists(image_path):
            print("Image not found.")
            return

        img_bgr = cv2.imread(image_path)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

        faces = self.face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60)
        )

        if len(faces) == 0:
            print("No faces detected in the image.")
            return

        result_img = img_rgb.copy()

        for (x, y, w, h) in faces:
            face_roi = img_rgb[y:y + h, x:x + w]
            face_resized = cv2.resize(face_roi, self.img_size)
            face_normalized = face_resized / 255.0
            face_input = np.expand_dims(face_normalized, axis=0)

            prediction = self.model.predict(face_input, verbose=0)[0][0]

            # sigmoid < 0.5 = with_mask, >= 0.5 = without_mask
            if prediction < 0.5:
                label = "With Mask"
                confidence = (1 - prediction) * 100
                color = (0, 200, 0)
            else:
                label = "No Mask"
                confidence = prediction * 100
                color = (220, 0, 0)

            cv2.rectangle(result_img, (x, y), (x + w, y + h), color, 2)
            cv2.putText(
                result_img,
                f"{label} ({confidence:.1f}%)",
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65, color, 2
            )
            print(f"  → {label} | Confidence: {confidence:.2f}%")

        os.makedirs("../results", exist_ok=True)
        plt.figure(figsize=(8, 6))
        plt.imshow(result_img)
        plt.axis("off")
        plt.title(f"Detection Result | {len(faces)} face(s) found")
        plt.tight_layout()
        plt.savefig("../results/detection_result.png", dpi=150)
        plt.show()

        return result_img, faces

    def predict_single_image(self, image_path):
        # programmatic version - used by the streamlit app
        img = cv2.imread(image_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, self.img_size)
        img_normalized = img_resized / 255.0
        img_input = np.expand_dims(img_normalized, axis=0)

        prediction = self.model.predict(img_input, verbose=0)[0][0]

        result = {
            "prediction": "With Mask" if prediction < 0.5 else "Without Mask",
            "confidence_with_mask": round((1 - prediction) * 100, 2),
            "confidence_without_mask": round(prediction * 100, 2),
            "raw_score": float(prediction)
        }
        return result

    def evaluate_on_test_set(self, test_generator):
        # full evaluation - accuracy alone is not enough, using f1 and auc as well
        print("Running evaluation on test set...")
        test_generator.reset()

        y_pred_probs = self.model.predict(test_generator, verbose=1)
        y_pred = (y_pred_probs >= 0.5).astype(int).flatten()
        y_true = test_generator.classes

        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred)
        rec = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        auc = roc_auc_score(y_true, y_pred_probs)

        print(f"\n--- Evaluation Metrics ---")
        print(f"Accuracy  : {acc:.4f}")
        print(f"Precision : {prec:.4f}")
        print(f"Recall    : {rec:.4f}")
        print(f"F1-Score  : {f1:.4f}")
        print(f"AUC-ROC   : {auc:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_true, y_pred, target_names=self.class_names))

        return y_true, y_pred, y_pred_probs



