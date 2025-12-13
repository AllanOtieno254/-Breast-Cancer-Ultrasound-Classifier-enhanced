# img_classification.py
import os
import json
import datetime
import sqlite3
import requests

import numpy as np
from keras.models import load_model
from PIL import Image, ImageOps
import cv2

# =====================================================
# MODEL AUTO-DOWNLOAD FROM GOOGLE DRIVE (STREAMLIT SAFE)
# =====================================================

MODEL_DIR = os.path.join(os.path.dirname(__file__), "model")
MODEL_FILENAME = "keras_model.h5"
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_FILENAME)

MODEL_URL = "https://drive.google.com/uc?export=download&id=12GIG-tD7YzWiWD9SLPhoBK-gsuseew74"

os.makedirs(MODEL_DIR, exist_ok=True)

if not os.path.exists(MODEL_PATH):
    with open(MODEL_PATH, "wb") as f:
        response = requests.get(MODEL_URL, stream=True)
        response.raise_for_status()
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)

# =====================================================
# OTHER CONFIG
# =====================================================

LABEL_FILENAME = "labels.txt"
LABEL_PATH = os.path.join(MODEL_DIR, LABEL_FILENAME)

INPUT_SIZE = (224, 224)

THRESH_HIGH_SUSPICION = 0.70
THRESH_HIGH_CONFIDENCE_BENIGN = 0.90
THRESH_LOW_CONFIDENCE = 0.55

DB_PATH = os.path.join(os.path.dirname(__file__), "predictions.db")

# =====================================================
# DATABASE HELPERS
# =====================================================

def ensure_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
    CREATE TABLE IF NOT EXISTS predictions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT,
        patient_name TEXT,
        patient_age INTEGER,
        patient_gender TEXT,
        notes TEXT,
        image_paths TEXT,
        prediction TEXT,
        probs TEXT,
        explanation TEXT,
        model_metadata TEXT
    )
    """)
    conn.commit()
    conn.close()

def insert_record(record: dict):
    ensure_db()
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
    INSERT INTO predictions (
        timestamp, patient_name, patient_age, patient_gender, notes,
        image_paths, prediction, probs, explanation, model_metadata
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        record.get("timestamp"),
        record.get("patient_name"),
        record.get("patient_age"),
        record.get("patient_gender"),
        record.get("notes"),
        json.dumps(record.get("image_paths", [])),
        record.get("prediction"),
        json.dumps(record.get("probs")),
        record.get("explanation"),
        json.dumps(record.get("model_metadata"))
    ))
    conn.commit()
    conn.close()

# =====================================================
# LOAD MODEL & LABELS
# =====================================================

if not os.path.exists(LABEL_PATH):
    raise FileNotFoundError(f"Labels not found at {LABEL_PATH}")

model = load_model(MODEL_PATH, compile=False)

with open(LABEL_PATH, "r", encoding="utf-8") as f:
    raw_class_names = [line.strip() for line in f.readlines()]

class_names = [name.split()[-1] if len(name.split()) > 1 else name for name in raw_class_names]

selected_classes = ["Malignant", "Benign"]
selected_indices = []

for target in selected_classes:
    for i, label in enumerate(class_names):
        if target.lower() in label.lower():
            selected_indices.append(i)
            break

if len(selected_indices) != 2:
    raise ValueError(f"Could not find both Malignant and Benign in labels.txt. Found: {class_names}")

# =====================================================
# PREPROCESSING
# =====================================================

def preprocess_pil(image: Image.Image):
    image = ImageOps.fit(image.convert("RGB"), INPUT_SIZE, Image.Resampling.LANCZOS)
    arr = np.asarray(image).astype(np.float32)
    arr = (arr / 127.5) - 1.0
    return np.expand_dims(arr, axis=0)

# =====================================================
# PREDICTION
# =====================================================

def teachable_machine_classification(image: Image.Image):
    x = preprocess_pil(image)
    raw_prediction = model.predict(x)[0]

    filtered = np.array([raw_prediction[i] for i in selected_indices], dtype=np.float32)
    total = filtered.sum()

    if total <= 0 or np.isnan(total):
        filtered += 1e-6
        total = filtered.sum()

    normalized = filtered / total
    pred_idx = int(np.argmax(normalized))
    return pred_idx, normalized

# =====================================================
# GRAD-CAM
# =====================================================

def grad_cam_for_pil(image: Image.Image, layer_name: str = None):
    orig = image.convert("RGB")
    x = preprocess_pil(orig)

    if layer_name is None:
        for layer in reversed(model.layers):
            shape = getattr(layer.output, "shape", None)
            if shape is not None and len(shape) == 4:
                layer_name = layer.name
                break

    if layer_name is None:
        raise ValueError("No convolutional layer found for Grad-CAM.")

    import tensorflow as tf

    grad_model = tf.keras.models.Model(
        [model.inputs],
        [model.get_layer(layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(tf.convert_to_tensor(x))
        pred_index = tf.argmax(predictions[0])
        loss = predictions[:, pred_index]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0].numpy()
    pooled_grads = pooled_grads.numpy()

    for i in range(pooled_grads.shape[-1]):
        conv_outputs[:, :, i] *= pooled_grads[i]

    heatmap = np.mean(conv_outputs, axis=-1)
    heatmap = np.maximum(heatmap, 0)

    if heatmap.max() != 0:
        heatmap /= heatmap.max()

    heatmap = cv2.resize(heatmap, INPUT_SIZE[::-1])
    heatmap_uint8 = np.uint8(255 * heatmap)

    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    orig_np = np.asarray(orig.resize(INPUT_SIZE))

    overlay = cv2.addWeighted(orig_np, 0.6, heatmap_color, 0.4, 0)

    return Image.fromarray(overlay), Image.fromarray(heatmap_uint8).convert("L")

# =====================================================
# IMAGE QUALITY CHECK
# =====================================================

def is_image_blurry(pil_img: Image.Image, threshold: float = 100.0):
    arr = np.array(pil_img.convert("L"))
    lap = cv2.Laplacian(arr, cv2.CV_64F)
    return lap.var() < threshold, lap.var()

# =====================================================
# MODEL METADATA
# =====================================================

def get_model_metadata():
    return {
        "model_path": MODEL_PATH,
        "input_size": INPUT_SIZE,
        "labels": class_names,
        "selected_labels": selected_classes,
        "keras_model_layers": [layer.name for layer in model.layers],
        "loaded_at": datetime.datetime.utcnow().isoformat() + "Z"
    }

# =====================================================
# RESULT RECORD
# =====================================================

def prepare_result_record(patient_info: dict, image_paths: list, prediction_label: str, probs: list, explanation: str):
    return {
        "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
        "patient_name": patient_info.get("name"),
        "patient_age": patient_info.get("age"),
        "patient_gender": patient_info.get("gender"),
        "notes": patient_info.get("notes"),
        "image_paths": image_paths,
        "prediction": prediction_label,
        "probs": [float(p) for p in probs],
        "explanation": explanation,
        "model_metadata": get_model_metadata()
    }
