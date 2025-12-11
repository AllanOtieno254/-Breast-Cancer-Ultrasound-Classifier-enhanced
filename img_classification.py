# img_classification.py
import os
import numpy as np
from keras.models import load_model
from PIL import Image, ImageOps
import cv2
import sqlite3
import json
import datetime

# --- CONFIG ---
MODEL_FILENAME = "keras_model.h5"       # inside model/ directory
LABEL_FILENAME = "labels.txt"
MODEL_DIR = os.path.join(os.path.dirname(__file__), "model")
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_FILENAME)
LABEL_PATH = os.path.join(MODEL_DIR, LABEL_FILENAME)

INPUT_SIZE = (224, 224)   # model input size; adjust if different

# Confidence thresholds (tunable)
THRESH_HIGH_SUSPICION = 0.70  # >= -> recommend biopsy strongly
THRESH_HIGH_CONFIDENCE_BENIGN = 0.90
THRESH_LOW_CONFIDENCE = 0.55

# Database file for storing history
DB_PATH = os.path.join(os.path.dirname(__file__), "predictions.db")

# --- HELPERS: DB ---
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
        image_paths TEXT,         -- JSON list
        prediction TEXT,
        probs TEXT,               -- JSON
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

# --- Load model and labels ---
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model not found at {MODEL_PATH}. Place {MODEL_FILENAME} in {MODEL_DIR}")

if not os.path.exists(LABEL_PATH):
    raise FileNotFoundError(f"Labels not found at {LABEL_PATH}")

model = load_model(MODEL_PATH, compile=False)

with open(LABEL_PATH, "r", encoding="utf-8") as f:
    raw_class_names = [line.strip() for line in f.readlines()]

# Normalize class names (take last token)
class_names = [name.split()[-1] if len(name.split()) > 1 else name for name in raw_class_names]
# We'll select Malignant and Benign indices robustly
selected_classes = ["Malignant", "Benign"]
selected_indices = []
for target in selected_classes:
    for i, label in enumerate(class_names):
        if target.lower() in label.lower():
            selected_indices.append(i)
            break
if len(selected_indices) != 2:
    raise ValueError(f"Could not find both Malignant and Benign in labels.txt. Found: {class_names}")

# --- Preprocessing ---
def preprocess_pil(image: Image.Image):
    """
    Resize, crop and normalize to model input.
    Returns numpy array (1, H, W, 3)
    """
    image = ImageOps.fit(image.convert("RGB"), INPUT_SIZE, Image.Resampling.LANCZOS)
    arr = np.asarray(image).astype(np.float32)
    # normalization same as Teachable Machine: [-1,1]
    arr = (arr / 127.5) - 1.0
    return np.expand_dims(arr, axis=0)

# --- Prediction ---
def teachable_machine_classification(image: Image.Image):
    """
    Returns:
      pred_idx (0 malignant or 1 benign relative to selected_classes),
      normalized_probs: np.array([malignant_prob, benign_prob])
    """
    x = preprocess_pil(image)
    raw_prediction = model.predict(x)[0]
    filtered = np.array([raw_prediction[i] for i in selected_indices], dtype=np.float32)
    total = filtered.sum()
    if total <= 0 or np.isnan(total):
        filtered = filtered + 1e-6
        total = filtered.sum()
    normalized = filtered / total
    pred_idx = int(np.argmax(normalized))
    return pred_idx, normalized

# --- Grad-CAM (basic implementation) ---
def grad_cam_for_pil(image: Image.Image, layer_name: str = None):
    """
    Produces a heatmap (PIL Image) overlayed on original image.
    If layer_name not provided, tries to find last conv layer.
    Returns (overlay_pil, heatmap_gray) where heatmap_gray is single-channel PIL.
    """
    # Convert PIL -> array & preprocess
    orig = image.convert("RGB")
    x = preprocess_pil(orig)

    # find conv layer
    if layer_name is None:
        # heuristic: choose last layer with 4D output (conv)
        for layer in reversed(model.layers):
            out_shape = getattr(layer.output, "shape", None)
            if out_shape is not None and len(out_shape) == 4:
                layer_name = layer.name
                break
    if layer_name is None:
        raise ValueError("Could not find conv layer for Grad-CAM. Please provide a layer_name.")

    # Build grad model using tf backend
    try:
        import tensorflow as tf
        grad_model = tf.keras.models.Model(
            [model.inputs], [model.get_layer(layer_name).output, model.output]
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
        # relu
        heatmap = np.maximum(heatmap, 0)
        if heatmap.max() == 0:
            heatmap = heatmap
        else:
            heatmap /= np.max(heatmap)

        # resize heatmap to original image size
        heatmap = cv2.resize(heatmap, INPUT_SIZE[::-1])
        heatmap_uint8 = np.uint8(255 * heatmap)
        heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
        orig_np = np.asarray(orig.resize(INPUT_SIZE))
        overlay = cv2.addWeighted(orig_np, 0.6, heatmap_color, 0.4, 0)

        overlay_pil = Image.fromarray(overlay)
        heatmap_gray = Image.fromarray(heatmap_uint8).convert("L")
        return overlay_pil, heatmap_gray
    except Exception as e:
        raise RuntimeError(f"Grad-CAM generation failed: {e}")

# --- Image quality check (blurriness via Laplacian) ---
def is_image_blurry(pil_img: Image.Image, threshold: float = 100.0):
    arr = np.array(pil_img.convert("L"))
    lap = cv2.Laplacian(arr, cv2.CV_64F)
    var = lap.var()
    return var < threshold, var  # returns (is_blurry, variance)

# --- Model metadata helper ---
def get_model_metadata():
    meta = {
        "model_path": MODEL_PATH,
        "input_size": INPUT_SIZE,
        "labels": class_names,
        "selected_labels": selected_classes,
        "keras_model_layers": [layer.name for layer in model.layers],
        "loaded_at": datetime.datetime.utcnow().isoformat() + "Z"
    }
    return meta

# --- Utility: prepare result object ---
def prepare_result_record(patient_info: dict, image_paths: list, prediction_label: str, probs: list, explanation: str):
    record = {
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
    return record
