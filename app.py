# app.py (final version: single image upload, CSV only)
import streamlit as st
from PIL import Image, ImageEnhance
from img_classification import (
    teachable_machine_classification,
    grad_cam_for_pil,
    is_image_blurry,
    prepare_result_record,
    insert_record,
    get_model_metadata,
)
import base64
import os
import requests
from dotenv import load_dotenv
from io import BytesIO
import json
import matplotlib.pyplot as plt
import tempfile
import sqlite3
import pandas as pd
from datetime import datetime

load_dotenv()

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
DEEPSEEK_BASE = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")

st.set_page_config(page_title="Breast Cancer Classifier", layout="wide")

DB_PATH = os.path.join(os.path.dirname(__file__), "predictions.db")

# ---------------------------
# Background Image Utility
# ---------------------------
def add_bg_image(image_file):
    img = Image.open(image_file).convert("RGB")
    enhancer = ImageEnhance.Brightness(img)
    img_dark = enhancer.enhance(0.35)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_file:
        img_dark.save(tmp_file.name, format="PNG")
        tmp_path = tmp_file.name
    with open(tmp_path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode()
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/png;base64,{encoded}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

add_bg_image(os.path.join("images", "bg1.jpg"))

# ---------------------------
# DATABASE SCHEMA ENSURE
# ---------------------------
REQUIRED_COLUMNS = {
    "id": "INTEGER PRIMARY KEY AUTOINCREMENT",
    "timestamp": "TEXT",
    "patient_name": "TEXT",
    "patient_age": "INTEGER",
    "patient_gender": "TEXT",
    "notes": "TEXT",
    "image_paths": "TEXT",
    "prediction": "TEXT",
    "probs": "TEXT",
    "explanation": "TEXT",
    "model_metadata": "TEXT",
    "psno": "TEXT"
}

def ensure_table_and_columns():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='predictions'")
    if not c.fetchone():
        cols_sql = ",\n    ".join([f"{col} {REQUIRED_COLUMNS[col]}" for col in REQUIRED_COLUMNS])
        create_sql = f"CREATE TABLE predictions (\n    {cols_sql}\n)"
        c.execute(create_sql)
        conn.commit()
        conn.close()
        return
    c.execute("PRAGMA table_info(predictions)")
    existing = [row[1] for row in c.fetchall()]
    for col, coldef in REQUIRED_COLUMNS.items():
        if col not in existing:
            try:
                c.execute(f"ALTER TABLE predictions ADD COLUMN {col} {coldef}")
            except Exception:
                pass
    conn.commit()
    conn.close()

ensure_table_and_columns()

# ---------------------------
# SIDEBAR: PATIENT INFO
# ---------------------------
st.sidebar.header("Patient Details")

if 'psno' not in st.session_state:
    st.session_state['psno'] = ""
if 'name' not in st.session_state:
    st.session_state['name'] = ""
if 'age' not in st.session_state:
    st.session_state['age'] = 40
if 'gender' not in st.session_state:
    st.session_state['gender'] = "Female"
if 'symptoms' not in st.session_state:
    st.session_state['symptoms'] = ""

def clear_patient_details():
    st.session_state['psno'] = ""
    st.session_state['name'] = ""
    st.session_state['age'] = 40
    st.session_state['gender'] = "Female"
    st.session_state['symptoms'] = ""

with st.sidebar.form("patient_form"):
    psno = st.text_input("Patient PSNO (unique ID)", value=st.session_state['psno'])
    name = st.text_input("Patient name", value=st.session_state['name'])
    age = st.number_input("Age", min_value=0, max_value=130, value=st.session_state['age'])
    gender = st.selectbox("Gender", ["Female", "Male", "Other"], index=["Female","Male","Other"].index(st.session_state['gender']))
    symptoms = st.text_area("Clinical notes (symptoms)", value=st.session_state['symptoms'])
    col1, col2 = st.columns([1,1])
    with col1:
        submit_patient = st.form_submit_button("Save patient info")
    with col2:
        clear_patient = st.form_submit_button("Clear patient details", on_click=clear_patient_details)

if submit_patient:
    if not psno:
        st.sidebar.error("PSNO is required.")
    else:
        st.session_state['psno'] = psno
        st.session_state['name'] = name
        st.session_state['age'] = age
        st.session_state['gender'] = gender
        st.session_state['symptoms'] = symptoms
        st.sidebar.success("Patient info saved (temporary)")

# ---------------------------
# MODEL METADATA
# ---------------------------
st.sidebar.header("Model Info")
meta = get_model_metadata()
st.sidebar.write(f"Model file: {os.path.basename(meta['model_path'])}")
st.sidebar.write(f"Input size: {meta['input_size']}")
st.sidebar.write(f"Labels: {meta['labels']}")
st.sidebar.write(f"Loaded layers: {len(meta['keras_model_layers'])} layers")

# ---------------------------
# MAIN: UPLOAD + CLASSIFY + GRAD-CAM + LLM + DB
# ---------------------------
st.title("🩺 Breast Cancer Ultrasound Classifier — Enhanced")
st.markdown("Upload an ultrasound image for automated analysis, Grad-CAM, explanation, and DB saving.")

uploaded_file = st.file_uploader("Upload image", type=["jpg","jpeg","png"], accept_multiple_files=False)
exp_mode = st.radio("Explanation mode", ("Doctor", "Patient (plain language)"))

if uploaded_file and psno:
    results = []
    temp_dir = tempfile.mkdtemp(prefix="bcapp2_")
    try:
        image = Image.open(uploaded_file).convert("RGB")
    except Exception:
        st.error("Could not open the uploaded image.")
    else:
        img_path = os.path.join(temp_dir, uploaded_file.name)
        image.save(img_path)

        blurry, lap_var = is_image_blurry(image)
        if blurry:
            st.warning(f"{uploaded_file.name} appears blurry (Laplacian variance={lap_var:.1f}).")

        pred_idx, probs = teachable_machine_classification(image)
        class_names = ["Malignant", "Benign"]
        pred_label = class_names[pred_idx]

        try:
            overlay, heatmap = grad_cam_for_pil(image)
            grad_path = os.path.join(temp_dir, f"gradcam_{uploaded_file.name}")
            overlay.save(grad_path)
        except Exception:
            grad_path = None

        # LLM explanation
        explanation = "LLM disabled (no API key configured)."
        if DEEPSEEK_API_KEY:
            headers = {"Authorization": f"Bearer {DEEPSEEK_API_KEY}", "Content-Type": "application/json"}
            today = datetime.today().strftime("%Y-%m-%d")
            memo_intro = ""
            if exp_mode == "Doctor":
                memo_intro = (
                    f"MEMORANDUM\n\n"
                    f"TO: Dr. James Okello\n"
                    f"FROM: Breast Cancer Medical Assistant\n"
                    f"DATE: {today}\n"
                    f"RE: URGENT REFERRAL - {name}, Age {age}, {gender}"
                )
            payload = {
                "model": DEEPSEEK_MODEL,
                "messages": [
                    {"role": "system", "content": "You are a medical assistant."},
                    {"role": "user", "content": (
                        f"{memo_intro}\n\nA breast ultrasound image was analyzed by a machine learning model.\n"
                        f"Prediction: {pred_label}\n"
                        f"Probabilities: Malignant {probs[0]*100:.2f}%, Benign {probs[1]*100:.2f}%.\n"
                        f"Patient name: {name}\nAge: {age}\nGender: {gender}\nNotes: {symptoms}\n"
                        f"Mode: {exp_mode}"
                    )}
                ],
                "max_tokens": 700,
                "temperature": 0.2
            }
            try:
                r = requests.post(f"{DEEPSEEK_BASE}/chat/completions", headers=headers, json=payload, timeout=30)
                if r.status_code == 200:
                    data = r.json()
                    explanation = data["choices"][0]["message"]["content"].strip()
                else:
                    explanation = f"DeepSeek error: {r.status_code} {r.text}"
            except Exception as e:
                explanation = f"DeepSeek API call failed: {e}"

        recommendation = ""
        if probs[0] >= 0.70:
            recommendation = "High suspicion for malignancy — urgent follow-up required."
        elif probs[1] >= 0.90:
            recommendation = "High confidence benign — routine follow-up."
        elif max(probs) < 0.55:
            recommendation = "Low confidence — further imaging advised."

        # Insert record
        patient_info = {"name": name, "age": age, "gender": gender, "notes": symptoms}
        try:
            rec = prepare_result_record(patient_info, [img_path], pred_label, probs, explanation)
            rec["psno"] = psno
            insert_record(rec)
        except Exception as e:
            st.error(f"DB insert error: {e}")

        results.append({
            "filename": uploaded_file.name,
            "image_path": img_path,
            "gradcam_path": grad_path,
            "prediction": pred_label,
            "probs": [float(probs[0]), float(probs[1])],
            "explanation": explanation,
            "recommendation": recommendation
        })

    # Display results
    st.markdown("### Results")
    for r in results:
        st.markdown(f"**{r['filename']}** — Prediction: **{r['prediction']}**")
        st.write(f"Malignant: {r['probs'][0]*100:.2f}% | Benign: {r['probs'][1]*100:.2f}%")
        col1, col2 = st.columns(2)
        with col1:
            st.image(r["image_path"], caption="Original")
        with col2:
            if r["gradcam_path"]:
                st.image(r["gradcam_path"], caption="Grad-CAM")
        st.info(r["recommendation"])
        with st.expander("Explanation"):
            st.write(r["explanation"])
        fig, ax = plt.subplots(figsize=(4,1.2))
        ax.barh(["Malignant","Benign"], [r['probs'][0]*100, r['probs'][1]*100])
        ax.set_xlim(0,100)
        st.pyplot(fig)
        st.markdown("---")

# ---------------------------
# DATABASE MANAGEMENT
# ---------------------------
st.header("📦 Database Management")

conn = sqlite3.connect(DB_PATH)
df = pd.read_sql_query("SELECT * FROM predictions ORDER BY id DESC", conn)
conn.close()

# Download CSV
if st.button("Download database as CSV"):
    if df.empty:
        st.info("Database is empty.")
    else:
        csv = df.to_csv(index=False).encode()
        b64 = base64.b64encode(csv).decode()
        st.markdown(f'<a href="data:file/csv;base64,{b64}" download="predictions.csv">Download CSV</a>', unsafe_allow_html=True)

# Clear DB
if st.button("Clear entire database"):
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)
        ensure_table_and_columns()
        st.success("Database cleared.")
    else:
        st.info("Database already empty.")

# Show DB table
if st.checkbox("Show database records"):
    if df.empty:
        st.info("No records found.")
    else:
        st.dataframe(df)
