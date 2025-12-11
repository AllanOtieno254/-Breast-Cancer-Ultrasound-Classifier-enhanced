# # app.py (merged, schema-migrating version — fixes "no such column" errors)
# import streamlit as st
# from PIL import Image
# from img_classification import (
#     teachable_machine_classification,
#     grad_cam_for_pil,
#     is_image_blurry,
#     prepare_result_record,
#     insert_record,
#     get_model_metadata,
# )
# import base64
# import os
# import requests
# from dotenv import load_dotenv
# from io import BytesIO
# import json
# import matplotlib.pyplot as plt
# import tempfile
# import sqlite3
# import pandas as pd

# load_dotenv()

# DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
# DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
# DEEPSEEK_BASE = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")

# st.set_page_config(page_title="Breast Cancer Classifier", layout="wide")

# DB_PATH = os.path.join(os.path.dirname(__file__), "predictions.db")

# # --------------------------------------------------------------------
# # DATABASE MIGRATION / ENSURE SCHEMA
# # --------------------------------------------------------------------
# REQUIRED_COLUMNS = {
#     "id": "INTEGER PRIMARY KEY AUTOINCREMENT",
#     "timestamp": "TEXT",
#     "patient_name": "TEXT",
#     "patient_age": "INTEGER",
#     "patient_gender": "TEXT",
#     "notes": "TEXT",
#     "image_paths": "TEXT",
#     "prediction": "TEXT",
#     "probs": "TEXT",
#     "explanation": "TEXT",
#     "model_metadata": "TEXT",
#     "psno": "TEXT"
# }

# def ensure_table_and_columns():
#     """
#     Ensures `predictions` table exists and contains all REQUIRED_COLUMNS.
#     If table exists but misses columns, add them via ALTER TABLE.
#     """
#     conn = sqlite3.connect(DB_PATH)
#     c = conn.cursor()

#     # If table doesn't exist, create with full schema
#     c.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='predictions'")
#     if not c.fetchone():
#         # build create statement using REQUIRED_COLUMNS in a deterministic order
#         cols_sql = ",\n    ".join([f"{col} {REQUIRED_COLUMNS[col]}" for col in REQUIRED_COLUMNS])
#         create_sql = f"CREATE TABLE predictions (\n    {cols_sql}\n)"
#         c.execute(create_sql)
#         conn.commit()
#         conn.close()
#         return

#     # Table exists: inspect columns
#     c.execute("PRAGMA table_info(predictions)")
#     existing = [row[1] for row in c.fetchall()]  # row format: (cid, name, type, notnull, dflt_value, pk)
#     # Add any missing columns
#     for col, coldef in REQUIRED_COLUMNS.items():
#         if col not in existing:
#             try:
#                 # For PK we skip altering since it's only for creation; but if id absent, we still add a column id (non-PK)
#                 if col == "id":
#                     # If id missing, add a plain INTEGER column (can't make it PK on alter)
#                     c.execute(f"ALTER TABLE predictions ADD COLUMN {col} INTEGER")
#                 else:
#                     c.execute(f"ALTER TABLE predictions ADD COLUMN {col} {coldef}")
#             except Exception:
#                 # best-effort: ignore failures (e.g., column already added by race condition)
#                 pass

#     conn.commit()
#     conn.close()

# # ensure schema before anything
# ensure_table_and_columns()

# # --------------------------------------------------------------------
# # SIDEBAR PATIENT INFO (includes PSNO)
# # --------------------------------------------------------------------
# st.sidebar.header("Patient Details")
# with st.sidebar.form("patient_form"):
#     psno = st.text_input("Patient PSNO (unique ID)")
#     name = st.text_input("Patient name")
#     age = st.number_input("Age", min_value=0, max_value=130, value=40)
#     gender = st.selectbox("Gender", ["Female", "Male", "Other"])
#     symptoms = st.text_area("Clinical notes (symptoms)")
#     submit_patient = st.form_submit_button("Save patient info")

# if submit_patient:
#     if not psno:
#         st.sidebar.error("PSNO is required.")
#     else:
#         st.sidebar.success("Patient info saved (temporary)")

# # --------------------------------------------------------------------
# # MODEL METADATA (unchanged)
# # --------------------------------------------------------------------
# st.sidebar.header("Model Info")
# meta = get_model_metadata()
# st.sidebar.write(f"Model file: {os.path.basename(meta['model_path'])}")
# st.sidebar.write(f"Input size: {meta['input_size']}")
# st.sidebar.write(f"Labels: {meta['labels']}")
# st.sidebar.write(f"Loaded layers: {len(meta['keras_model_layers'])} layers")

# # --------------------------------------------------------------------
# # MAIN: Upload, classify, grad-cam, LLM, insert
# # --------------------------------------------------------------------
# st.title("🩺 Breast Cancer Ultrasound Classifier — Enhanced")
# st.markdown("Upload ultrasound images for automated analysis, Grad-CAM, explanation, and DB saving.")

# uploaded_files = st.file_uploader("Upload image(s)", accept_multiple_files=True, type=["jpg","jpeg","png"])
# exp_mode = st.radio("Explanation mode", ("Doctor", "Patient (plain language)"))

# if uploaded_files:
#     results = []
#     temp_dir = tempfile.mkdtemp(prefix="bcapp2_")

#     for uploaded in uploaded_files:
#         try:
#             image = Image.open(uploaded).convert("RGB")
#         except Exception:
#             st.error(f"Could not open {uploaded.name}")
#             continue

#         img_path = os.path.join(temp_dir, uploaded.name)
#         image.save(img_path)

#         blurry, lap_var = is_image_blurry(image)
#         if blurry:
#             st.warning(f"{uploaded.name} appears blurry (Laplacian variance={lap_var:.1f}).")

#         pred_idx, probs = teachable_machine_classification(image)
#         class_names = ["Malignant", "Benign"]
#         pred_label = class_names[pred_idx]

#         # GRAD-CAM
#         try:
#             overlay, heatmap = grad_cam_for_pil(image)
#             grad_path = os.path.join(temp_dir, f"gradcam_{uploaded.name}")
#             overlay.save(grad_path)
#         except Exception:
#             grad_path = None

#         # LLM (keep behavior from app.py2)
#         explanation = "LLM disabled (no API key configured)."
#         if DEEPSEEK_API_KEY:
#             headers = {"Authorization": f"Bearer {DEEPSEEK_API_KEY}", "Content-Type": "application/json"}
#             payload = {
#                 "model": DEEPSEEK_MODEL,
#                 "messages": [
#                     {"role": "system", "content": "You are a medical assistant."},
#                     {"role": "user", "content": (
#                         f"A breast ultrasound image was analyzed by a machine learning model.\n"
#                         f"Prediction: {pred_label}\n"
#                         f"Probabilities: Malignant {probs[0]*100:.2f}%, Benign {probs[1]*100:.2f}%.\n"
#                         f"Patient name: {name}\nAge: {age}\nGender: {gender}\nNotes: {symptoms}\n"
#                         f"Mode: {exp_mode}"
#                     )}
#                 ],
#                 "max_tokens": 700,
#                 "temperature": 0.2
#             }
#             try:
#                 r = requests.post(f"{DEEPSEEK_BASE}/chat/completions", headers=headers, json=payload, timeout=30)
#                 if r.status_code == 200:
#                     data = r.json()
#                     explanation = data["choices"][0]["message"]["content"].strip()
#                 else:
#                     explanation = f"DeepSeek error: {r.status_code} {r.text}"
#             except Exception as e:
#                 explanation = f"DeepSeek API call failed: {e}"

#         # Recommendation logic (same)
#         recommendation = ""
#         if probs[0] >= 0.70:
#             recommendation = "High suspicion for malignancy — urgent follow-up required."
#         elif probs[1] >= 0.90:
#             recommendation = "High confidence benign — routine follow-up."
#         elif max(probs) < 0.55:
#             recommendation = "Low confidence — further imaging advised."

#         # Prepare record using original prepare_result_record (keeps compatibility)
#         patient_info = {"name": name, "age": age, "gender": gender, "notes": symptoms}
#         try:
#             rec = prepare_result_record(patient_info, [img_path], pred_label, probs, explanation)
#             # rec has fields: timestamp, patient_name, patient_age, patient_gender, notes, image_paths,
#             #                 prediction, probs, explanation, model_metadata
#             # Add psno into record so insert_record will store it in the DB's psno column (if insert_record supports it)
#             rec["psno"] = psno
#             # Use the original insert_record() from img_classification.py
#             insert_record(rec)
#         except Exception as e:
#             st.error(f"DB insert error: {e}")

#         # collect for display
#         results.append({
#             "filename": uploaded.name,
#             "image_path": img_path,
#             "gradcam_path": grad_path,
#             "prediction": pred_label,
#             "probs": [float(probs[0]), float(probs[1])],
#             "explanation": explanation,
#             "recommendation": recommendation
#         })

#     # Display results
#     st.markdown("### Results")
#     for r in results:
#         st.markdown(f"**{r['filename']}** — Prediction: **{r['prediction']}**")
#         st.write(f"Malignant: {r['probs'][0]*100:.2f}% | Benign: {r['probs'][1]*100:.2f}%")
#         col1, col2 = st.columns(2)
#         with col1:
#             st.image(r["image_path"], caption="Original")
#         with col2:
#             if r["gradcam_path"]:
#                 st.image(r["gradcam_path"], caption="Grad-CAM")
#         st.info(r["recommendation"])
#         with st.expander("Explanation"):
#             st.write(r["explanation"])
#         fig, ax = plt.subplots(figsize=(4,1.2))
#         ax.barh(["Malignant","Benign"], [r['probs'][0]*100, r['probs'][1]*100])
#         ax.set_xlim(0,100)
#         st.pyplot(fig)
#         st.markdown("---")

# # --------------------------------------------------------------------
# # DATABASE MANAGEMENT UI (replaces PDF export)
# # --------------------------------------------------------------------
# st.header("📦 Database Management")

# # Download CSV
# if st.button("Download database as CSV"):
#     ensure_table_and_columns = None  # placeholder to avoid linter confusion (we used ensure_table_and_columns earlier)
#     # ensure schema and read
#     ensure_table_and_columns = None
#     # Re-run the ensure (safe to call)
#     ensure_table_and_columns = lambda: ensure_table_and_columns  # no-op
#     # We'll just make sure the DB exists and then read
#     conn = sqlite3.connect(DB_PATH)
#     try:
#         df = pd.read_sql_query("SELECT * FROM predictions", conn)
#     except Exception as e:
#         st.error(f"Failed to read DB: {e}")
#         conn.close()
#         df = pd.DataFrame()
#     conn.close()

#     if df.empty:
#         st.info("Database is empty.")
#     else:
#         csv = df.to_csv(index=False).encode()
#         b64 = base64.b64encode(csv).decode()
#         st.markdown(f'<a href="data:file/csv;base64,{b64}" download="predictions.csv">Download CSV</a>', unsafe_allow_html=True)

# # Download Excel
# if st.button("Download database as Excel"):
#     conn = sqlite3.connect(DB_PATH)
#     try:
#         df = pd.read_sql_query("SELECT * FROM predictions", conn)
#     except Exception as e:
#         st.error(f"Failed to read DB: {e}")
#         conn.close()
#         df = pd.DataFrame()
#     conn.close()

#     if df.empty:
#         st.info("Database is empty.")
#     else:
#         excel_path = os.path.join(tempfile.gettempdir(), "predictions.xlsx")
#         df.to_excel(excel_path, index=False)
#         with open(excel_path, "rb") as f:
#             excel_bytes = f.read()
#             b64 = base64.b64encode(excel_bytes).decode()
#             st.markdown(f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}" download="predictions.xlsx">Download Excel</a>', unsafe_allow_html=True)

# # Clear DB
# if st.button("Clear entire database"):
#     if os.path.exists(DB_PATH):
#         try:
#             os.remove(DB_PATH)
#         except Exception as e:
#             st.error(f"Could not remove DB file: {e}")
#         # Recreate table with required columns so UI continues to work
#         ensure_table_and_columns()
#         st.success("Database cleared.")
#     else:
#         st.info("Database already empty.")

# # Show DB table
# if st.checkbox("Show database records"):
#     # Ensure schema present and then show up to 100 rows
#     ensure_table_and_columns()
#     conn = sqlite3.connect(DB_PATH)
#     try:
#         df = pd.read_sql_query("SELECT * FROM predictions ORDER BY id DESC LIMIT 100", conn)
#     except Exception as e:
#         st.error(f"Failed to read DB: {e}")
#         df = pd.DataFrame()
#     conn.close()
#     if df.empty:
#         st.info("No records found.")
#     else:
#         st.dataframe(df)
