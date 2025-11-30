import streamlit as st
from PIL import Image
from img_classification import teachable_machine_classification
import numpy as np
import base64

# ---------------------------
# DARKENED BACKGROUND IMAGE
# ---------------------------
def add_bg_image(image_file):
    with open(image_file, "rb") as f:
        data = f.read()
    encoded = base64.b64encode(data).decode()

    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image:
                linear-gradient(rgba(0,0,0,0.65), rgba(0,0,0,0.65)),
                url("data:image/jpg;base64,{encoded}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
        }}

        /* Optional: Make text brighter for visibility */
        h1, h2, h3, h4, h5, h6, p, span, label {{
            color: white !important;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# Path to your background image
add_bg_image("images/bg1.jpg")
# ---------------------------


st.title("Breast Cancer Ultrasound Classifier")
st.header("Upload an image for prediction")
st.text("Model trained using Google's Teachable Machine")

uploaded_file = st.file_uploader("Choose an ultrasound scan...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Reduce image size for display
    image = Image.open(uploaded_file)
    image = image.resize((350, 350))  # Medium size
    
    st.image(image, caption='Uploaded Scan', use_container_width=False)

    st.write("Classifying...")

    # Get predictions
    label, probabilities = teachable_machine_classification(
        Image.open(uploaded_file), 
        'model/keras_model.h5'
    )

    class_names = ["Normal", "Malignant", "Benign"]

    st.subheader(f"Prediction: **{class_names[label]}**")

    st.write("### Prediction Probabilities:")
    for i, class_name in enumerate(class_names):
        st.write(f"**{class_name}: {probabilities[i] * 100:.2f}%**")
