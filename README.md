# breast-cancer-detection

Original dataset
[https://www.kaggle.com/aryashah2k/breast-ultrasound-images-dataset](https://www.kaggle.com/datasets/aryashah2k/breast-ultrasound-images-dataset)

Test dataset
[https://drive.google.com/file/d/1xjp_6O8C-Z5sAAFn3UeR03tlSYUlRkdc/view](https://data.mendeley.com/datasets/7fvgj4jsp7/1)

# Breast Cancer Ultrasound Classifier – Project Documentation

<img width="1597" height="726" alt="img1" src="https://github.com/user-attachments/assets/93fb290a-d56c-4e81-b6ab-a629d3617ba9" />

## 1. Project Overview
The Breast Cancer Ultrasound Classifier is a web-based application designed to assist medical professionals in the analysis of breast ultrasound images. Leveraging deep learning, Grad-CAM visualization, and large language model (LLM) explanations, this system allows for:
•  Automated classification of breast ultrasound images into Malignant or Benign categories.  
•  Visualization of model attention using Grad-CAM heatmaps.  
•  Explanation of predictions in both medical terminology (for doctors) and plain language (for patients).  
•  Storage and management of patient records in a local SQLite database with CSV export capabilities.  

The project aims to enhance diagnostic efficiency, improve interpretability of ML predictions, and serve as a decision-support tool in clinical workflows.

The system integrates deep learning, Grad-CAM visualization, and natural language explanations to assist healthcare professionals and patients in understanding the model predictions. Additionally, it maintains a database for patient records, predictions, and model metadata, allowing structured follow-up and data management.

This project leverages Streamlit for the user interface, TensorFlow/Keras for deep learning, and SQLite for local database storage. Optional integration with DeepSeek LLM API provides contextual textual explanations of predictions.

________________________________________

### Objectives
1. Automated Breast Cancer Detection: Enable accurate classification of breast lesions from ultrasound images.  
2. Explainable AI: Provide Grad-CAM heatmaps to highlight critical regions influencing the model's decision.  
3. Patient Communication: Generate plain-language explanations for patients and detailed clinical notes for doctors.  
4. Record-Keeping: Store predictions, images, and explanations in a database for future reference.  
5. User-Friendly Interface: Interactive web application with image upload, real-time analysis, visualization, and CSV export.  

________________________________________

## 2. System Architecture

### Components:

1. **Frontend/UI (Streamlit)**  
   o Patient detail form with PSNO, Name, Age, Gender, and clinical notes.
     <img width="1597" height="726" alt="img1" src="https://github.com/user-attachments/assets/8395516b-9ad9-46ca-8458-2c6e43449b2b" />

   o Image upload (single-image mode) with optional blurriness warnings.  
   o Display of predictions, probabilities, Grad-CAM overlays, and bar charts.
    <img width="1594" height="722" alt="img pred" src="https://github.com/user-attachments/assets/c4baa245-377a-4367-b3c8-1db81b802c76" />

   o PDF report generation for professional clinical use.  
   o Database management with CSV export and table view.
     <img width="1564" height="726" alt="Dbs" src="https://github.com/user-attachments/assets/1bd7e6f2-8ef9-46ea-898e-0db222e0737f" />


3. **Backend (Image Classification & Grad-CAM)**  
   o Preprocessing and normalization of input images.  
   o Classification using Keras deep learning model (keras_model.h5).  
   o Grad-CAM overlay generation for explainable AI.  
   o Confidence-level warnings based on probabilities.  

4. **Database Layer (SQLite)**  
   o Stores patient info, predictions, probabilities, Grad-CAM paths, explanations, and model metadata.  
   o Supports export to CSV for record-keeping and tracking.  

5. **LLM Layer (DeepSeek API)**  
   o Generates textual explanations of model predictions.  
   o Supports two explanation modes:  
     ▪ Doctor – medical terminology.

      <img width="1597" height="733" alt="doc expl" src="https://github.com/user-attachments/assets/06aca841-9571-43de-bc38-bbc0562d7584" />

  

     ▪ Patient – plain language “Explain Like I’m 5”.

      <img width="1595" height="727" alt="patient expl" src="https://github.com/user-attachments/assets/1b4a5bb5-44a4-4e4e-9a14-2c3d647fb478" />



### Data Flow:
Patient Info → Image Upload → Preprocessing → Model Prediction → Grad-CAM → LLM Explanation → DB Storage → Display & PDF Export

________________________________________

## 3. Detailed Component Description

### 3.1 Frontend (Streamlit UI)

• **Patient Details Form**  
  o Fields: PSNO, Name, Age, Gender, Symptoms/History.  
  o State managed using st.session_state.  
  o Information used in DeepSeek prompt and PDF reports.  

• **Image Upload & Classification**  
  o Accepts JPEG/PNG images.  
  o Blurriness detection using Laplacian variance, warns if image is low quality.  
  o Sends preprocessed images to the Keras model for prediction.  
  o Generates Grad-CAM heatmaps to visualize regions influencing the prediction.  

• **Prediction Display**  
  o Shows predicted label (Malignant/Benign) and probability distribution.  
  o Bar chart visualization of probabilities for immediate clarity.  
  o Confidence warnings:  
    ▪ Malignant ≥ 70% → High suspicion; recommend biopsy.  
    ▪ Benign ≥ 90% → Reassuring; routine follow-up.  
    ▪ Maximum probability < 55% → Low confidence; further imaging advised.  

• **PDF Report Generation**  
  o Includes prediction, probabilities, Grad-CAM image, LLM explanation, doctor notes, timestamp, and optionally hospital logo.  
  o Supports professional clinical documentation.  

• **Database Management**  
  o SQLite-based storage of predictions, explanations, images, and metadata.  
  o Supports export to CSV and clearing of database.  

________________________________________

### 3.2 Backend (Image Classification & Grad-CAM)

#### 3.2.1 Model Details
• Model: keras_model.h5 (Keras/TensorFlow)  
• Input Size: 224×224×3 RGB images  
• Labels: Malignant, Benign (selected from labels.txt)  

#### Prediction Workflow:
1. Preprocess image (resize, crop, normalize to [-1,1]).  
2. Run through the model.  
3. Extract only Malignant/Benign probabilities.  
4. Normalize probabilities.  
5. Return predicted label and probability distribution.  

#### 3.2.2 Grad-CAM (Explainable AI)
• Highlights image regions influencing the prediction.  
• Outputs:  
  o Overlay image (original + heatmap)  
  o Grayscale heatmap  
• Color interpretation:  
  o Red = suspicious/malignant regions  
  o Blue = less significant/benign regions  
• Supports transparency and interpretability in clinical decisions.  

#### 3.2.3 Image Quality Checker
• Measures blurriness using Laplacian variance.  
• Warns users if input images may yield unreliable predictions.  

#### 3.2.4 Model Metadata & Versioning
• Metadata displayed includes:  
  o Model path  
  o Input size  
  o Labels  
  o Number of layers  
  o Load timestamp  
• Enhances trust and reproducibility.  

________________________________________

### 3.3 LLM Explanation (DeepSeek API)
• Generates structured explanations for predictions.  
• Two modes:  
  o Doctor: Uses professional medical terminology.  
  o Patient: Plain language explanation.  
• Explanations are incorporated into PDF reports and database records.  
• Default message used if API key is not configured.  

________________________________________

### 3.4 Database Management (SQLite)
• Stores:  
  o Timestamp, patient info, image paths, prediction, probabilities, Grad-CAM paths, explanations, model metadata.  
• Features:  
  o View database records in a table.  
  o Export database to CSV.  
  o Clear database entirely.  
• Enables longitudinal tracking and auditability.  

________________________________________

### 3.5 Visualization & Reporting
• Bar Chart: Horizontal bar chart showing Malignant vs. Benign probabilities.

________________________________________

## 4. Software Dependencies
• Python 3.9+  
• Packages:  
  o streamlit – Web UI  
  o tensorflow / keras – Deep learning  
  o opencv-python – Image processing  
  o Pillow – Image handling  
  o numpy – Numerical computations  
  o matplotlib – Bar charts  
  o requests – DeepSeek API  
  o sqlite3 – Database  
  o pandas – CSV handling  
  o python-dotenv – Environment variables  
  o base64 – Background image encoding  

________________________________________

## 5. Project Highlights
1. End-to-End AI Pipeline: Image upload → preprocessing → prediction → explanation → storage.  
2. Explainable AI: Grad-CAM highlights critical regions in ultrasound images.  
3. Clinically Responsible: Confidence-based warnings ensure safe recommendations.  
4. User-Centric Design: Supports both patient-friendly and doctor-focused outputs.  
5. Professional Reporting: PDF generation with full patient, prediction, and Grad-CAM details.  
6. Data Management: Database storage and CSV export for historical tracking.  

### Future Improvements:
• Integration with hospital PACS systems.  
• Cloud deployment for multi-user access.  
• Advanced model fine-tuning on larger datasets.  
• More interactive LLM explanations and templated reports.  

________________________________________

## 7. Conclusion
The Breast Cancer Ultrasound Classifier is a research-driven, clinically oriented AI system that:  
• Combines deep learning classification with explainable AI (Grad-CAM).  
• Provides interpretable outputs for doctors and patients.  
• Ensures clinically safe recommendations via probability-based thresholds.  
• Stores patient records and model outputs in a structured database.  
• Generates professional PDF reports for hospital documentation.  

This project demonstrates how AI can enhance breast cancer diagnostics, improve workflow efficiency, and maintain transparency in clinical settings.

________________________________________

## 8. Directory Structure
breast-cancer-detection/
│
├─ app.py # Main Streamlit app
├─ img_classification.py # Model, Grad-CAM, preprocessing, DB
├─ model/
│ ├─ keras_model.h5
│ └─ labels.txt
├─ images/
│ └─ bg1.jpg # Background image
├─ fonts/
│ └─ DejaVuSans.ttf
├─ predictions.db # SQLite database (auto-generated)
└─ .env # Environment variables (API keys)


________________________________________

## 9. References & Tools
• TensorFlow / Keras  
• Streamlit  
• OpenCV (image processing)  
• Grad-CAM: Selvaraju et al., 2017  
• Teachable Machine workflow  
• DeepSeek API  
• SQLite3 / Pandas for data management  

