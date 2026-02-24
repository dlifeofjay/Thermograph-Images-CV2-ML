# Diabetic Foot Complication Analysis Project

## 👣 Project Goal
This project aims to develop a comprehensive system for the early detection and analysis of diabetic foot complications using thermographic imaging. By leveraging statistical analysis, signal processing, and machine learning, the project seeks to identify thermal anomalies indicative of risks such as ischemia and inflammation across different foot angiosomes.

## 🚀 Project Overview
This project provides an automated end-to-end diagnostic tool for diabetic foot complications. It includes a complete processing pipeline from raw thermograms to a diagnostic dashboard.

### ✅ Completed: Exploratory Data Analysis (EDA)
Statistical comparison of **Control vs. Diabetes Mellitus (DM)** groups to understand baseline thermal differences.
*   **Focus**: Angiosome temperature distributions and Temperature Change Index (TCI) values.

### ✅ Completed: Hybrid Segmentation & Pipeline
*   **Goal**: Automated segmentation of foot regions (angiosomes).
*   **Technique**: "Hybrid Segmentation" (Histogram + Morphology) optimized for "Cold/Ischemic" feet.
*   **Pipeline**: Segmentation > Registration > Symmetry Logic ($|T_{left} - T_{right}|$).

### ✅ Completed: Streamlit Diagnostic Dashboard
A user-friendly web interface for clinicians to analyze foot thermograms.
*   **Features**:
    *   Drag-and-drop thermogram and CSV temperature data upload.
    *   "Traffic Light" diagnosis (Normal/Local Risk/Diffuse Risk).
    *   Interactive Angiosome visualization.

## 🛠 Architecture
*   **Language**: Python 3.x
*   **Libraries**: Streamlit, OpenCV, NumPy, Pandas, Matplotlib
*   **Main Dashboard**: `streamlit_app.py`
*   **Processing Core**: `full_pipeline.py`, `segmentation.py`
*   **Data Source**: `ThermoDataBase`

## 🚀 Getting Started

### Prerequisites
```bash
pip install streamlit opencv-python numpy pandas matplotlib
```

### Running the Dashboard
1.  Navigate to the project directory.
2.  Run the Streamlit app:
    ```bash
    streamlit run streamlit_app.py
    ```
3.  Upload the Left/Right images and their corresponding temperature CSV files.

---

## 📈 Future Work
*   **Sensitivity Analysis (Ongoing)**: Refinement of diagnostic thresholds using ROC Curves.
*   **Improved Registration**: Enhancing the alignment between Left and Right foot thermograms.

---
*Created as part of the Thermograph Images CV2 ML Project.*
