# Diabetic Foot Complication Analysis Project

## Project Goal
This project aims to develop a comprehensive system for the early detection and analysis of diabetic foot complications using thermographic imaging. By leveraging statistical analysis, signal processing, and machine learning, the project seeks to identify thermal anomalies indicative of risks such as ischemia and inflammation across different foot angiosomes.

The ultimate goal is to provide a non-invasive, automated diagnostic tool that can assist clinicians in early intervention, potentially preventing severe complications like ulceration and amputation.

## Project Overview
This project focuses on the early detection of diabetic foot complications. Currently, the **Exploratory Data Analysis (EDA)** phase has been completed, highlighting significant thermal differences between Control and Diabetes Mellitus (DM) groups. The subsequent phases for automated segmentation, signal processing, and a diagnostic dashboard are planned as part of the future roadmap.

### ✅ Completed: Exploratory Data Analysis (EDA)
Statistical comparison of **Control vs. Diabetes Mellitus (DM)** groups to understand baseline thermal differences.
*   **Focus**: Angiosome temperature distributions and Temperature Change Index (TCI) values.
*   **Key Findings & Visualizations**:
    *   **Boxplots**: Comparing temperature ranges between groups.
    *   **Symmetry Analysis**: Assessing bilateral thermal differences (Left vs. Right foot).
    *   **Correlation Heatmaps**: Understanding relationships between different angiosomes.

![Control Boxplot](control_boxplot.png)
![DM Boxplot](DM_boxplot.png)
*Figure 1: Comparison of mean temperatures across different foot regions for Control and DM groups.*

---

## 🚀 Project Roadmap & Future Work
The following modules are planned to build a complete end-to-end diagnostic system:

### 1. Preprocessing & Segmentation Report
*   **Goal**: Visual validation of specific "Hybrid Segmentation" techniques (Histogram + Morphology).
*   **Objective**: Demonstrate success on "Cold/Ischemic" feet where thermal contrast is low.

### 2. End-to-End Signal Processing Pipeline
*   **Goal**: Develop a fully integrated code block for automated processing.
*   **Stages**: Segmentation > Registration > Symmetry Logic ($|T_{left} - T_{right}|$).

### 3. Sensitivity Analysis & Threshold Tuning
*   **Goal**: Optimize diagnostic thresholds using ROC Curves.
*   **Plan**: Evaluate performance trade-offs between specific thresholds (e.g., 1.8° C vs. 2.6° C).

### 4. Final Evaluation Report
*   **Goal**: Comprehensive performance reporting.
*   **Metrics**: Sensitivity, Specificity, F1-Score (Diagnostic) and Dice Coefficient (Segmentation).

### 5. Streamlit Diagnostic Dashboard
*   **Goal**: Deploy a user-friendly web interface for clinicians.
*   **Features**:
    *   Drag-and-drop thermogram upload.
    *   "Traffic Light" diagnosis (Normal/Local Risk/Diffuse Risk).
    *   Interactive Angiosome visualization.

## Architecture
*   **Language**: Python
*   **Notebook**: `Jubril's Notebook.ipynb` (Main analysis code)
*   **Data Source**: `ThermoDataBase`

## Getting Started
1.  Ensure you have the required dependencies installed.
2.  Open `Jubril's Notebook.ipynb` to review the current EDA analysis.

---
*Created as part of the Thermograph Images CV2 ML Project.*
