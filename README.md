# Diabetic Foot Complication Analysis Project

## 👣 Project Goal
This project develops a comprehensive system for the early detection and analysis of diabetic foot complications using thermographic imaging. By leveraging statistical analysis, signal processing, and machine learning, it identifies thermal anomalies indicative of risks such as ischemia and inflammation across different foot angiosomes.

---

## 🚀 Project Overview
An automated end-to-end diagnostic tool for diabetic foot complications — from raw thermograms to an interactive diagnostic dashboard. All six core deliverables are completed and documented below.

---

## ✅ Deliverables

### 1. Exploratory Data Analysis (EDA)
Statistical comparison of **Control vs. Diabetes Mellitus (DM)** groups to understand baseline thermal differences.
*   **Focus**: Angiosome temperature distributions and Temperature Change Index (TCI) values.
*   **Where to find it**:
    *   📄 [`EXPLORATORY DATA ANALYSIS(Jubril).pdf`](./EXPLORATORY%20DATA%20ANALYSIS(Jubril).pdf) — Full EDA report with visualizations.
    *   📓 [`Jubril's Notebook & Other Deliverables.ipynb`](./Jubril's%20Notebook%20%26%20Other%20Deliverables.ipynb) — Executable code for EDA (data loading, statistical summaries, distribution plots).

---

### 2. Preprocessing & Segmentation Report
Automated segmentation of foot regions (angiosomes) from raw thermograms.
*   **Technique**: "Hybrid Segmentation" (Histogram Thresholding + Morphological Operations), optimized for "Cold/Ischemic" feet.
*   **Where to find it**:
    *   📄 [`Preprocessing & Segmentation.pdf`](./Preprocessing%20%26%20Segmentation.pdf) — Visual validation report demonstrating segmentation success.
    *   🐍 [`segmentation.py`](./segmentation.py) — Core segmentation logic (`image_proc` function: standardization → thresholding → morphology → noise removal).
    *   📓 [`Jubril's Notebook & Other Deliverables.ipynb`](./Jubril's%20Notebook%20%26%20Other%20Deliverables.ipynb) — Step-by-step preprocessing walkthrough.

---

### 3. End-to-End Signal Processing Pipeline
A fully integrated pipeline: **Segmentation → Registration → Symmetry Logic**.
*   Segments both feet into 4 angiosomes (MPA, LPA, LCA, MCA).
*   Flips the right foot for anatomical registration.
*   Computes per-angiosome mean temperatures and **ΔT** (absolute left-right difference).
*   **Where to find it**:
    *   🐍 [`full_pipeline.py`](./full_pipeline.py) — The complete pipeline (`pipeline()` and `extract_and_save_angiosomes()` functions).
    *   📓 [`Jubril's Notebook & Other Deliverables.ipynb`](./Jubril's%20Notebook%20%26%20Other%20Deliverables.ipynb) — `run_full_analysis()` function with comparison grid visualization and the **Final Symmetry Report** (ΔT classification with a 2.2°C risk threshold).

---

### 4. Sensitivity Analysis & Threshold Tuning (ROC Curve)
Performance trade-off analysis between diagnostic thresholds of **1.8°C** and **2.6°C**.
*   Computes the ROC curve using `max_asymmetry` as the scoring metric against ground-truth labels.
*   Marks both threshold operating points on the curve with their corresponding Sensitivity values.
*   **Where to find it**:
    *   📓 [`Jubril's Notebook & Other Deliverables.ipynb`](./Jubril's%20Notebook%20%26%20Other%20Deliverables.ipynb) — ROC Curve generation cell (uses `sklearn.metrics.roc_curve` and `auc`). The plot is saved as `roc.png`.

---

### 5. Final Evaluation Report (Diagnostic & Segmentation Metrics)
Classification performance metrics at both threshold levels.
*   **Threshold 1.8°C**: Classification Report (Precision, Recall, F1-Score) + Confusion Matrix (`conf 1.8.png`).
*   **Threshold 2.6°C**: Classification Report (Precision, Recall, F1-Score) + Confusion Matrix (`con 2.6.png`).
*   **Where to find it**:
    *   📄 [`EVALUATION REPORT.pdf`](./EVALUATION%20REPORT.pdf) — Compiled evaluation report.
    *   📓 [`Jubril's Notebook & Other Deliverables.ipynb`](./Jubril's%20Notebook%20%26%20Other%20Deliverables.ipynb) — Executable cells under **"Metrics Measurements"** section (using `sklearn.metrics`: `classification_report`, `confusion_matrix`, `ConfusionMatrixDisplay`).

---

### 6. Streamlit Diagnostic Dashboard
A user-friendly web interface for clinicians to analyze foot thermograms in real time.
*   **Features**:
    *   Drag-and-drop upload for thermogram images and CSV temperature data (Left & Right foot).
    *   **Global Diagnosis Banner**: Classifies patient status as `NORMAL`, `MONITORING REQUIRED`, `LOCAL RISK`, or `DIFFUSE RISK`.
    *   **Angiosome Segmentation Visualization**: 2×5 grid showing the foot overview with segmentation lines and the 4 individual angiosome cuts for both feet.
    *   **Traffic Light Symmetry Metrics**: Per-angiosome ΔT cards with color-coded risk levels:
        | ΔT Range       | Color   | Label       |
        |-----------------|---------|-------------|
        | ≤ 1.8°C         | 🟢 Green | NORMAL      |
        | 1.8°C – 2.6°C   | 🟡 Orange | WARNING     |
        | > 2.6°C         | 🔴 Red   | HIGH RISK   |
*   **Where to find it**:
    *   🐍 [`streamlit_app.py`](./streamlit_app.py) — Full dashboard source code.

---

## 🛠 Architecture
| Component          | File(s)                                    |
|--------------------|--------------------------------------------|
| **Language**       | Python 3.x                                 |
| **Libraries**      | Streamlit, OpenCV, NumPy, Pandas, Matplotlib, scikit-learn |
| **Dashboard**      | `streamlit_app.py`                         |
| **Pipeline Core**  | `full_pipeline.py`, `segmentation.py`      |
| **Visualization**  | `foot_plot.py`                             |
| **Data Source**     | `ThermoDataBase/`                          |
| **Reports**        | `EXPLORATORY DATA ANALYSIS(Jubril).pdf`, `Preprocessing & Segmentation.pdf`, `EVALUATION REPORT.pdf` |
| **Notebook**       | `Jubril's Notebook & Other Deliverables.ipynb` |

---

## 🚀 Getting Started

### Prerequisites
```bash
pip install streamlit opencv-python numpy pandas matplotlib scikit-learn
```

### Running the Dashboard
1.  Navigate to the project directory.
2.  Run the Streamlit app:
    ```bash
    streamlit run streamlit_app.py
    ```
3.  Upload the Left/Right foot images and their corresponding temperature CSV files.

---

*Created as part of the Thermograph Images CV2 ML Project.*
