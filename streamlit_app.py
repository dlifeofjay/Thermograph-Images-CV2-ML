import streamlit as st
import cv2 as cv
import numpy as np
import pandas as pd
import tempfile
import os
import matplotlib.pyplot as plt
from full_pipeline import pipeline

# --- Page Config ---
st.set_page_config(
    page_title="DFU Thermography Dashboard", 
    page_icon="👣",
    layout="wide"
)

# --- Styling ---
st.markdown("""
<style>
    .reportview-container {
        background: #0e1117;
    }
    .main-header {
        font-size: 2.5rem;
        color: #4da6ff;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #262730;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #4da6ff;
        text-align: center;
    }
    .diagnosis-banner {
        padding: 1.5rem;
        border-radius: 0.5rem;
        text-align: center;
        margin-bottom: 2rem;
        font-size: 1.5rem;
        font-weight: bold;
    }
    h3 {
        color: #fafafa;
    }
</style>
""", unsafe_allow_html=True)

# --- Helper Functions ---
def save_uploaded_file(uploaded_file):
    """Save uploaded streamlit file to temp and return path."""
    if uploaded_file is None:
        return None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            return tmp_file.name
    except Exception as e:
        st.error(f"Error saving file: {e}")
        return None

# --- Main App ---
def main():
    st.markdown('<div class="main-header">Diabetic Foot Thermography Analysis</div>', unsafe_allow_html=True)

    # --- Sidebar: Inputs ---
    with st.sidebar:
        st.header("1. Upload Data")
        st.info("Both Image and CSV are REQUIRED.")
        
        st.subheader("Left Foot")
        left_img_file = st.file_uploader("Left Image (PNG/JPG)", type=["png", "jpg", "jpeg"], key="left_img")
        left_csv_file = st.file_uploader("Left Temperature CSV", type=["csv"], key="left_csv")
        
        st.subheader("Right Foot")
        right_img_file = st.file_uploader("Right Image (PNG/JPG)", type=["png", "jpg", "jpeg"], key="right_img")
        right_csv_file = st.file_uploader("Right Temperature CSV", type=["csv"], key="right_csv")

        run_btn = st.button("Run Analysis", type="primary")

    # --- Main Analysis ---
    if run_btn:
        if not all([left_img_file, left_csv_file, right_img_file, right_csv_file]):
            st.error("Please upload ALL 4 files (Left/Right Images AND CSVs) to proceed.")
            return

        with st.spinner("Processing thermograms..."):
            # 1. Save temp files
            l_img_p = save_uploaded_file(left_img_file)
            l_csv_p = save_uploaded_file(left_csv_file)
            r_img_p = save_uploaded_file(right_img_file)
            r_csv_p = save_uploaded_file(right_csv_file)
            
            try:
                # 2. Run Pipeline
                left_res, right_res = pipeline(l_img_p, l_csv_p, r_img_p, r_csv_p)
                (l_data, l_cuts, l_bounds, l_means) = left_res
                (r_data, r_cuts, r_bounds, r_means) = right_res

                # Cleanup
                for p in [l_img_p, l_csv_p, r_img_p, r_csv_p]:
                    if os.path.exists(p): os.remove(p)

                # --- 1. GLOBAL DIAGNOSIS ---
                angiosomes = ["MPA", "LPA", "MCA", "LCA"]
                high_risk_count = 0
                warning_count = 0
                diffs = {}

                for angio in angiosomes:
                    diff = abs(l_means.get(angio, 0) - r_means.get(angio, 0))
                    diffs[angio] = diff
                    if diff > 2.6:
                        high_risk_count += 1
                    elif diff > 1.8:
                        warning_count += 1
                
                # Diagnosis Logic
                if high_risk_count >= 2:
                    diag_status = "DIFFUSE RISK"
                    diag_color = "#ff3333" # Red
                    diag_msg = f"Multiple High Risk regions ({high_risk_count}) detected."
                elif high_risk_count == 1:
                    diag_status = "LOCAL RISK"
                    diag_color = "#ff3333" # Red
                    diag_msg = "Single High Risk region detected."
                elif warning_count > 0:
                    diag_status = "MONITORING REQUIRED"
                    diag_color = "#ffcc00" # Orange
                    diag_msg = f"{warning_count} regions in Warning zone."
                else:
                    diag_status = "NORMAL"
                    diag_color = "#00cc66" # Green
                    diag_msg = "No significant temperature asymmetry."

                st.markdown(f"""
                <div class="diagnosis-banner" style="background-color: {diag_color}22; border: 2px solid {diag_color}; color: {diag_color};">
                    <div>DIAGNOSTIC STATUS: {diag_status}</div>
                    <div style="font-size: 1rem; font-weight: normal; margin-top: 5px;">{diag_msg}</div>
                </div>
                """, unsafe_allow_html=True)


                # --- 2. VISUALIZATION ---
                st.header("Angiosome Segmentation")
                
                # Prepare plot 
                fig, axes = plt.subplots(2, 5, figsize=(22, 10))
                fig.patch.set_facecolor('#0e1117')
                
                foot_labels = ["Left Foot", "Right Foot (Flipped)"]
                all_res = [left_res, right_res]

                for row in range(2):
                    data, cuts, bounds, _ = all_res[row]
                    v_split, h_split_medial, toe_limit, h_split_heel = bounds
                    
                    # Col 0: Grid
                    ax = axes[row, 0]
                    ax.imshow(data, cmap='inferno')
                    ax.vlines(h_split_medial, toe_limit, v_split, color='white', lw=1)
                    ax.vlines(h_split_heel, v_split, data.shape[0], color='white', lw=1)
                    ax.hlines(v_split, 0, data.shape[1], color='white', lw=1)
                    ax.hlines(toe_limit, 0, data.shape[1], color='yellow', linestyles='dashed', lw=1)
                    ax.set_title(f"{foot_labels[row]} Grid", color='white')
                    ax.axis('off')
                    
                    # Col 1-4: Cuts
                    for i, (name, cut_arr) in enumerate(cuts.items(), 1):
                        ax_cut = axes[row, i]
                        viz = np.where(cut_arr > 0, cut_arr, np.nan)
                        ax_cut.imshow(viz, cmap='inferno')
                        ax_cut.set_title(name, color='white')
                        ax_cut.axis('off')

                st.pyplot(fig)

                # --- 3. METRICS ---
                st.header("Symmetry Metrics (Traffic Light System)")
                cols = st.columns(4)
                
                for i, angio in enumerate(angiosomes):
                    diff = diffs[angio]
                    l_val = l_means.get(angio, 0)
                    r_val = r_means.get(angio, 0)
                    
                    # Traffic Light Thresholds
                    if diff > 2.6:
                        risk_color = "#ff3333" # Red
                        risk_label = "HIGH RISK"
                    elif diff > 1.8:
                        risk_color = "#ffcc00" # Orange
                        risk_label = "WARNING"
                    else:
                        risk_color = "#00cc66" # Green
                        risk_label = "NORMAL"
                    
                    with cols[i]:
                        st.markdown(f"""
                        <div class="metric-card" style="border-color: {risk_color};">
                            <h3>{angio}</h3>
                            <div style="font-size: 2rem; font-weight: bold;">{diff:.2f}°C</div>
                            <div style="color: {risk_color}; font-weight: bold;">{risk_label}</div>
                            <div style="font-size: 0.8rem; color: #888;">L: {l_val:.1f} | R: {r_val:.1f}</div>
                        </div>
                        """, unsafe_allow_html=True)

            except Exception as e:
                st.error(f"An error occurred: {e}")
                st.exception(e)

if __name__ == "__main__":
    main()
