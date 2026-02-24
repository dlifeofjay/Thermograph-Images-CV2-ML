import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from full_pipeline import pipeline

# --- CONFIG ---
DB_PATH = "ThermoDataBase" # Adjust if your path is different
CONTROL_DIR = os.path.join(DB_PATH, "Control_Group")
DM_DIR = os.path.join(DB_PATH, "DM_Group")
OUTPUT_FILE = "Sensitivity_Report.csv"
PLOT_FILE = "ROC_Curve.png"

def get_patient_data(group_dir, label):
    """
    Scans a group directory for patient subfolders.
    """
    patients = []
    if not os.path.exists(group_dir):
        print(f"Directory not found: {group_dir}")
        return []

    for pat_id in os.listdir(group_dir):
        pat_path = os.path.join(group_dir, pat_id)
        if not os.path.isdir(pat_path): continue
        
        files = os.listdir(pat_path)
        l_img = next((f for f in files if "_L.png" in f or "_L.jpg" in f), None)
        r_img = next((f for f in files if "_R.png" in f or "_R.jpg" in f), None)
        l_csv = next((f for f in files if "_L.csv" in f), None)
        r_csv = next((f for f in files if "_R.csv" in f), None)
        
        if all([l_img, r_img, l_csv, r_csv]):
            patients.append({
                "id": pat_id,
                "label": label, # 0 for Control, 1 for DM
                "l_img_p": os.path.join(pat_path, l_img),
                "r_img_p": os.path.join(pat_path, r_img),
                "l_csv_p": os.path.join(pat_path, l_csv),
                "r_csv_p": os.path.join(pat_path, r_csv)
            })
    return patients

def run_batch_analysis():
    print("Scanning Database...")
    controls = get_patient_data(CONTROL_DIR, 0)
    dms = get_patient_data(DM_DIR, 1)
    
    all_patients = controls + dms
    print(f"Found {len(controls)} Controls and {len(dms)} DM patients. Total: {len(all_patients)}")
    
    results = []
    
    print("Running Pipeline on all patients (this may take time)...")
    for i, p in enumerate(all_patients):
        if i % 10 == 0: print(f"Processing {i}/{len(all_patients)}...")
        
        try:
            left_res, right_res = pipeline(p['l_img_p'], p['l_csv_p'], p['r_img_p'], p['r_csv_p'])
            _, _, _, l_means = left_res
            _, _, _, r_means = right_res
            
            # Calculate Max Diff across all angiosomes
            max_diff = 0.0
            for angio in ["MPA", "LPA", "MCA", "LCA"]:
                diff = abs(l_means.get(angio, 0) - r_means.get(angio, 0))
                if diff > max_diff: max_diff = diff
            
            results.append({
                "Patient_ID": p['id'],
                "Group": "Control" if p['label'] == 0 else "DM",
                "True_Label": p['label'],
                "Max_Temp_Diff": max_diff
            })
            
        except Exception as e:
            print(f"Error processing {p['id']}: {e}")
            continue

    df = pd.DataFrame(results)
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"Saved results to {OUTPUT_FILE}")
    return df

def calculate_roc(y_true, y_scores):
    """
    Manual ROC calculation to avoid sklearn dependency.
    """
    # Sort by score descending
    desc_score_indices = np.argsort(y_scores)[::-1]
    y_scores = y_scores[desc_score_indices]
    y_true = y_true[desc_score_indices]
    
    distinct_value_indices = np.where(np.diff(y_scores))[0]
    threshold_idxs = np.r_[distinct_value_indices, y_true.size - 1]
    
    tps = np.cumsum(y_true)[threshold_idxs]
    fps = (1 + threshold_idxs) - tps
    
    tps = np.r_[0, tps]
    fps = np.r_[0, fps]
    
    if fps[-1] <= 0:
        fpr = np.zeros_like(fps)
    else:
        fpr = fps / fps[-1]
        
    if tps[-1] <= 0:
        tpr = np.zeros_like(tps)
    else:
        tpr = tps / tps[-1]
        
    return fpr, tpr

def analyze_sensitivity(df):
    labels = df["True_Label"].values
    scores = df["Max_Temp_Diff"].values
    
    # 1. ROC Curve
    fpr, tpr = calculate_roc(labels, scores)
    
    # AUC (Trapezoidal rule manual)
    # Ensure sorted by FPR
    sort_idx = np.argsort(fpr)
    fpr_sorted = fpr[sort_idx]
    tpr_sorted = tpr[sort_idx]
    roc_auc = np.sum((fpr_sorted[1:] - fpr_sorted[:-1]) * (tpr_sorted[1:] + tpr_sorted[:-1])) / 2
    
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate (Sensitivity)')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")
    plt.savefig(PLOT_FILE)
    print(f"Saved ROC Plot to {PLOT_FILE}")
    
    # 2. Specific Threshold Metrics
    print("\n--- Threshold Analysis ---")
    for thresh in [1.8, 2.2, 2.6]:
        # Predictions at this threshold
        pred = (scores > thresh).astype(int)
        
        TP = ((pred == 1) & (labels == 1)).sum()
        TN = ((pred == 0) & (labels == 0)).sum()
        FP = ((pred == 1) & (labels == 0)).sum()
        FN = ((pred == 0) & (labels == 1)).sum()
        
        sens = TP / (TP + FN) if (TP+FN) > 0 else 0
        spec = TN / (TN + FP) if (TN+FP) > 0 else 0
        f1 = 2*TP / (2*TP + FP + FN) if (2*TP + FP + FN) > 0 else 0
        
        print(f"Threshold > {thresh}°C:")
        print(f"  Sensitivity: {sens:.4f}")
        print(f"  Specificity: {spec:.4f}")
        print(f"  F1-Score:    {f1:.4f}")

if __name__ == "__main__":
    # If using existing results, ensure columns match
    if os.path.exists(OUTPUT_FILE):
        print("Loading existing results...")
        df = pd.read_csv(OUTPUT_FILE)
    else:
        df = run_batch_analysis()
    
    if not df.empty:
        analyze_sensitivity(df)
    else:
        print("No data found to analyze.")
