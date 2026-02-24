from segmentation import image_proc
import numpy as np
import cv2 as cv
import pandas as pd
import os

def extract_and_save_angiosomes(data, mask, side_prefix, output_dir="separated_cuts"):
    """Cuts the foot into 4 distinct files and returns the arrays for plotting."""
    if not os.path.exists(output_dir): os.makedirs(output_dir)
    
    # Ensure mask is binary and uint8
    mask = (mask > 0).astype(np.uint8)
    
    coords = np.column_stack(np.where(mask > 0))
    if coords.size == 0:
        # Return empty/zeros if mask is empty
        return {}, {}, (0,0,0,0)

    y_min, x_min = coords.min(axis=0); y_max, x_max = coords.max(axis=0)
    h, w = y_max - y_min, x_max - x_min
    
    # Definitions based on bounding box
    v_split = y_min + int(h * 0.60)
    h_split_medial = x_min + int(w * 0.35) 
    toe_limit = y_min + int((v_split - y_min) * 0.50)
    h_split_heel = x_min + int(w * 0.50)

    # 1. MPA (Medial 35% Column)
    mpa_mask = np.zeros_like(mask)
    mpa_mask[y_min:v_split, x_min:h_split_medial] = 1
    mpa_final = np.where((mpa_mask > 0) & (mask > 0), data, 0)
    
    # 2. LPA (Overlapping L-Shape: All Toes + Lateral Mid)
    lpa_mask = np.zeros_like(mask)
    lpa_mask[y_min:toe_limit, x_min:x_max] = 1 
    lpa_mask[toe_limit:v_split, h_split_medial:x_max] = 1 
    lpa_final = np.where((lpa_mask > 0) & (mask > 0), data, 0)
    
    # 3. LCA (Lateral Heel 50%)
    lca_mask = np.zeros_like(mask)
    lca_mask[v_split:y_max, x_min:h_split_heel] = 1
    lca_final = np.where((lca_mask > 0) & (mask > 0), data, 0)
    
    # 4. MCA (Medial Heel 50%)
    mca_mask = np.zeros_like(mask)
    mca_mask[v_split:y_max, h_split_heel:x_max] = 1
    mca_final = np.where((mca_mask > 0) & (mask > 0), data, 0)

    cuts = {"MPA": mpa_final, "LPA": lpa_final, "LCA": lca_final, "MCA": mca_final}
    
    # Save CSVs (Optional feature preserved from user snippet)
    # We can skip actual file saving if not needed for Streamlit performance, 
    # but keeping it as requested logic implies. Only if output_dir is provided.
    
    means = {k: np.mean(v[v > 0]) if np.any(v > 0) else 0 for k, v in cuts.items()}
    return means, cuts, (v_split, h_split_medial, toe_limit, h_split_heel)

def pipeline(l_img_p, l_csv_p, r_img_p, r_csv_p):
    # Process Left
    l_img, _ = image_proc(l_img_p) # We need the mask from image_proc for shape? 
    # Actually, image_proc returns (segmented_foot, binary_mask).
    # But user snippet logic: l_mask = core_segmentation(l_img) ... resize to csv shape.
    # So we need the binary mask from image_proc.
    _, l_mask_orig = image_proc(l_img_p)
    
    # Load Left CSV
    l_data = pd.read_csv(l_csv_p, header=None).to_numpy()
    
    # Resize mask to fit CSV Data
    # CSV data is the "Ground Truth" for resolution in this logic
    l_mask = cv.resize(l_mask_orig, (l_data.shape[1], l_data.shape[0]), interpolation=cv.INTER_NEAREST)
    
    l_means, l_cuts, l_bounds = extract_and_save_angiosomes(l_data, l_mask, "Left")

    # Process Right (Flipped)
    # Right Image
    _, r_mask_orig = image_proc(r_img_p)
    
    # Right CSV (Flipped)
    r_data_raw = pd.read_csv(r_csv_p, header=None).to_numpy()
    r_data = np.fliplr(r_data_raw)
    
    # Resize mask (after flipping the mask? No, mask corresponds to image. Image needs flip? 
    # User snippet: r_img = cv.flip(cv.imread(r_img_p), 1). 
    # r_mask = core_segmentation(r_img).
    # So we should flip the mask from image_proc (which is based on original image).
    r_mask_flipped = cv.flip(r_mask_orig, 1)
    
    r_mask = cv.resize(r_mask_flipped, (r_data.shape[1], r_data.shape[0]), interpolation=cv.INTER_NEAREST)
    
    r_means, r_cuts, r_bounds = extract_and_save_angiosomes(r_data, r_mask, "Right")

    return (l_data, l_cuts, l_bounds, l_means), (r_data, r_cuts, r_bounds, r_means)