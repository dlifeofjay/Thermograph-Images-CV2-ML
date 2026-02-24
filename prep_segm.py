import cv2 as cv
import os
import numpy as np
from scipy import ndimage

def load_image():
    path = r'C:\Users\USER\Documents\ThermoDataBase'
    groups = ['Control_Group', 'DM_Group']
    output_path = r"C:\Users\USER\Documents\Processed_Data"
    
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    def image_proc(img_path, target_size=(256, 256)):
        img = cv.imread(img_path)
        if img is None:
            return None
    
        # 1. Standardize
        resized = cv.resize(img, target_size, interpolation=cv.INTER_AREA)
        gray = cv.cvtColor(resized, cv.COLOR_BGR2GRAY)
        enhanced = cv.createCLAHE(clipLimit=3.0, tileGridSize=(8,8)).apply(gray)
    
        # 2. Hybrid Thresholding - FORCE WHITE FOREGROUND
        # We use _INV here because the background of the scans is darker than the foot
        _, mask_otsu = cv.threshold(enhanced, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
        mask_adapt = cv.adaptiveThreshold(enhanced, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                          cv.THRESH_BINARY, 21, 2)
    
        # 3. Combine and INVERT if necessary 
        # Logic: We want the FOOT to be white. 
        combined = cv.bitwise_or(mask_otsu, mask_adapt)
        
        # Check if the mask is mostly white (inverted). If so, flip it.
        if np.mean(combined) > 127:
            combined = cv.bitwise_not(combined)
    
        # 4. Morphology (The "Healing" phase for Cold Feet)
        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (7, 7))
        # Close gaps in ischemic zones
        mop = cv.morphologyEx(combined, cv.MORPH_CLOSE, kernel, iterations=2)
        # Fill internal "cold" holes
        mask_filled = ndimage.binary_fill_holes(mop).astype(np.uint8) * 255
    
        # 5. Extract Largest Component (Isolates Foot from Noise)
        contours, _ = cv.findContours(mask_filled, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        binary_mask = np.zeros_like(mask_filled)
        if contours:
            largest_cnt = max(contours, key=cv.contourArea)
            cv.drawContours(binary_mask, [largest_cnt], -1, 255, thickness=cv.FILLED)
    
        # 6. Final Segmentation
        segmented_foot = cv.bitwise_and(resized, resized, mask=binary_mask)
        return segmented_foot

    for group in groups:
        folder = os.path.join(path, group)
        if not os.path.exists(folder):
            continue
            
        for patient_id in os.listdir(folder):
            patient_folder_path = os.path.join(folder, patient_id)
            if not os.path.isdir(patient_folder_path):
                continue
                
            images = [img for img in os.listdir(patient_folder_path) if img.endswith(".png")]
            
            if len(images) != 2:
                print(f"Patient {patient_id} does not have 2 images.")
                continue
                
            for img_name in images:
                img_path = os.path.join(patient_folder_path, img_name)
                processed_img = image_proc(img_path)
                
                if processed_img is not None:
                    save_name = f"{group}_{patient_id}_{img_name}"
                    save_path = os.path.join(output_path, save_name)
                    
                    # FIXED: No need to multiply by 255, just save directly
                    cv.imwrite(save_path, processed_img)
                    
    print("Hybrid Segmentation and Standardization finished!")

if __name__ == "__main__":
    load_image()