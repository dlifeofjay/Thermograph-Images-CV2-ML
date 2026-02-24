import cv2 as cv
import os
import numpy as np
from scipy import ndimage

def image_proc(img_path):
        img = cv.imread(img_path)
        if img is None:
            return None
    
        # 1. grayscaling
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        enhanced = cv.createCLAHE(clipLimit=3.0, tileGridSize=(8,8)).apply(gray)
    
        # 2. Hybrid Thresholding - FORCE WHITE FOREGROUND
        # We use _INV here because the background of your scans is darker than the foot
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
        segmented_foot = cv.bitwise_and(img, img, mask=binary_mask)
        return segmented_foot, binary_mask