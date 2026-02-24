import cv2 as cv
import os
import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt

def image_proc(img_path, target_size=(256, 256)):
    img = cv.imread(img_path)
    if img is None: return None

    # 1. Standardize
    resized = cv.resize(img, target_size, interpolation=cv.INTER_AREA)
    gray = cv.cvtColor(resized, cv.COLOR_BGR2GRAY)
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
    segmented_foot = cv.bitwise_and(resized, resized, mask=binary_mask)

    # Display results
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1); plt.imshow(cv.cvtColor(resized, cv.COLOR_BGR2RGB)); plt.title("Raw Image")
    plt.subplot(1, 3, 2); plt.imshow(binary_mask, cmap='gray'); plt.title("Binary Mask")
    plt.subplot(1, 3, 3); plt.imshow(cv.cvtColor(segmented_foot, cv.COLOR_BGR2RGB)); plt.title("Final Segmented Foot")
    plt.show()
    plt.close() # Good practice in loops to free up memory


image_proc(r"C:\Users\USER\Documents\ThermoDataBase\Control_Group\CG005_F\CG005_F_L.png")