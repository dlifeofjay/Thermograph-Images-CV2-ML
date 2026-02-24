# Deliverable 2: Preprocessing & Segmentation Technical Report

**Project Title:** Diabetic Foot Complication Analysis System  
**Module:** Preprocessing & Segmentation (`prep_segm.py`)  
**Date:** February 9, 2026  
**Author:** Jubril  

---

## **Table of Contents**

1.0 **Executive Summary**  
2.0 **Introduction**  
3.0 **Technical Implementation**  
&nbsp;&nbsp;&nbsp;&nbsp;3.1 Standardization Pipeline  
&nbsp;&nbsp;&nbsp;&nbsp;3.2 Hybrid Segmentation Algorithm  
&nbsp;&nbsp;&nbsp;&nbsp;3.3 Morphological Post-Processing  
4.0 **Validation Strategy**  
&nbsp;&nbsp;&nbsp;&nbsp;4.1 Handling Cold/Ischemic Feet  
&nbsp;&nbsp;&nbsp;&nbsp;4.2 Deliverable Compliance Matrix  
5.0 **Usage Instructions**  

---

## **1.0 Executive Summary**

This document serves as the technical specification for the Preprocessing and Segmentation module (`prep_segm.py`) of the Diabetic Foot Complication Analysis System. The primary objective of this module is to ingest raw plantar thermograms and isolate the foot region of interest (ROI) from the background. 

The implemented solution utilizes a **Hybrid Segmentation Approach**, combining global histogram analysis (Otsu’s Method) with local adaptive thresholding. This dual strategy specifically addresses the challenge of segmenting "Cold" or "Ischemic" feet, where thermal contrast against the background is minimal. The resulting segmented images form the foundational dataset for subsequent signal processing and diagnostic modeling.

---

## **2.0 Introduction**

Diabetic foot complications, particularly ischemia, manifest as reduced skin temperature. In thermal imaging, this results in a low signal-to-noise ratio, making standard segmentation techniques unreliable. 

The `prep_segm.py` script aims to:
1.  **Standardize** all input images to a uniform resolution ($256 \times 256$) and dynamic range.
2.  **Segment** the foot shape accurately, preserving anatomical boundaries even in low-temperature regions.
3.  **Automate** the processing pipeline for large datasets (Control vs. DM groups).

This report details the algorithmic logic and validates its alignment with the project's **Deliverable 2** requirements.

---

## **3.0 Technical Implementation**

The segmentation pipeline is implemented in Python using OpenCV (`cv2`) and SciPy (`ndimage`). The process flow is strictly sequential:

### **3.1 Standardization Pipeline**

Before segmentation, raw images undergo preprocessing to ensure consistency across the dataset.

*   **Resizing**: Images are resized to $256 \times 256$ pixels using Area Interpolation (`cv.INTER_AREA`) to minimize aliasing artifacts.
*   **Grayscale Conversion**: Multi-channel color information is discarded to focus solely on thermal intensity (luminance).
*   **Contrast Enhancement (CLAHE)**: 
    *   *Technique*: Contrast Limited Adaptive Histogram Equalization.
    *   *Parameters*: Clip Limit = 3.0, Tile Grid Size = $(8,8)$.
    *   *Purpose*: Unlike global histogram equalization, CLAHE operates on small regions. This locally enhances the contrast at the foot boundaries, making the faint edges of cold feet distinguishable from the background.

### **3.2 Hybrid Segmentation Algorithm**

To ensure robustness, the system employs a hybrid thresholding logic:

1.  **Global Thresholding (Otsu's Binarization)**:
    *   *Function*: Calculates a single intensity threshold that separates pixels into two classes (foreground/background) by minimizing intra-class variance.
    *   *Strength*: Highly effective for "Warm" feet with high thermal contrast.
    
2.  **Local Thresholding (Adaptive Gaussian)**:
    *   *Function*: Calculates thresholds for small neighborhoods based on a Gaussian-weighted sum.
    *   *Strength*: Essential for differing lighting conditions and "Cold" feet where the temperature gradient fades into the background.

**Fusion Logic**: The two masks are combined using a bitwise **OR** operation. This ensures that if *either* method detects a foot pixel, it is retained.

### **3.3 Morphological Post-Processing**

The raw binary mask often contains noise or gaps, particularly in ischemic regions.

*   **Morphological Closing**: A dilation followed by an erosion closes small gaps in the contour, ensuring the foot boundary is continuous.
*   **Hole Filling**: The `ndimage.binary_fill_holes` function is applied to identifying and filling internal "black holes" within the foot mask. These holes correspond to the coldest parts of the foot (e.g., arch or heel) that fell below the threshold. Filling them ensures a complete ROI.
*   **Largest Component Extraction**: Finally, the system identifies all connected components and retains only the largest one, effectively filtering out background thermal noise or artifacts.

---

## **4.0 Validation Strategy**

This section outlines how the implementation satisfies the specific requirements of Deliverable 2.

### **4.1 Handling Cold/Ischemic Feet**

The primary failure mode in thermal segmentation is the loss of cold extremities (toes/heels) which blend into the background. The `prep_segm.py` script mitigates this via:

1.  **CLAHE Pre-processing**: Boosts the edge signal of cold feet before thresholding is attempted.
2.  **Adaptive Thresholding**: Catches the faint gradients of ischemic toes that Otsu's method misses.
3.  **Hole Filling**: Recovers the internal structure of the foot if the center is "too cold" to be detected initially.

### **4.2 Deliverable Compliance Matrix**

| Requirement Ref. | Requirement Description | Implementation Verification |
| :--- | :--- | :--- |
| **D2.1** | Visual Validation of Hybrid Segmentation | Script generates specific segmented outputs in `Processed_Data` for visual inspection. |
| **D2.2** | Success on "Cold/Ischemic" Feet | Combination of CLAHE + Adaptive Thresholding + Morphology explicitly targets low-contrast regions. |
| **D2.3** | Histogram + Morphology Logic | Implemented via `cv.threshold` (Otsu) and `cv.morphologyEx` / `ndimage.binary_fill_holes`. |

---

## **5.0 Usage Instructions**

To execute the segmentation pipeline:

1.  **Environment Setup**: Ensure `opencv-python` and `scipy` are installed.
2.  **Directory Structure**: Place raw images in `Documents\ThermoDataBase`.
3.  **Execution**: Run the script via terminal:
    ```bash
    python prep_segm.py
    ```
4.  **Output**: Processed images will be saved to `Documents\Processed_Data`, retaining the original filename structure for traceability.

---
**End of Report**
