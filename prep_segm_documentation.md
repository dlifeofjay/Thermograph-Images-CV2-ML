# Documentation: `prep_segm.py` - Preprocessing & Segmentation Module

## 1. Overview
The `prep_segm.py` script serves as the foundational preprocessing module for the Diabetic Foot Complication Analysis project. It automates the ingestion, standardization, and segmentation of plantar thermograms from the `ThermoDataBase`.

This module directly addresses **Deliverable 2: Preprocessing & Segmentation Report** by implementing a robust "Hybrid Segmentation" algorithm designed to handle the specific challenges of diabetic foot thermography, particularly for "Cold/Ischemic" feet where thermal contrast is compromised.

## 2. Methodology & Algorithm Implementation

The script executes a sequential image processing pipeline:

### Phase 1: Standardization
*   **Input**: Raw thermogram images (various resolutions).
*   **Resize**: All images are resized to a fixed `256x256` resolution (Line 20) to ensure consistency for future Machine Learning models.
*   **Grayscale Conversion**: Converts BGR images to single-channel grayscale (Line 21).
*   **CLAHE (Contrast Limited Adaptive Histogram Equalization)** (Line 22):
    *   **Purpose**: Enhances local contrast in small regions (tiles) of the image.
    *   **Relevance to Deliverable 2**: This is critical for **"Cold/Ischemic" feet**. Ischemic regions have low temperatures similar to the background, creating poor contrast. A standard global histogram equalization would fail here. CLAHE amplifies the subtle thermal differences at the foot boundaries, making segmentation possible even for cold feet.

### Phase 2: Hybrid Thresholding
The core "Hybrid Segmentation" logic combines two thresholding techniques (Line 32):

1.  **Otsu's Binarization** (Line 26):
    *   Automatically calculates an optimal global threshold based on the image histogram.
    *   *Role*: Captures the general foot shape effectively when there is good thermal contrast.
2.  **Adaptive Gaussian Thresholding** (Line 27):
    *   Calculates thresholds for small regions based on a Gaussian-weighted sum of neighbors.
    *   *Role*: Handles varying lighting/temperature gradients across the foot. It is essential for capturing the edges of **ischemic feet** where the temperature might drop gradually to match the background.

**Combination Logic**: `cv.bitwise_or(mask_otsu, mask_adapt)` creates a union of both masks, ensuring that if either method detects a foot region, it is preserved.

### Phase 3: Morphological "Healing"
This phase specifically targets the structural integrity of the segmentation for cold feet:

1.  **Morphological Closing** (Line 41):
    *   **Operation**: Dilation followed by Erosion.
    *   **Purpose**: "Closes" small gaps or breaks in the foot contour. For ischemic feet, the toes or heel might appear disconnected due to low temperature; this step bridges those gaps.
2.  **Fill Holes** (Line 43):
    *   **Operation**: `ndimage.binary_fill_holes`.
    *   **Purpose**: Fills any internal black regions within the white foot mask. Ischemic feet often have "cold spots" in the center (e.g., the arch or heel) that drop below the threshold. This step forces them to be part of the foreground, ensuring a complete, solid mask.

### Phase 4: Contour Extraction & Masking
*   **Largest Component Selection** (Line 46-50): Identifies all contours and selects the largest one, assuming it is the foot. This removes background noise and artifacts.
*   **Final Segmentation** (Line 53): Applies the generated binary mask to the original (resized) image, isolating the foot against a pure black background.

## 3. Compliance with Deliverable 2

**Deliverable 2 Requirement**: *"Visual validation of the 'Hybrid Segmentation' (Histogram + Morphology), specifically demonstrating success on 'Cold/Ischemic' feet."*

How `prep_segm.py` satisfies this:
1.  **Hybrid Approach**: The code explicitly implements the **Histogram** (Otsu) + **Adaptive** thresholding combination described in the deliverable.
2.  **Morphology for Ischemia**: The dedicated "Morphology" section (Lines 38-43) is the technical solution for "Cold/Ischemic" feet. By using CLAHE to boost contrast and Morphological Closing/Filling to repair signal dropout in cold regions, the script ensures that ischemic feet are not fragmented or lost during segmentation.
3.  **Automated Pipeline**: It provides the mechanism to generate the segmented images required for the "Visual Validation" report.

## 4. Usage
To run the segmentation pipeline:

```bash
python prep_segm.py
```

*   **Input Directory**: `C:\Users\USER\Documents\ThermoDataBase`
*   **Output Directory**: `C:\Users\USER\Documents\Processed_Data`
*   **Logic**: Iterates through `Control_Group` and `DM_Group`, processes every patient's images, and saves the segmented output.
