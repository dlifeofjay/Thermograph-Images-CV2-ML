"""Test script: Run the full pipeline on DM076_M and visualize all stages."""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
from segmentation import image_proc
from full_pipeline import extract_angiosomes, register_feet, symmetry_analysis, pipeline, mean_temp
import cv2 as cv
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt

# --- Paths ---
LEFT  = r"C:\Users\USER\Documents\MY PORTFOLIO\Jubril_Project\ThermoDataBase\DM_Group\DM076_M\DM076_M_L.png"
RIGHT = r"C:\Users\USER\Documents\MY PORTFOLIO\Jubril_Project\ThermoDataBase\DM_Group\DM076_M\DM076_M_R.png"

print("=" * 60)
print("PIPELINE TEST: DM076_M")
print("=" * 60)

# --- Step 1: Segmentation ---
print("\n[Step 1] Segmentation...")
left_foot, left_mask = image_proc(LEFT)
right_foot, right_mask = image_proc(RIGHT)

print(f"  Left foot shape:  {left_foot.shape}, dtype: {left_foot.dtype}")
print(f"  Left mask shape:  {left_mask.shape}, dtype: {left_mask.dtype}")
print(f"  Left mask unique: {np.unique(left_mask)}")
print(f"  Left foot pixel range: [{left_foot.min()}, {left_foot.max()}]")
print(f"  Right foot shape: {right_foot.shape}, dtype: {right_foot.dtype}")
print(f"  Right mask shape: {right_mask.shape}, dtype: {right_mask.dtype}")
print(f"  Right mask unique: {np.unique(right_mask)}")
print(f"  Right foot pixel range: [{right_foot.min()}, {right_foot.max()}]")

# --- Step 2: Registration ---
print("\n[Step 2] Registration (flip right foot)...")
right_foot_reg, right_mask_reg = register_feet(right_foot, right_mask)
print(f"  Right foot (registered) shape: {right_foot_reg.shape}")

# --- Step 3: Angiosome Extraction ---
print("\n[Step 3] Angiosome Extraction...")
left_angio = extract_angiosomes(left_foot, left_mask)
right_angio = extract_angiosomes(right_foot_reg, right_mask_reg)

angio_names = ["MPA", "LPA", "MCA", "LCA"]
print("  Left angiosome temperatures:")
for name, region in zip(angio_names, left_angio):
    print(f"    {name}: shape={region.shape}, mean_temp={mean_temp(region):.2f}")

print("  Right angiosome temperatures:")
for name, region in zip(angio_names, right_angio):
    print(f"    {name}: shape={region.shape}, mean_temp={mean_temp(region):.2f}")

# --- Step 4: Symmetry Analysis ---
print("\n[Step 4] Symmetry Analysis (|T_left - T_right|)...")
diffs = symmetry_analysis(left_angio, right_angio)
for name, diff in diffs.items():
    flag = "[HIGH RISK]" if diff > 2.6 else ("[WATCH]" if diff > 1.8 else "[NORMAL]")
    print(f"  {name}: {diff:.2f} C  ->  {flag}")

# --- Visualization ---
print("\n[Saving Visualization...]")

fig, axes = plt.subplots(3, 4, figsize=(20, 15))
fig.suptitle("Pipeline Test: DM076_M", fontsize=16, fontweight='bold')

# Row 1: Raw images, masks
axes[0,0].imshow(cv.cvtColor(left_foot, cv.COLOR_BGR2RGB)); axes[0,0].set_title("Left Foot (Segmented)")
axes[0,1].imshow(left_mask, cmap='gray'); axes[0,1].set_title("Left Mask")
axes[0,2].imshow(cv.cvtColor(right_foot, cv.COLOR_BGR2RGB)); axes[0,2].set_title("Right Foot (Segmented)")
axes[0,3].imshow(right_mask, cmap='gray'); axes[0,3].set_title("Right Mask")

# Row 2: Registration + angiosomes
axes[1,0].imshow(cv.cvtColor(left_foot, cv.COLOR_BGR2RGB)); axes[1,0].set_title("Left (original)")
axes[1,1].imshow(cv.cvtColor(right_foot_reg, cv.COLOR_BGR2RGB)); axes[1,1].set_title("Right (flipped/registered)")

# Draw angiosome grid on left
left_vis = left_foot.copy()
x, y, w, h = cv.boundingRect(left_mask)
medial_split = int(0.35 * w)
upper_split = int(0.60 * h)
cv.line(left_vis, (x + medial_split, y), (x + medial_split, y + h), (0, 255, 0), 2)
cv.line(left_vis, (x, y + upper_split), (x + w, y + upper_split), (0, 255, 0), 2)
cv.rectangle(left_vis, (x, y), (x + w, y + h), (255, 0, 0), 2)
axes[1,2].imshow(cv.cvtColor(left_vis, cv.COLOR_BGR2RGB)); axes[1,2].set_title("Left Angiosome Grid")

# Draw angiosome grid on right (registered)
right_vis = right_foot_reg.copy()
x2, y2, w2, h2 = cv.boundingRect(right_mask_reg)
medial_split2 = int(0.35 * w2)
upper_split2 = int(0.60 * h2)
cv.line(right_vis, (x2 + medial_split2, y2), (x2 + medial_split2, y2 + h2), (0, 255, 0), 2)
cv.line(right_vis, (x2, y2 + upper_split2), (x2 + w2, y2 + upper_split2), (0, 255, 0), 2)
cv.rectangle(right_vis, (x2, y2), (x2 + w2, y2 + h2), (255, 0, 0), 2)
axes[1,3].imshow(cv.cvtColor(right_vis, cv.COLOR_BGR2RGB)); axes[1,3].set_title("Right Angiosome Grid")

# Row 3: Angiosome regions close-up
for i, (name, l_reg, r_reg) in enumerate(zip(angio_names, left_angio, right_angio)):
    combined = np.hstack([
        cv.cvtColor(l_reg, cv.COLOR_BGR2RGB) if len(l_reg.shape) == 3 else l_reg,
        cv.cvtColor(r_reg, cv.COLOR_BGR2RGB) if len(r_reg.shape) == 3 else r_reg
    ])
    axes[2, i].imshow(combined)
    axes[2, i].set_title(f"{name}: L={mean_temp(l_reg):.1f}° | R={mean_temp(r_reg):.1f}° | Δ={diffs[name]:.2f}°")

for ax in axes.flat:
    ax.axis('off')

plt.tight_layout()
plt.savefig(r"C:\Users\USER\Documents\MY PORTFOLIO\Jubril_Project\test_dm076_output.png", dpi=150)
print("  Saved: test_dm076_output.png")
print("\n" + "=" * 60)
print("TEST COMPLETE")
print("=" * 60)
