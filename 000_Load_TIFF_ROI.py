import numpy as np
import matplotlib.pyplot as plt
import os
from tifffile import imread
from read_roi import read_roi_file
from skimage.draw import polygon, rectangle, ellipse
import pandas as pd

# === FILE BASE SETUP ===
base_dir = 'your/base/directory/path'  # Change this to your actual base directory
base_name = "your_base_name_here"  # Change this to your actual base name
roi_path = os.path.join(base_dir, "your_roi_file.roi")  # Change this to your actual ROI file path
output_npy_path = os.path.join(base_dir, base_name + ".npy")
output_csv_path = os.path.join(base_dir, base_name + ".csv")
output_png_path = os.path.join(base_dir, base_name + "_trace.png")

# === FRAME RATE (FPS) ===
fps = 458.66  # from framePeriod = 0.002180313 s in Bruker metadata
frame_time = 1 / fps

# === 1. Automatically find TIFF file ===
possible_extensions = [".ome.tif", ".tif", ".ome", ""]
raw_path = None
for ext in possible_extensions:
    candidate = os.path.join(base_dir, base_name + ext)
    if os.path.isfile(candidate):
        raw_path = candidate
        break
if raw_path is None:
    raise FileNotFoundError(f"❌ Could not find TIFF file using base name: {base_name} in {base_dir}")
print(f"✅ Found raw TIFF file: {raw_path}")

# === 2. Load TIFF ===
raw_data = imread(raw_path)
print("📦 Loaded TIFF shape:", raw_data.shape)

# === 3. Load ROI and create mask ===
roi_dict = read_roi_file(roi_path)
roi = next(iter(roi_dict.values()))
shape = raw_data.shape[1:]
mask = np.zeros(shape, dtype=bool)
roi_type = roi.get("type", "").lower()

if 'x' in roi and 'y' in roi:
    rr, cc = polygon(roi['y'], roi['x'], shape=shape)
    mask[rr, cc] = True
elif roi_type == "rectangle":
    rr, cc = rectangle((roi["top"], roi["left"]), extent=(roi["height"], roi["width"]), shape=shape)
    mask[rr, cc] = True
elif roi_type == "oval":
    cy = roi["top"] + roi["height"] // 2
    cx = roi["left"] + roi["width"] // 2
    ry = roi["height"] // 2
    rx = roi["width"] // 2
    rr, cc = ellipse(cy, cx, ry, rx, shape=shape)
    mask[rr, cc] = True
else:
    raise ValueError(f"Unsupported ROI type: {roi_type}")

# === 4. Extract trace from ROI ===
trace = raw_data[:, mask].mean(axis=1)

# === 5. Normalize to −ΔF/F ===
def deltaF_over_F(trace, baseline_frames=500):
    F0 = np.percentile(trace[:baseline_frames], 20)
    return (trace - F0) / F0

df_over_f = -1 * deltaF_over_F(trace)  # Negated

# === 6. Create time vector ===
n_frames = len(df_over_f)
time_s = np.arange(n_frames) * frame_time

# === 7. Save to .npy ===
np.save(output_npy_path, df_over_f)
print(f"💾 Saved −ΔF/F to: {output_npy_path}")

# === 8. Save to .csv ===
df = pd.DataFrame({"time_s": time_s, "df_over_f": df_over_f})
df.to_csv(output_csv_path, index=False)
print(f"📄 Saved CSV to: {output_csv_path}")

# === 9. Save plot to .png ===
plt.figure(figsize=(10, 4))
plt.plot(time_s, df_over_f, color="blue")
plt.title("−ΔF/F Trace from ROI (Raw TIFF)")
plt.xlabel("Time (s)")
plt.ylabel("−ΔF/F")
plt.grid(True)
plt.tight_layout()
plt.savefig(output_png_path, dpi=300)
print(f"🖼️ Saved PNG plot to: {output_png_path}")
plt.show()
