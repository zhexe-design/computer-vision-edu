# Day 3 — Lens Distortion

Apply artificial **radial (barrel) distortion** to photos and remove it with OpenCV.

Produces a side-by-side comparison: **Original | Distorted | Undistorted**.

## Result

![Comparison](results/1_comparison.jpg)

Left → right: Original | Distorted (barrel) | Undistorted

## What this demonstrates

- Radial distortion coefficients `k1`, `k2`, `k3` (and tangential `p1`, `p2`)
- Barrel distortion with negative `k1`
- Approximate *adding* of distortion via inverse undistortion map
- Correction with `cv2.undistort`
- Visual comparison of straight lines before / after

## Distortion model (OpenCV)

```text
dist = [k1, k2, p1, p2, k3]
Coefficients used in this project:
PythonDIST_COEFFS = [-0.3, 0.1, 0.0, 0.0, 0.0]  # k1 < 0 → barrel

Barrel — lines curve outward from the center (k1 < 0)
Pincushion — lines curve inward (k1 > 0)

Pipeline
textClean image
    │
    ▼
initUndistortRectifyMap + remap   ≈ add distortion
    │
    ▼
cv2.undistort                     → remove distortion
    │
    ▼
hstack: Original | Distorted | Undistorted
Note: Adding distortion is approximated by applying the undistortion map to a clean image (inverse warp). Undistorting that result recovers a close version of the original. This is a standard learning demo; a full forward distortion model is more involved.
How to run
Bashcd week1_camera_models/day3_distortion

python -m venv venv
# Windows:     venv\Scripts\activate
# macOS/Linux: source venv/bin/activate

pip install numpy opencv-python
python main.py

Put photos in images/ (jpg / png).
Run the script.
Press any key in the OpenCV window to go to the next image.
Results are written to results/.

Project structure
textday3_distortion/
├── main.py
├── images/
│   ├── 1.jpg
│   ├── 2.jpg
│   └── 3.jpg
├── results/
│   ├── 1_original.jpg
│   ├── 1_distorted.jpg
│   ├── 1_undistorted.jpg
│   └── 1_comparison.jpg
└── README.md