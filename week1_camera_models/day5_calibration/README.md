# Day 5 — Camera Calibration

Full camera calibration pipeline with a chessboard pattern (OpenCV + NumPy).

**Mean reprojection error: ~0.18–0.26 px** (depends on your image set).

## Results

### Before vs After

![Original | Undistorted](results/comparison_1.jpg)

### Coordinate axes

![Axes](results/axes_visualization.jpg)

## What this demonstrates

- Chessboard corner detection + sub-pixel refinement
- `cv2.calibrateCamera` → intrinsic matrix `K` and distortion coefficients
- Mean and per-image **reprojection error**
- Undistortion (`cv2.undistort`)
- Pose visualization with `cv2.drawFrameAxes`
- Saving parameters to `camera_calibration.npz` for reuse

## Pipeline

```text
calibration_images/
        │
        ▼
findChessboardCorners + cornerSubPix
        │
        ▼
calibrateCamera → K, dist, rvecs, tvecs
        │
        ▼
reprojection error
        │
        ├── camera_calibration.npz
        ├── corners_*.jpg
        ├── comparison_*.jpg
        └── axes_visualization.jpg
Chessboard settings
PythonCHESSBOARD_SIZE = (9, 6)  # internal corners
SQUARE_SIZE = 0.025       # meters (2.5 cm)
Use the number of internal corners, not squares (10×7 squares → 9×6 corners).
How to run
Bashcd week1_camera_models/day5_calibration

python -m venv venv
# Windows:     venv\Scripts\activate
# macOS/Linux: source venv/bin/activate

pip install numpy opencv-python
python main.py
Put chessboard photos in calibration_images/ (different angles and distances work best).
Project structure
textday5_calibration/
├── main.py
├── calibration_images/
│   ├── img01.jpg
│   └── ...
├── results/
│   ├── camera_calibration.npz
│   ├── corners_*.jpg
│   ├── comparison_1.jpg
│   ├── axes_visualization.jpg
│   └── ...
└── README.md
Outputs

























FileDescriptioncamera_calibration.npzK, dist, errors, image size, board settingscorners_*.jpgDetected cornerscomparison_*.jpgOriginal | Undistortedaxes_visualization.jpg3D axes on the board
Load saved parameters
Pythondata = np.load("results/camera_calibration.npz")
K = data["K"]
dist = data["dist"]
Capture tips

Real phone photos are fine (this project uses them)
10–20 images, varied poses
Full board visible and sharp
Avoid extreme blur and reflections

Why it matters
Accurate K and distortion are the base for undistortion, pose estimation, SfM, AR, and any metric work from images.
text---

### 3. `requirements.txt` (по желанию)

```text
numpy>=1.24
opencv-python>=4.8