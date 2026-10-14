# Day 4 — Chessboard Corner Detection

Detect internal chessboard corners and prepare **3D object points** + **2D image points** for camera calibration.

## Result

![Detected corners](results/example_corners.jpg)

*(After running the script, each successful detection is saved as `results/{name}_corners.jpg`.)*

## What this demonstrates

- Chessboard as a calibration pattern (internal corners, not squares)
- 3D object points on the plane `Z = 0`
- `cv2.findChessboardCorners` + `cv2.cornerSubPix`
- Data ready for `cv2.calibrateCamera` (Day 5)

## Pipeline

```text
calibration images
        │
        ▼
findChessboardCorners
        │
        ▼
cornerSubPix (sub-pixel refine)
        │
        ▼
objpoints (3D) + imgpoints (2D)
        │
        ▼
results/  +  *.npy for Day 5
Chessboard settings
PythonCHESSBOARD_SIZE = (9, 6)  # internal corners: columns × rows
SQUARE_SIZE = 0.025       # meters (2.5 cm)
Important: use the number of internal corners, not the number of squares.

A board with 10 × 7 squares has 9 × 6 internal corners.
How to run
Bashcd week1_camera_models/day4_find_corners

python -m venv venv
# Windows:     venv\Scripts\activate
# macOS/Linux: source venv/bin/activate

pip install numpy opencv-python
python main.py

Put chessboard photos in calibration_images/.
Run the script.
A window shows each detection (~0.5 s).
Outputs go to results/.

Project structure
textday4_find_corners/
├── main.py
├── calibration_images/
│   ├── img01.jpg
│   ├── img02.jpg
│   └── ...
├── results/
│   ├── img01_corners.jpg
│   ├── objpoints.npy
│   └── imgpoints.npy
└── README.md
Outputs





















FileDescriptionresults/{name}_corners.jpgImage with drawn cornersresults/objpoints.npyList of 3D point setsresults/imgpoints.npyList of 2D point sets
Tips for good detection

Use 8–15 images from different angles and distances
Keep the full board visible and in focus
Avoid strong blur and extreme perspective
Match CHESSBOARD_SIZE to your physical board

text---

### 3. `requirements.txt` (по желанию)

```text
numpy>=1.24
opencv-python>=4.8