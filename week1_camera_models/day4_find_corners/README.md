# Day 4 — Chessboard Corner Detection

Detect internal chessboard corners and prepare **3D object points** and **2D image points** for camera calibration.

## Result

![Detected corners](results/example_corners.jpg)

*After running the script, each successful detection is saved as `results/{name}_corners.jpg`.*

## What This Demonstrates

- Using a chessboard as a camera calibration pattern.
- Detecting internal corners rather than squares.
- Generating 3D object points on the plane `Z = 0`.
- Detecting corners with `cv2.findChessboardCorners()`.
- Refining corner positions to sub-pixel accuracy with `cv2.cornerSubPix()`.
- Preparing calibration data for `cv2.calibrateCamera()` (Day 5).

## Pipeline

```text
Calibration images
        |
        v
findChessboardCorners()
        |
        v
cornerSubPix()
        |
        v
objpoints (3D) + imgpoints (2D)
        |
        v
Save detected corners and calibration data
        |
        v
results/
```

## Chessboard Settings

```python
CHESSBOARD_SIZE = (9, 6)  # Internal corners: columns x rows
SQUARE_SIZE = 0.025       # Square size in meters (2.5 cm)
```

**Important:** Use the number of internal corners, not the number of squares.

For example, a chessboard with **10 × 7 squares** has **9 × 6 internal corners**.

The `SQUARE_SIZE` value must match the actual size of one square on the physical chessboard.

## How to Run

### 1. Navigate to the Project Directory

From the repository root:

```bash
cd week1_camera_models/day4_find_corners
```

### 2. Create a Virtual Environment (Optional)

```bash
python -m venv venv
```

Activate it on Windows:

```powershell
venv\Scripts\Activate.ps1
```

On macOS or Linux:

```bash
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install numpy opencv-python
```

### 4. Add Chessboard Images

Place your chessboard photos in the `calibration_images/` directory.

### 5. Run the Script

```bash
python main.py
```

The script processes the input images and displays the detected corners in an OpenCV window. Press any key to proceed to the next image.

Successfully processed images and calibration point arrays are saved in the `results/` directory.

## Project Structure

```text
day4_find_corners/
├── main.py
├── calibration_images/
│   ├── img01.jpg
│   ├── img02.jpg
│   └── ...
├── results/
│   ├── img01_corners.jpg
│   ├── img02_corners.jpg
│   ├── objpoints.npy
│   └── imgpoints.npy
└── README.md
```

## Outputs

- **`*_corners.jpg`** — Images with detected chessboard corners visualized.
- **`objpoints.npy`** — 3D coordinates of the chessboard corners.
- **`imgpoints.npy`** — Corresponding 2D image coordinates.

The `.npy` files can be loaded in Day 5 to perform camera calibration using `cv2.calibrateCamera()`.

## Technologies Used

- Python
- NumPy
- OpenCV