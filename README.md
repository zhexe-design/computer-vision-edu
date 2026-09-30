# 3D Computer Vision Learning Projects

Learning repository focused on the foundations of 3D Computer Vision and camera geometry.

Currently completed: **Month 1 — Weeks 1–2**
(Camera models, distortion, chessboard detection, and full smartphone camera calibration.)

## Why this repository

I am building practical skills for roles in:

* 3D Computer Vision
* Photogrammetry
* Neural Rendering (3D Gaussian Splatting / NeRF)
* Scan-to-BIM / Digital Twins
* Spatial AI

All projects use real smartphone data whenever possible.

## Projects (Month 1)

| Folder                                                           | Description                               | Key Result                            |
| ---------------------------------------------------------------- | ----------------------------------------- | ------------------------------------- |
| [day1_pinhole](./week1_camera_models/day1_pinhole)               | Pinhole camera model + 3D → 2D projection | Cube + axes projection                |
| [day2_camera_control](./week1_camera_models/day2_camera_control) | Camera orbit and control                  | Orbit animation                       |
| [day3_distortion](./week1_camera_models/day3_distortion)         | Lens distortion (barrel / pincushion)     | Before / After comparison             |
| [day4_find_corners](./week1_camera_models/day4_find_corners)     | Chessboard corner detection               | Prepared data for calibration         |
| [day5_calibration](./week1_camera_models/day5_calibration)       | Full camera calibration                   | **Reprojection error ≈ 0.18–0.26 px** |
| [day6_testing](./week1_camera_models/day6_testing)               | Additional tests and verification         | —                                     |

## Best Result so far

### Smartphone Camera Calibration

The main result of this part of the project is a full camera calibration using smartphone images.

* Mean reprojection error: **~0.18–0.26 pixels**
* Intrinsic camera matrix **K** calculated
* Distortion coefficients calculated
* Undistortion results included
* Camera axes visualization included
* Calibration parameters saved for further use

## Visual Results

### Camera Calibration

Calibration results and visualizations are available in the [`day5_calibration`](./week1_camera_models/day5_calibration) folder.

![Camera axes visualization](./week1_camera_models/day5_calibration/axes_visualization.jpg)

### Undistortion

Examples of the original and corrected images are available in the calibration results.

![Calibration comparison](./week1_camera_models/day5_calibration/comparison_1.jpg)

## Tech Stack

* Python 3
* OpenCV
* NumPy
* ImageIO

## How to Run

Clone the repository:

```bash
git clone https://github.com/zhexe-design/computer-vision-edu.git
cd computer-vision-edu
```

Create a virtual environment:

```bash
python -m venv venv
```

Activate it:

```bash
# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

Install the required packages:

```bash
pip install opencv-python numpy imageio
```

## Repository Structure

```text
computer-vision-edu/
│
└── week1_camera_models/
    ├── day1_pinhole/
    ├── day2_camera_control/
    ├── day3_distortion/
    ├── day4_find_corners/
    ├── day5_calibration/
    └── day6_testing/
```

## Progress

**Month 1 — Weeks 1–2: Completed**

* [x] Pinhole camera model
* [x] 3D → 2D projection
* [x] Camera orbit and control
* [x] Lens distortion
* [x] Chessboard corner detection
* [x] Smartphone camera calibration
* [x] Reprojection error analysis
* [x] Undistortion
* [x] Calibration testing
