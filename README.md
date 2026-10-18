# 3D Computer Vision Learning Projects

A practical learning repository focused on the foundations of **3D Computer Vision, camera geometry, and image processing**.

The projects are organized by weeks and days, documenting the implementation process, experiments, results, and lessons learned.

**Current progress: Month 1 — Weeks 1–2**

Topics covered:
- Pinhole camera model and 3D-to-2D projection
- Camera orbit and control
- Lens distortion
- Chessboard corner detection
- Smartphone camera calibration
- Reprojection error analysis
- Image undistortion
- Camera calibration quality improvement

---

## Why This Repository?

I am building practical skills for future roles and projects in:

- 3D Computer Vision
- Photogrammetry
- Neural Rendering (3D Gaussian Splatting / NeRF)
- Scan-to-BIM and Digital Twins
- Spatial AI

Whenever possible, the projects use real smartphone photographs rather than synthetic datasets. This provides practical experience with real-world image quality, camera distortion, lighting conditions, and calibration challenges.

---

## Projects Overview

### Week 1 — Camera Models

| Project | Description | Key Result |
|---|---|---|
| [Day 1 — Pinhole Camera](./week1_camera_models/day1_pinhole) | Pinhole camera model and 3D-to-2D projection | Projected 3D cube and coordinate axes |
| [Day 2 — Camera Control](./week1_camera_models/day2_camera_control) | Camera orbit and look-at transformation | Camera orbit animation |
| [Day 3 — Lens Distortion](./week1_camera_models/day3_distortion) | Radial lens distortion and image correction | Before-and-after distortion comparison |
| [Day 4 — Chessboard Detection](./week1_camera_models/day4_find_corners) | Chessboard corner detection | Detected internal corners |
| [Day 5 — Camera Calibration](./week1_camera_models/day5_calibration) | Camera calibration using a chessboard pattern | Initial calibration error of approximately 0.18–0.26 px |
| [Day 6 — Calibration Testing](./week1_camera_models/day6_testing) | Additional calibration tests and verification | Additional validation experiments |

### Week 2 — Camera Calibration and Quality Analysis

Week 2 builds on the camera geometry concepts from Week 1 and focuses on a more complete smartphone camera calibration workflow.

| Project | Description | Key Result |
|---|---|---|
| [Day 1 — Full Camera Calibration](./week2_camera_calibration/day1) | Complete calibration pipeline using real smartphone photographs | Intrinsic matrix, distortion coefficients, and undistortion results |
| [Day 2 — Calibration Quality Analysis](./week2_camera_calibration/day2) | Per-image reprojection error analysis and calibration refinement | Mean reprojection error reduced from 1.43 px to 0.96 px |

**Note:** The Week 1 and Week 2 calibration experiments use different datasets or calibration runs. Their results should be evaluated separately rather than treated as directly comparable measurements.

---

## Best Result So Far

### Week 2 — Improved Smartphone Camera Calibration

The main result of Week 2 is an improved camera calibration obtained by analyzing individual image reprojection errors and excluding four high-error images.

| Metric | Initial Calibration | Final Calibration |
|---|---:|---:|
| Calibration images | 23 | 19 |
| Mean reprojection error | 1.43 px | 0.96 px |
| RMS calibration error | 1.97 px | 1.32 px |
| Images removed | 0 | 4 |

**Mean reprojection error improvement: 32.94%.**

The mean reprojection error decreased by 0.47 px after filtering the four images with the highest errors.

The final calibration also produced:
- An estimated camera intrinsic matrix `K`
- Radial and tangential distortion coefficients
- Original-versus-undistorted image comparisons
- Chessboard corner detection visualizations
- A 3D coordinate axes visualization
- Saved calibration parameters for reuse

These results demonstrate how image-level error analysis can help improve calibration metrics.

However, a lower reprojection error does not automatically guarantee better undistortion for every photograph. Calibration quality should also be evaluated using independent test images and the intended downstream application.

---

## Visual Results

### Week 1 — Camera Calibration

The Week 1 calibration visualizations are available in the corresponding project folder.

**Camera axes visualization**

![Camera axes visualization](./week1_camera_models/day5_calibration/axes_visualization.jpg)

**Undistortion comparison**

![Week 1 calibration comparison](./week1_camera_models/day5_calibration/comparison_1.jpg)

### Week 2 — Calibration Quality

Week 2 includes additional visualizations for evaluating calibration quality.

**Undistortion tests**

The original and corrected versions of ordinary photographs are saved in the Week 2 results directory.

**Chessboard corner detection**

Detected internal corners are visualized to verify successful chessboard detection.

**Calibration quality comparison**

Initial and final reprojection errors are compared to evaluate the effect of excluding four high-error images.

See the [Week 2 calibration project](./week2_camera_calibration/day1) and [calibration quality analysis](./week2_camera_calibration/day2) for details and generated outputs.

---

## Technologies Used

- **Python 3** — implementation and experimentation
- **OpenCV** — camera calibration, chessboard detection, image undistortion, and visualization
- **NumPy** — numerical operations and camera geometry
- **ImageIO** — image and animation processing where required
- **PyYAML** — reading and writing YAML calibration parameters

---

## How to Run

### 1. Clone the Repository

```bash
git clone https://github.com/zhexe-design/computer-vision-edu.git
cd computer-vision-edu
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
```

### 3. Activate the Environment

**Windows PowerShell:**

```powershell
.\venv\Scripts\Activate.ps1
```

**Windows Command Prompt:**

```bat
venv\Scripts\activate.bat
```

**macOS / Linux:**

```bash
source venv/bin/activate
```

### 4. Install Dependencies

For the camera calibration projects:

```bash
pip install opencv-python numpy imageio pyyaml
```

Individual projects may require additional dependencies. Check the relevant project README before running a script.

### 5. Run a Project

Navigate to the appropriate project directory and follow its README instructions.

For example, to run the Week 2 calibration quality analysis:

```bash
cd week2_camera_calibration/day2
python day2_calibration_quality.py
```

**Important:** Script names and relative paths must match the actual repository structure. Run each script from the directory expected by its implementation.

---

## Repository Structure

```text
computer-vision-edu/
│
├── week1_camera_models/
│   ├── day1_pinhole/
│   ├── day2_camera_control/
│   ├── day3_distortion/
│   ├── day4_find_corners/
│   ├── day5_calibration/
│   └── day6_testing/
│
├── week2_camera_calibration/
│   ├── day1/
│   │   ├── calibration_images/
│   │   ├── results/
│   │   ├── day1_full_calibration.py
│   │   └── README.md
│   │
│   ├── day2/
│   │   ├── results/
│   │   ├── day2_calibration_quality.py
│   │   └── README.md
│   │
│   └── README.md
│
└── README.md
```

*The structure above illustrates the intended organization. Adjust folder and script names if your actual repository uses different names.*

---

## Progress Tracker

### Month 1 — Week 1

- [x] Pinhole camera model
- [x] 3D-to-2D projection
- [x] Camera orbit and control
- [x] Lens distortion
- [x] Chessboard corner detection
- [x] Initial camera calibration
- [x] Calibration testing

### Month 1 — Week 2

- [x] Full smartphone camera calibration pipeline
- [x] Intrinsic matrix estimation
- [x] Lens distortion coefficient estimation
- [x] Per-image reprojection error analysis
- [x] Identification of high-error calibration images
- [x] Calibration refinement using 19 selected images
- [x] Before-and-after calibration metric comparison
- [x] Image undistortion tests
- [x] Saving calibration parameters for reuse

---

## Key Takeaways

Through these projects, I have gained practical experience with:

- The mathematical foundations of camera projection
- Camera pose and coordinate transformations
- Radial and tangential lens distortion
- Chessboard-based camera calibration
- Sub-pixel corner detection
- Intrinsic camera matrix estimation
- Reprojection error calculation and interpretation
- Calibration dataset quality assessment
- Image undistortion and visual validation
- Reusable calibration parameter files

The next step is to build on these foundations and explore more advanced 3D computer vision techniques, including pose estimation, photogrammetry, and neural scene reconstruction.