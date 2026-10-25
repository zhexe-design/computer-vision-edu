# 3D Computer Vision Learning Projects

A practical learning repository focused on the foundations of **3D Computer Vision, camera geometry, and image processing**.

The projects are organized by weeks and days, documenting the implementation process, experiments, results, and lessons learned.

**Current progress: Month 1 — Weeks 1–3**

Topics covered:
- Pinhole camera model and 3D-to-2D projection
- Camera orbit and control
- Lens distortion
- Chessboard corner detection
- Smartphone camera calibration
- Reprojection error analysis
- Image undistortion
- Camera calibration quality improvement
- 3D point cloud visualization with Open3D
- Voxel downsampling
- Statistical and Radius Outlier Removal
- Point cloud filtering parameter studies
- Surface normal estimation
- Region of Interest (ROI) cropping
- Point cloud cleaning pipelines

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

### Week 3 — Point Cloud Processing with Open3D

Week 3 introduces 3D point cloud processing, focusing on noise removal, parameter evaluation, surface normal estimation, and ROI cropping. The experiments use the Stanford Bunny point cloud to develop a preprocessing workflow for subsequent 3D computer vision tasks.

| Project | Description | Key Result |
|---|---|---|
| [Day 1 — Point Cloud Basics](./week3/day1) | Loading, inspecting, and visualizing 3D point clouds with Open3D | Stanford Bunny point cloud visualization |
| [Day 2 — Point Cloud Filtering](./week3/day2) | Parameter study comparing voxel downsampling, Statistical Outlier Removal, and Radius Outlier Removal | Systematic comparison of filtering configurations to identify a suitable preprocessing strategy |
| [Day 3 — Normals and ROI Pipeline](./week3/day3) | Combining downsampling, outlier removal, normal estimation, and ROI cropping | Processed point cloud reduced from 30,000 to 11,869 points |

**Key result:** Day 2 investigates how filtering parameters affect noise removal and geometry preservation. Day 3 applies the selected filtering configuration in a combined cleaning pipeline, followed by normal estimation and ROI cropping.

**Day 3 point count summary:**

| Processing Stage | Points |
|---|---:|
| Original point cloud | 30,000 |
| After voxel downsampling | 20,893 |
| After statistical filtering | 20,260 |
| After radius filtering | 20,180 |
| After ROI cropping | 11,869 |

The final ROI contains approximately 39.56% of the original point count. This reduction includes downsampling, outlier removal, and cropping; it does not represent noise removal alone.

[View the Week 3 point cloud processing projects](./week3/).

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

This section presents selected visual results from the camera geometry and calibration experiments developed throughout the repository.

The examples demonstrate the practical application of camera projection, chessboard-based calibration, lens distortion correction, and camera pose visualization using real images and OpenCV.

---

### Week 1 — Camera Models and Camera Calibration

Week 1 focuses on the fundamentals of camera geometry, including 3D-to-2D projection, camera control, lens distortion, and chessboard-based camera calibration.

#### Day 1 — Pinhole Camera Model

Demonstrates the projection of 3D points onto a 2D image plane using a pinhole camera model.

**Key concepts:**
- Perspective projection.
- Mapping 3D coordinates to 2D image coordinates.
- Camera intrinsic parameters.
- Visualization of a projected 3D cube and coordinate axes.


![Pinhole camera projection](./week1_camera_models/day1_pinhole/output/result.png)

---

#### Day 2 — Camera Orbit Control

Demonstrates camera movement around a 3D object using orbit control and a look-at transformation.

**Key concepts:**
- Camera position and orientation.
- Look-at transformations.
- Circular camera movement around a target.
- Visualization of a 3D scene from different viewpoints.


![Camera orbit animation](./week1_camera_models/day2_camera_control/output/orbit.gif)

---

#### Day 3 — Lens Distortion

Explores radial lens distortion and its effect on image geometry.

The experiment demonstrates how lens distortion changes the apparent positions of image points and how distortion correction can improve geometric consistency.

**Key concepts:**
- Radial lens distortion.
- Distortion parameters.
- Image correction.
- Comparison of distorted and corrected images.


---

#### Day 4 — Chessboard Corner Detection

Focuses on detecting internal chessboard corners in images, providing image points for camera calibration.

The detected corners can be refined to sub-pixel accuracy before being used in the calibration process.

**Key concepts:**
- Chessboard pattern detection.
- Internal corner localization.
- Sub-pixel corner refinement.
- Preparation of 2D image points for calibration.

**Result:**

<!-- Add your Day 4 chessboard detection image here -->

![Detected chessboard corners](./week1_camera_models/day4_find_corners/results/example_corners.jpg)

---

#### Day 5 — Camera Calibration

Implements a complete camera calibration pipeline using OpenCV, NumPy, and a chessboard pattern.

The script detects chessboard corners, estimates the camera's intrinsic matrix and lens distortion coefficients, evaluates calibration accuracy, and generates visualizations of the results.

Calibration was performed using chessboard images with different angles and positions.

**Calibration settings:**
- Chessboard inner corners: `9 × 6`
- Square size: `0.025 m` (2.5 cm)
- Calibration method: `cv2.calibrateCamera()`
- Corner refinement: `cv2.cornerSubPix()`
- Image undistortion: `cv2.undistort()`

**Mean reprojection error: approximately 0.18–0.26 pixels**, depending on the input images and calibration results.

##### 1. Chessboard Corner Detection

The calibration process begins by identifying the internal corners of the chessboard.

These image points are matched with known 3D coordinates on the chessboard plane. Their correspondences provide the geometric information required to estimate the camera parameters.

**Result:**

![Day 5 — Detected chessboard corners](./week1_camera_models/day5_calibration/results/corners_img01.jpg)


##### 2. Original vs. Undistorted Image

After estimating the camera matrix and distortion coefficients, the script applies lens distortion correction.

The comparison illustrates the difference between the original image and its undistorted version.

**Result:**


![Day 5 — Original vs. undistorted image](./week1_camera_models/day5_calibration/results/comparison_1.jpg)

*Comparison of the original image and the image after lens distortion correction using the estimated camera parameters.*

##### 3. 3D Coordinate Axes Visualization

The estimated camera parameters and chessboard pose can be used to project 3D coordinate axes onto the image.

This visualization connects the calibration results with camera pose estimation and 3D-to-2D projection.

**Result:**


![Day 5 — 3D coordinate axes](./week1_camera_models/day5_calibration/results/axes_visualization.jpg)

*3D coordinate axes projected onto the chessboard using the estimated camera parameters and pose.*

##### What Day 5 Demonstrates

- Chessboard corner detection and sub-pixel refinement.
- Estimation of the camera intrinsic matrix `K`.
- Estimation of radial and tangential distortion coefficients.
- Camera calibration using `cv2.calibrateCamera()`.
- Calculation of mean and per-image reprojection errors.
- Lens distortion correction using `cv2.undistort()`.
- Camera pose visualization using `cv2.drawFrameAxes()`.
- Saving calibration parameters for reuse.

The saved parameters can be reused in subsequent experiments without repeating the entire calibration process.

[View the complete Day 5 implementation and instructions](./week1_camera_models/day5_calibration).

---

#### Day 6 — Calibration Testing

Extends the initial calibration work with additional testing and verification.

**Key concepts:**
- Testing estimated camera parameters.
- Examining calibration behavior on additional images.
- Validating the calibration workflow.


---

### Week 2 — Smartphone Camera Calibration and Quality Analysis

Week 2 builds on the camera geometry concepts from Week 1 and focuses on a more complete calibration workflow using real smartphone photographs.

The experiments investigate calibration accuracy, image-level reprojection errors, and the effect of removing high-error images from the calibration dataset.

#### Day 1 — Full Camera Calibration

The calibration pipeline estimates the intrinsic camera matrix and distortion coefficients from smartphone photographs of a chessboard.

The resulting parameters are then used to generate undistorted images and calibration visualizations.

**Key outputs:**
- Camera intrinsic matrix.
- Lens distortion coefficients.
- Chessboard corner visualizations.
- Original-versus-undistorted image comparisons.
- Saved calibration parameters.

**Result:**

[View the Week 2 Day 1 implementation](./week2_camera_calibration/day1).

#### Day 2 — Calibration Quality Analysis

Analyzes the reprojection error of individual calibration images to identify observations that contribute disproportionately to the overall error.

Four high-error images were excluded from the initial dataset, reducing the mean reprojection error from `1.43 px` to `0.96 px`.

| Metric | Initial Calibration | Refined Calibration |
|---|---:|---:|
| Calibration images | 23 | 19 |
| Mean reprojection error | 1.43 px | 0.96 px |
| RMS calibration error | 1.97 px | 1.32 px |
| Images removed | 0 | 4 |

**Mean reprojection error improvement: 32.94%.**

The results demonstrate how examining individual image errors can help refine a calibration dataset.

However, a lower reprojection error does not automatically guarantee better undistortion for every photograph. Visual validation and testing on independent images remain important.

**Result:**


[View the Week 2 calibration quality analysis](./week2_camera_calibration/day2).

### Week 3 — Point Cloud Cleaning Pipeline

Week 3 extends the repository from camera calibration to 3D point cloud processing using Open3D.

The main result is a three-stage progression from point cloud visualization to systematic filter evaluation and a combined preprocessing pipeline.

**Day 2 — Parameter Study**

Different configurations of Statistical Outlier Removal and Radius Outlier Removal are evaluated to understand their effects on point density, outlier removal, and geometry preservation.

The goal is to identify a suitable filtering configuration rather than simply remove the largest possible number of points.

**Day 3 — Normals and ROI Pipeline**

The selected filtering configuration is reused in a pipeline combining voxel downsampling, outlier removal, surface normal estimation, and ROI cropping.

| Metric | Result |
|---|---:|
| Original points | 30,000 |
| After voxel downsampling | 20,893 |
| After statistical filtering | 20,260 |
| After radius filtering | 20,180 |
| After ROI cropping | 11,869 |
| Total point count reduction | 60.44% |

**Key takeaway:** Point cloud preprocessing is not just about reducing the number of points. Parameter selection must balance noise removal, geometry preservation, and the requirements of downstream operations.

The resulting pipeline provides a foundation for point cloud registration, Point-to-Plane ICP, mesh reconstruction, and other 3D geometry processing tasks.

---

### Summary of Visual Results

The visual examples illustrate the progression from basic camera geometry to practical camera calibration and quality assessment.

| Experiment | Main Result |
|---|---|
| Day 1 — Pinhole Camera | 3D-to-2D projection |
| Day 2 — Camera Control | Camera orbit animation |
| Day 3 — Lens Distortion | Distortion correction |
| Day 4 — Chessboard Detection | Detected internal corners |
| Day 5 — Camera Calibration | Camera parameters, reprojection error, and 3D axes visualization |
| Week 2 — Full Calibration | Smartphone calibration and undistortion |
| Week 2 — Quality Analysis | Reduced mean reprojection error after dataset refinement |

### Week 3 — Point Cloud Processing

Week 3 focuses on preparing 3D point clouds for downstream geometry processing. The experiments progress from basic visualization to filtering parameter evaluation and a complete cleaning pipeline.

#### Day 1 — Point Cloud Basics

Introduces 3D point cloud representation and visualization using Open3D.

**Key concepts:**
- XYZ coordinates and 3D geometry.
- Loading and rendering point clouds.
- Inspecting point cloud structure.

[View Day 1 — Point Cloud Basics](./week3/day1).

#### Day 2 — Point Cloud Filtering

Compares voxel downsampling, Statistical Outlier Removal, and Radius Outlier Removal using the Stanford Bunny dataset with 3% synthetic noise.

**Key concepts:**
- Point cloud density reduction.
- Statistical and neighborhood-based outlier detection.
- Filtering parameter comparison.
- Balancing noise removal and geometry preservation.

The main outcome is a parameter study designed to identify a suitable filtering configuration for the subsequent pipeline.

[View Day 2 — Point Cloud Filtering](./week3/day2).

#### Day 3 — Normals Estimation and ROI Cropping

Combines the selected filtering configuration with normal estimation and ROI cropping.

**Key results:**
- Original point cloud: 30,000 points.
- After voxel downsampling: 20,893 points.
- After statistical filtering: 20,260 points.
- After radius filtering: 20,180 points.
- After ROI cropping: 11,869 points.

**Pipeline comparison**

![Week 3 Day 3 — Point Cloud Cleaning Pipeline](./week3/day3/results/05_pipeline_comparison.png)

**Surface normals**

![Week 3 Day 3 — Surface Normals](./week3/day3/results/with_normals/03_normals.png)

**ROI cropping**

![Week 3 Day 3 — Region of Interest](./week3/day3/results/cropped/04_roi.png)

[View Day 3 — Normals and ROI Pipeline](./week3/day3).

---

### Week 3 Summary

| Day | Focus | Main Outcome |
|---|---|---|
| Day 1 | Point cloud basics | 3D point cloud visualization |
| Day 2 | Filtering parameter study | Comparison of filtering configurations |
| Day 3 | Normals and ROI pipeline | Integrated preprocessing pipeline with 11,869 points remaining after cropping |

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
├── README.md
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
└── week3/
    ├── README.md
    │
    ├── day1/
    │   ├── main.py
    │   ├── results/
    │   │   ├── before.png
    │   │   └── after.png
    │   └── README.md
    │
    ├── day2/
    │   ├── main.py
    │   ├── results/
    │   │   ├── experiment_summary.txt
    │   │   └── ...
    │   └── README.md
    │
    └── day3/
        ├── day3_normals_and_roi.py
        ├── results/
        │   ├── cleaned/
        │   ├── with_normals/
        │   ├── cropped/
        │   ├── 05_pipeline_comparison.png
        │   └── day3_summary.txt
        └── README.md

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