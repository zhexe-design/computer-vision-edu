# 3D Computer Vision Learning Projects

A practical learning repository focused on the foundations of **3D Computer Vision, camera geometry, image processing, and point cloud processing**.

The projects are organized by weeks and days, documenting implementations, experiments, results, and lessons learned.

**Current progress: Month 1 — Weeks 1–3**

## Topics Covered

- Pinhole camera model and 3D-to-2D projection
- Camera orbit and control
- Radial and tangential lens distortion
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

The point cloud exercises complement the camera geometry projects by introducing 3D geometry inspection and preprocessing.

---

## Projects Overview

### Week 1 — Camera Models

| Project | Description | Key Result |
|---|---|---|
| [Day 1 — Pinhole Camera](./week1_camera_models/day1_pinhole/) | Pinhole camera model and 3D-to-2D projection | Projected 3D cube and coordinate axes |
| [Day 2 — Camera Control](./week1_camera_models/day2_camera_control/) | Camera orbit and look-at transformation | Camera orbit animation |
| [Day 3 — Lens Distortion](./week1_camera_models/day3_distortion/) | Radial lens distortion and image correction | Distorted and corrected image comparison |
| [Day 4 — Chessboard Detection](./week1_camera_models/day4_find_corners/) | Chessboard corner detection | Detected internal corners |
| [Day 5 — Camera Calibration](./week1_camera_models/day5_calibration/) | Camera calibration using a chessboard pattern | Initial mean reprojection error of approximately 0.18–0.26 px |
| [Day 6 — Calibration Testing](./week1_camera_models/day6_testing/) | Additional calibration tests and verification | Calibration validation experiments |

### Week 2 — Camera Calibration and Quality Analysis

Week 2 builds on the camera geometry concepts from Week 1 and focuses on a more complete smartphone camera calibration workflow.

| Project | Description | Key Result |
|---|---|---|
| [Day 1 — Full Camera Calibration](./week2_camera_calibration/quality_analysis/) | Camera calibration using smartphone photographs | Camera parameters and undistortion results |
| [Day 2 — Calibration Quality Analysis](./week2_camera_calibration/quality_analysis/) | Per-image reprojection error analysis and calibration refinement | Mean reprojection error reduced from 1.43 px to 0.96 px |

**Note:** Week 1 and Week 2 use different calibration runs or datasets. Their results should be evaluated separately rather than treated as directly comparable measurements.

### Week 3 — Point Cloud Processing with Open3D

Week 3 introduces 3D point cloud processing using the Stanford Bunny dataset. The exercises progress from visualization to filtering parameter evaluation and an integrated preprocessing pipeline.

| Project | Description | Key Result |
|---|---|---|
| [Day 1 — Point Cloud Basics](./week3_pointclouds/README.md#day-1--point-cloud-basics) | Loading, inspecting, and visualizing point clouds with Open3D | Stanford Bunny point cloud visualization |
| [Day 2 — Point Cloud Filtering](./week3_pointclouds/README.md#day-2--point-cloud-filtering-and-parameter-study) | Parameter study comparing voxel downsampling, Statistical Outlier Removal, and Radius Outlier Removal | Comparison of filtering configurations |
| [Day 3 — Normals and ROI Pipeline](./week3_pointclouds/README.md#day-3--normals-roi-and-cleaning-pipeline) | Downsampling, outlier removal, normal estimation, and ROI cropping | Point count reduced from 30,000 to 11,869 |

**Key result:** Day 2 investigates how filtering parameters affect point density and geometry preservation. Day 3 reuses the selected Statistical Outlier Removal configuration (`nb_neighbors=20`, `std_ratio=2.0`) in a combined cleaning pipeline.

#### Day 3 — Point Count Summary

| Processing Stage | Points |
|---|---:|
| Original point cloud | 30,000 |
| After voxel downsampling | 20,893 |
| After statistical filtering | 20,260 |
| After radius filtering | 20,180 |
| After ROI cropping | 11,869 |

The final ROI contains approximately **39.56% of the original point count**, representing an overall reduction of approximately **60.44%**.

This reduction includes downsampling, outlier filtering, and cropping. It should not be interpreted as the percentage of noise removed.

[View all Week 3 point cloud projects and results](./week3_pointclouds/README.md).

---

## Best Result So Far

### Week 2 — Improved Smartphone Camera Calibration

The main result of Week 2 is an improved calibration obtained by analyzing individual image reprojection errors and excluding four high-error images.

| Metric | Initial Calibration | Final Calibration |
|---|---:|---:|
| Calibration images | 23 | 19 |
| Mean reprojection error | 1.43 px | 0.96 px |
| RMS calibration error | 1.97 px | 1.32 px |
| Images removed | 0 | 4 |

**Mean reprojection error improvement: 32.94%.**

The mean reprojection error decreased by 0.47 px after removing the four images with the highest errors.

The final calibration also produced:

- An estimated camera intrinsic matrix `K`
- Radial and tangential distortion coefficients
- Original-versus-undistorted image comparisons
- Chessboard corner detection visualizations
- A 3D coordinate axes visualization
- Saved calibration parameters for reuse

A lower reprojection error does not automatically guarantee better undistortion for every photograph. Calibration quality should also be evaluated using independent test images and the intended downstream application.

### Week 3 — Point Cloud Cleaning Pipeline

Week 3 adds a second practical result: an integrated point cloud preprocessing pipeline.

The pipeline combines:

1. Voxel downsampling
2. Statistical Outlier Removal
3. Radius Outlier Removal
4. Surface normal estimation
5. ROI cropping using an axis-aligned bounding box

The selected statistical filter from Day 2 is reused in Day 3, connecting the parameter study to the implementation of a complete pipeline.

[Read the Week 3 implementation and experimental results](./week3_pointclouds/README.md).

---

## Visual Results

The following sections highlight selected results from camera geometry, camera calibration, and point cloud processing.

### Week 1 — Camera Models

#### Day 1 — Pinhole Camera Model

Demonstrates the projection of 3D points onto a 2D image plane using a pinhole camera model.

**Key concepts:**
- Perspective projection
- Mapping 3D coordinates to 2D image coordinates
- Camera intrinsic parameters
- Visualization of a projected 3D cube and coordinate axes

![Pinhole camera projection](./week1_camera_models/day1_pinhole/output/result.png)

#### Day 2 — Camera Orbit Control

Demonstrates camera movement around a 3D object using orbit control and a look-at transformation.

**Key concepts:**
- Camera position and orientation
- Look-at transformations
- Circular camera movement around a target
- Visualization of a 3D scene from different viewpoints

![Camera orbit animation](./week1_camera_models/day2_camera_control/output/orbit.gif)

#### Day 3 — Lens Distortion

Explores radial lens distortion and its effect on image geometry.

The experiment demonstrates how lens distortion changes the apparent positions of image points and how distortion correction can improve geometric consistency.

**Key concepts:**
- Radial lens distortion
- Distortion parameters
- Image correction
- Comparison of distorted and corrected images

#### Day 4 — Chessboard Corner Detection

Focuses on detecting internal chessboard corners in images, providing image points for camera calibration.

The detected corners can be refined to sub-pixel accuracy before being used in the calibration process.

**Key concepts:**
- Chessboard pattern detection
- Internal corner localization
- Sub-pixel corner refinement
- Preparation of 2D image points for calibration

**Result:**

![Detected chessboard corners](./week1_camera_models/day4_find_corners/results/example_corners.jpg)

#### Day 5 — Camera Calibration

Implements a camera calibration pipeline using OpenCV, NumPy, and a chessboard pattern.

**Calibration settings:**
- Chessboard inner corners: `9 × 6`
- Square size: `0.025 m`
- Calibration method: `cv2.calibrateCamera()`
- Corner refinement: `cv2.cornerSubPix()`
- Image undistortion: `cv2.undistort()`

**Mean reprojection error: approximately 0.18–0.26 px**, depending on the calibration run.

##### Chessboard Corner Detection

![Day 5 — Detected chessboard corners](./week1_camera_models/day5_calibration/results/corners_img01.jpg)

##### Original vs. Undistorted Image

![Day 5 — Original vs. undistorted image](./week1_camera_models/day5_calibration/results/comparison_1.jpg)

##### 3D Coordinate Axes Visualization

![Day 5 — 3D coordinate axes](./week1_camera_models/day5_calibration/results/axes_visualization.jpg)

**What Day 5 demonstrates:**
- Chessboard corner detection and sub-pixel refinement
- Estimation of the camera intrinsic matrix `K`
- Estimation of radial and tangential distortion coefficients
- Calibration using `cv2.calibrateCamera()`
- Calculation of reprojection errors
- Image undistortion using `cv2.undistort()`
- Camera pose visualization using `cv2.drawFrameAxes()`
- Saving calibration parameters for reuse

[View the Day 5 implementation](./week1_camera_models/day5_calibration/).

#### Day 6 — Calibration Testing

Extends the initial calibration work with additional testing and verification.

**Key concepts:**
- Testing estimated camera parameters
- Examining calibration behavior on additional images
- Validating the calibration workflow

---

### Week 2 — Smartphone Camera Calibration and Quality Analysis

Week 2 investigates calibration accuracy, image-level reprojection errors, and the effect of excluding high-error images from the calibration dataset.

#### Calibration Quality Analysis

Four high-error images were excluded from the initial dataset, reducing the mean reprojection error from `1.43 px` to `0.96 px`.

| Metric | Initial Calibration | Refined Calibration |
|---|---:|---:|
| Calibration images | 23 | 19 |
| Mean reprojection error | 1.43 px | 0.96 px |
| RMS calibration error | 1.97 px | 1.32 px |
| Images removed | 0 | 4 |

**Mean reprojection error improvement: 32.94%.**

The results demonstrate how examining individual image errors can help refine a calibration dataset. Visual validation and testing on independent images remain important.

[View the Week 2 quality analysis](./week2_camera_calibration/quality_analysis/).

---

### Week 3 — Point Cloud Processing

Week 3 focuses on preparing 3D point clouds for downstream geometry processing. The experiments progress from basic visualization to filtering parameter evaluation and a complete cleaning pipeline.

#### Day 1 — Point Cloud Basics

Introduces 3D point cloud representation and visualization using Open3D.

**Key concepts:**
- XYZ coordinates and 3D geometry
- Loading and rendering point clouds
- Inspecting point cloud structure

[View Day 1 — Point Cloud Basics](./week3_pointclouds/README.md#day-1--point-cloud-basics).

#### Day 2 — Point Cloud Filtering

Compares voxel downsampling, Statistical Outlier Removal, and Radius Outlier Removal using the Stanford Bunny dataset with 3% synthetic noise.

**Key concepts:**
- Point cloud density reduction
- Statistical and neighborhood-based outlier detection
- Filtering parameter comparison
- Balancing noise removal and geometry preservation

The parameter study compares conservative, balanced, and aggressive configurations. The selected Statistical Outlier Removal configuration uses `nb_neighbors=20` and `std_ratio=2.0`.

[View Day 2 — Point Cloud Filtering](./week3_pointclouds/README.md#day-2--point-cloud-filtering-and-parameter-study).

#### Day 3 — Normals Estimation and ROI Cropping

Combines the selected filtering configuration with normal estimation and ROI cropping.

| Processing Stage | Number of Points |
|---|---:|
| Original | 30,000 |
| After voxel downsampling | 20,893 |
| After statistical filtering | 20,260 |
| After radius filtering | 20,180 |
| After ROI cropping | 11,869 |

**Pipeline comparison**

![Week 3 Day 3 — Point Cloud Cleaning Pipeline](./week3_pointclouds/results/05_pipeline_comparison.png)

**Surface normals**

![Week 3 Day 3 — Surface Normals](./week3_pointclouds/results/with_normals/03_normals.png)

**ROI cropping**

![Week 3 Day 3 — Region of Interest](./week3_pointclouds/results/cropped/04_roi.png)

[View Day 3 — Normals and ROI Pipeline](./week3_pointclouds/README.md#day-3--normals-roi-and-cleaning-pipeline).

---

## Technologies Used

- **Python 3** — implementation and experimentation
- **OpenCV** — camera calibration, chessboard detection, image undistortion, and visualization
- **Open3D** — 3D point cloud loading, visualization, filtering, normal estimation, and ROI cropping
- **NumPy** — numerical operations and camera geometry
- **Matplotlib** — visualization and plotting
- **ImageIO** — image and animation processing where required
- **PyYAML** — reading and writing YAML calibration parameters

---

## How to Run

### 1. Clone the Repository

```bash
git clone https://github.com/zhexe-design/computer-vision-edu.git
cd computer-vision-edu
```

### 2. Create a Virtual Environment (Optional)

**Windows PowerShell:**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, you can run the environment's Python executable directly or use another supported activation method for your system.

### 3. Install Dependencies

From the repository root:

```bash
python -m pip install -r requirements.txt
```

If a particular project requires additional packages not listed in `requirements.txt`, install them according to that project's instructions.

### 4. Run Week 1 — Camera Models

For example, run the pinhole camera project:

```bash
cd week1_camera_models/day1_pinhole
python main.py
```

Other Week 1 projects are located in their respective directories. Check each project's README for its script name and execution instructions.

### 5. Run Week 2 — Camera Calibration

The calibration quality analysis project is located in:

```text
week2_camera_calibration/quality_analysis/
```

Run it using the actual script name in that directory. For example, if the script is named `day2_calibration_quality.py`:

```bash
cd ../../week2_camera_calibration/quality_analysis
python day2_calibration_quality.py
```

If you are already at the repository root, use:

```bash
cd week2_camera_calibration/quality_analysis
python day2_calibration_quality.py
```

### 6. Run Week 3 — Point Cloud Processing

The Week 3 scripts are grouped in `week3_pointclouds/`.

From the repository root:

```bash
cd week3_pointclouds
python day1_pointcloud_basics.py
python day2_filtering.py
python day3_normals_and_roi.py
```

The scripts may download the Stanford Bunny model through Open3D if no suitable local dataset is available and the corresponding fallback is implemented.

**Important:** Run each script from the directory expected by its implementation. Relative input and output paths are resolved according to the script's working directory.

---

## Repository Structure

```text
computer-vision-edu/
├── README.md
├── requirements.txt
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
│   ├── quality_analysis/
│   └── apply_calibration/
│
└── week3_pointclouds/
    ├── README.md
    ├── day1_pointcloud_basics.py
    ├── day2_filtering.py
    ├── day3_normals_and_roi.py
    ├── data/
    └── results/
        ├── before.png
        ├── voxel_comparison.png
        ├── experiment_summary.txt
        ├── statistical/
        ├── radius/
        ├── 01_original.png
        ├── 05_pipeline_comparison.png
        ├── day3_summary.txt
        ├── cleaned/
        ├── with_normals/
        └── cropped/
```

This tree summarizes the main project organization. Individual project directories may contain additional scripts, images, calibration files, and local README files.

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

- [x] Smartphone camera calibration workflow
- [x] Intrinsic matrix estimation
- [x] Lens distortion coefficient estimation
- [x] Per-image reprojection error analysis
- [x] Identification of high-error calibration images
- [x] Calibration refinement using 19 selected images
- [x] Before-and-after calibration metric comparison
- [x] Image undistortion tests
- [x] Saving calibration parameters for reuse

### Month 1 — Week 3

- [x] Point cloud loading and visualization
- [x] Voxel downsampling
- [x] Statistical Outlier Removal
- [x] Radius Outlier Removal
- [x] Filtering parameter study
- [x] Surface normal estimation
- [x] ROI cropping using an axis-aligned bounding box
- [x] Integrated point cloud cleaning pipeline
- [x] Exporting processed point clouds and visualizations
- [x] Recording point counts and pipeline parameters

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
- 3D point cloud inspection and visualization
- Voxel downsampling and outlier removal
- Filtering parameter evaluation and geometry preservation
- Surface normal estimation
- ROI cropping with axis-aligned bounding boxes
- Building reusable point cloud preprocessing pipelines

The next step is to explore more advanced 3D computer vision techniques, including point cloud registration, Point-to-Plane ICP, photogrammetry, mesh reconstruction, and neural scene reconstruction.