# Camera Calibration (Smartphone)

## Overview

This project implements a complete camera calibration pipeline using a chessboard pattern, Python, OpenCV, and NumPy.

The calibration was performed using real smartphone photographs captured under uncontrolled indoor lighting. The project covers chessboard corner detection, camera parameter estimation, reprojection error analysis, image undistortion, and 3D coordinate axes visualization.

To improve calibration quality, the initial calibration results were analyzed on a per-image basis. Four images with the highest reprojection errors were removed before performing the final calibration.

## Camera and Calibration Setup

### Camera

| Parameter | Value |
|---|---|
| Device | Smartphone |
| Camera model | Not specified |
| Image resolution | 2528 × 1284 |
| Capture environment | Indoor lighting, no studio setup |

### Chessboard

| Parameter | Value |
|---|---|
| Internal corners | 9 × 6 |
| Square size | 0.025 m (2.5 cm) |
| Calibration pattern | Planar chessboard |

The chessboard dimensions refer to the number of internal corners, not the number of squares.

## Calibration Results

The calibration was performed in two stages. First, all 23 calibration images were used. The per-image reprojection errors were then analyzed, and the four images with the highest errors were excluded from the final calibration.

| Metric | Initial Calibration | Final Calibration |
|---|---:|---:|
| Number of images | 23 | 19 |
| Mean reprojection error | 1.43 px | 0.96 px |
| Images removed | 0 | 4 |

**Mean reprojection error improvement: 32.94%.**

The mean reprojection error decreased by 0.47 pixels after filtering the calibration images.

This indicates an improvement in the reprojection accuracy for this dataset. However, reprojection error alone does not guarantee that the estimated camera parameters will produce optimal undistortion for every photograph.

## Camera Intrinsic Matrix

The final intrinsic matrix `K` describes the camera's focal lengths in pixel units and the principal point.

```text
K =
[[1783.644,    0.000, 1264.213],
 [   0.000, 1775.917,  623.787],
 [   0.000,    0.000,    1.000]]
```

The matrix contains:

- `fx = 1783.644` — focal length along the horizontal image axis, in pixels.
- `fy = 1775.917` — focal length along the vertical image axis, in pixels.
- `cx = 1264.213` — horizontal coordinate of the principal point.
- `cy = 623.787` — vertical coordinate of the principal point.

The focal lengths are estimated independently, so `fx` and `fy` do not have to be identical.

## Distortion Coefficients

The estimated distortion coefficients follow OpenCV's standard ordering:

```text
dist = [k1, k2, p1, p2, k3]
```

The final values are:

```text
[0.168043, -0.929010, -0.028054, -0.004844, 0.956781]
```

Where:

- `k1`, `k2`, `k3` — radial distortion coefficients.
- `p1`, `p2` — tangential distortion coefficients.

These coefficients are used together with the intrinsic matrix to model lens distortion and undistort images.

## Capture Conditions

The calibration images were captured under real-world conditions rather than in a controlled laboratory environment.

- Real smartphone photographs.
- Indoor lighting without a dedicated studio setup.
- Different chessboard angles, distances, and positions.
- Potential motion blur and variations in image sharpness.
- A physical chessboard with a square size of 2.5 cm.

Using varied viewpoints helps constrain the camera parameters more effectively than capturing many nearly identical images.

## Calibration Pipeline

The project follows these steps:

1. **Detect chessboard corners** using `cv2.findChessboardCorners()`.
2. **Refine corner positions** to sub-pixel accuracy using `cv2.cornerSubPix()`.
3. **Perform initial calibration** with `cv2.calibrateCamera()`.
4. **Calculate reprojection errors** for individual calibration images.
5. **Identify problematic frames** with the highest reprojection errors.
6. **Remove four high-error images** from the calibration dataset.
7. **Perform final calibration** using the remaining 19 images.
8. **Undistort test images** using the estimated camera matrix and distortion coefficients.
9. **Visualize 3D coordinate axes** on the calibration board using `cv2.drawFrameAxes()`.
10. **Save the final calibration parameters** for later use.

## Results and Visualizations

The project generates visual outputs for evaluating the calibration process.

### Before and After Undistortion

Original and undistorted images are compared to visualize the effect of the estimated distortion parameters.

### Chessboard Corner Detection

Detected chessboard corners are visualized to check whether the calibration pattern was identified correctly.

### Coordinate Axes Visualization

The 3D coordinate axes are projected onto a calibration image using the estimated camera parameters and board pose.

### Calibration Quality Comparison

The initial and final mean reprojection errors are compared to evaluate the effect of removing high-error images.

### Saved Calibration Parameters

The final camera parameters are saved in:

```text
camera_params.yaml
```

The file stores the calibration data required to reuse the estimated camera parameters without repeating the entire calibration process.

## How to Run

### 1. Install Dependencies

Install the required Python packages:

```bash
pip install opencv-python numpy pyyaml
```

### 2. Run the Calibration Script

From the appropriate project directory, run:

```bash
python apply_calibration.py
```

Make sure the calibration images and any required configuration files are located in the directories expected by the script.

## Technologies Used

- **Python** — implementation of the calibration pipeline.
- **OpenCV** — chessboard detection, camera calibration, undistortion, and coordinate axes visualization.
- **NumPy** — numerical operations and calibration data processing.
- **PyYAML** — reading and writing YAML configuration files.

## What I Learned

- How to perform camera calibration using OpenCV and a chessboard pattern.
- How to detect and refine chessboard corners to sub-pixel accuracy.
- How to estimate a camera intrinsic matrix and lens distortion coefficients.
- How to calculate and interpret reprojection errors.
- How to identify high-error calibration images and evaluate their impact on calibration results.
- How to undistort images using estimated camera parameters.
- How to visualize 3D coordinate axes on a calibration image.
- How to save and reuse calibration parameters.

## Conclusion

This project demonstrates a complete smartphone camera calibration workflow using real photographs captured under uncontrolled indoor conditions.

By analyzing per-image reprojection errors and removing four high-error images, the mean reprojection error decreased from **1.43 px to 0.96 px**, an improvement of approximately **32.94%**.

The final intrinsic matrix and distortion coefficients were saved for reuse in subsequent computer vision tasks, including image undistortion, pose estimation, and 3D reconstruction.