# Day 1 — Full Camera Calibration

## Camera

Real smartphone camera.

The calibration was done using real photos taken with a smartphone camera.

## Chessboard

* Inner corners: 9 × 6
* Square size: 0.025 m
* Number of calibration images: 23

The same chessboard was photographed from different angles and distances.

The chessboard pattern was used to find known points in the real world and match them with their positions in the camera images.

## Calibration Process

The calibration script performs the following steps:

1. Loads all chessboard images from the `calibration_images` folder.
2. Converts each image to grayscale.
3. Detects the chessboard corners using `findChessboardCorners`.
4. Refines the detected corners using `cornerSubPix`.
5. Creates 3D object points for the chessboard.
6. Collects the detected 2D image points.
7. Uses `cv2.calibrateCamera` to calculate the camera parameters.
8. Calculates the reprojection error for each calibration image.
9. Saves the camera parameters to a YAML file.
10. Uses the calibration parameters to remove lens distortion from test images.

## Results

* Calibration images used: **23**
* Mean reprojection error: **1.425803 pixels**
* Intrinsic matrix `K` calculated and saved
* Distortion coefficients calculated and saved
* Camera parameters saved to `camera_params.yaml`

The mean reprojection error shows the average distance between the detected chessboard corners and the points projected using the calculated camera parameters.

A lower error means that the detected points are closer to the points predicted by the calibration model.

## Reprojection Error

The reprojection error was also calculated separately for every calibration image.

Some images had a lower error, while some images had a higher error because the chessboard was photographed from different angles and distances.

This helped check how consistently the calibration worked across all images.

## Capture Conditions

* Real smartphone photos
* Indoor lighting
* No studio setup
* Different camera angles
* Different distances from the chessboard
* 23 calibration images

## Undistortion

After calibration, the calculated camera parameters were used to remove lens distortion.

Several normal photos were used as test images. The script creates a Before/After comparison for each image.

The original image is shown next to the corrected image so it is easier to see the effect of the calibration.

Test images are stored in:

`test_images/`

The generated comparisons are stored in:

`results/before_after/`

## Corner Detection Results

The detected chessboard corners are saved as preview images.

These images are stored in:

`results/corners/`

They can be used to check if the chessboard was detected correctly on the calibration images.

## Axes Visualization

A coordinate system was also drawn on a chessboard image using the calibration parameters.

The result is saved as:

`results/axes_visualization.jpg`

This shows the camera coordinate axes projected onto the chessboard.

## Output Files

The project produces the following files and folders:

```text
week2_camera_calibration/
├── calibration_images/
├── test_images/
├── results/
│   ├── corners/
│   ├── before_after/
│   └── axes_visualization.jpg
├── day1_full_calibration.py
├── camera_params.yaml
└── README.md
```

### Main output

* `camera_params.yaml` — saved camera matrix and distortion coefficients
* `results/corners/` — chessboard corner detection results
* `results/before_after/` — original and undistorted image comparisons
* `results/axes_visualization.jpg` — 3D coordinate axes visualization

## Main Parameters

The calibration uses the following chessboard parameters:

```text
Inner corners: 9 × 6
Square size: 0.025 m
```

The main camera parameters calculated during calibration are:

* Camera matrix `K`
* Distortion coefficients `dist`
* Rotation vectors
* Translation vectors

The camera matrix and distortion coefficients are saved so they can be reused later without running the calibration again.

## Conclusion

The main goal of this task was to perform a complete camera calibration using real smartphone images.

The final script can:

* detect chessboard corners;
* calculate camera parameters;
* calculate reprojection error;
* save the calibration parameters;
* detect lens distortion;
* undistort normal photos;
* create Before/After comparisons;
* visualize 3D coordinate axes.

The final mean reprojection error was **1.425803 pixels**.
