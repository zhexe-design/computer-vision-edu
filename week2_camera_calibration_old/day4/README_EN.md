# Camera Calibration (Smartphone)

## Overview

A complete smartphone camera calibration pipeline implemented with OpenCV and a chessboard pattern.

The project covers camera calibration, distortion analysis, image undistortion, reprojection error evaluation, and 3D axes visualization. The calibration was performed using real smartphone photos captured under uncontrolled indoor conditions.

## Key Result

**Average reprojection error: 1.425803 pixels**

The calibration parameters were saved and used to test distortion correction on images outside the calibration dataset.

## Camera

* **Model:** [Smartphone model]
* **Calibration image resolution:** [Width × Height]
* **Environment:** Indoor, natural lighting

## Chessboard Configuration

* **Inner corners:** 9 × 6
* **Square size:** 0.025 m
* **Calibration images:** 23

## Capture Conditions

* Real smartphone photographs
* Indoor lighting without a controlled studio setup
* Different camera angles, distances, and positions
* Potential motion blur and variations in image quality

## Pipeline

1. Chessboard corner detection and sub-pixel refinement
2. Camera intrinsic parameter estimation
3. Distortion coefficient calculation
4. Per-image and average reprojection error evaluation
5. Image undistortion and visual comparison
6. 3D coordinate axes visualization
7. Saving camera parameters for reuse
8. Testing distortion correction on new images

## Challenges and Improvements

### 1. Image Distortion After Undistortion

One of the main challenges was that some test images appeared more distorted after applying the undistortion process. Although the chessboard images looked relatively normal after correction, ordinary photographs showed unexpected warping.

This indicated that the calibration quality alone was not enough to guarantee satisfactory results on every image. The calibration parameters, image resolution, and undistortion settings also needed to be checked.

### 2. Adjusting Calibration Weights

I experimented with different `weights` values in an attempt to improve the calibration and reduce the visible distortion in the corrected images.

The goal was to find settings that produced more natural-looking results while preserving straight lines and reducing unwanted warping. I compared the outputs visually and investigated how changes in the calibration settings affected the final images.

**Note:** The final effect of these adjustments must be evaluated using the same test images and consistent comparison settings.

### 3. Improving Script Execution

I also improved the script so that it could be launched directly without manually navigating to its directory or editing absolute file paths.

The script was configured to resolve file paths relative to its own location. This made it easier to load calibration parameters, access input images, and save results regardless of the current working directory.

The script also supports creating the output directory when it does not already exist, making the workflow more convenient and reproducible.

## Results and Outputs

The project generates or stores the following outputs:

* Camera intrinsic matrix `K`
* Lens distortion coefficients
* Reprojection error measurements
* Original and undistorted image comparisons
* Difference images for visual analysis
* 3D coordinate axes visualization
* Saved camera calibration parameters (`.yaml` or `.npz`)

## Technologies

* Python
* OpenCV
* NumPy
* PyYAML

## Future Improvements

* Evaluate calibration quality using a wider variety of test images
* Investigate the causes of unexpected image warping
* Compare alternative undistortion settings
* Improve calibration robustness by reviewing images with high reprojection errors
* Document the effect of different calibration settings on the final results
