Camera Calibration (Smartphone)
Overview

Full camera calibration pipeline using a chessboard pattern and OpenCV.

Performed on a real smartphone under uncontrolled indoor lighting.

Camera
Model: Not specified
Resolution used: 2528 × 1284
Chessboard
Inner corners: 9 × 6
Square size: 0.025 m
Calibration Results

Average reprojection error: 0.96 pixels

Initial calibration used 23 images with an average reprojection error of 1.43 pixels. After removing 4 images with the highest reprojection error, the final calibration used 19 images and achieved an average error of 0.96 pixels.

Intrinsic Matrix K
[[1783.644,    0.000, 1264.213],
 [   0.000, 1775.917,  623.787],
 [   0.000,    0.000,    1.000]]
Distortion Coefficients
[0.168043, -0.929010, -0.028054, -0.004844, 0.956781]
Capture Conditions
Real smartphone photos
Indoor lighting, no studio setup
Different angles, distances and positions of the chessboard
Natural motion blur possible
Pipeline
Detect chessboard corners (findChessboardCorners + cornerSubPix)
Perform initial camera calibration (cv2.calibrateCamera)
Compute reprojection error for each image
Find images with the highest reprojection error
Remove 4 images with the highest error
Perform the final camera calibration
Undistort test images
Visualize camera axes with drawFrameAxes
Save final camera parameters
Results
Before / After undistortion
Chessboard corner detection examples
Axes visualization on the calibration board
Initial and final reprojection error comparison
Saved camera parameters (camera_params.yaml)
How to run
pip install opencv-python numpy pyyaml
python day3_full_calibration.py
What I learned
How to perform full camera calibration with OpenCV
How to calculate and evaluate reprojection error
How removing high-error images can improve calibration
How to undistort images using camera parameters
How to visualize camera axes on a chessboard
How to save and reuse camera calibration parameters  