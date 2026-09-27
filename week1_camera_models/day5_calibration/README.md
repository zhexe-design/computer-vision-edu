# Day 5 - Camera Calibration

## Goal

The goal of this project is to learn how to calibrate a camera using a chessboard pattern and understand the results of the calibration.

After this project, I can:

* Use a set of chessboard calibration images
* Detect chessboard corners
* Refine the detected corner positions
* Use `cv2.calibrateCamera()`
* Get the camera matrix `K`
* Get distortion coefficients
* Calculate the mean reprojection error
* Use camera parameters to undistort images
* Visualize the camera coordinate axes

Camera calibration is an important step in many Computer Vision and 3D applications.

## Why Camera Calibration Is Needed

A real camera has some distortion, so the image is not always a perfect representation of the real world.

Camera calibration helps us find the camera parameters that describe how 3D points are projected onto a 2D image.

Calibration is used in many areas, for example:

* Structure-from-Motion and COLMAP
* 3D reconstruction
* Augmented Reality
* Robotics and SLAM
* Measurements from images
* Photogrammetry
* Scan-to-BIM

If the camera parameters are not accurate, later 3D calculations can also contain errors.

## Project Structure

```text
day5_calibration/
│
├── calibration_images/
│   ├── image1.jpg
│   ├── image2.jpg
│   ├── image3.jpg
│   └── ...
│
├── results/
│   ├── corners_image1.jpg
│   ├── corners_image2.jpg
│   ├── camera_calibration.npz
│   ├── axes_visualization.jpg
│   ├── comparison_1.jpg
│   ├── comparison_2.jpg
│   └── comparison_3.jpg
│
└── calibration.py
```

## Requirements

Python 3 is required.

Install the libraries:

```bash
pip install opencv-python numpy
```

## Calibration Board

The project uses a chessboard with:

```python
CHESSBOARD_SIZE = (9, 6)
```

This means the program looks for 9 by 6 inner corners.

The size of one chessboard square is:

```python
SQUARE_SIZE = 0.025
```

The size is given in meters, so one square is 0.025 m.

## How the Program Works

### 1. Find Calibration Images

The program searches for `.jpg`, `.jpeg`, and `.png` files inside the calibration images folder.

```python
images = (
    glob.glob(os.path.join(CALIBRATION_FOLDER, "*.jpg"))
    + glob.glob(os.path.join(CALIBRATION_FOLDER, "*.jpeg"))
    + glob.glob(os.path.join(CALIBRATION_FOLDER, "*.png"))
)
```

If no images are found, the program stops.

### 2. Create 3D Points

The program creates the real 3D coordinates of the chessboard corners.

The chessboard is considered to be on the same plane, so the Z coordinate is 0.

These points are stored in `objpoints`.

### 3. Detect Chessboard Corners

For every image, OpenCV tries to find the chessboard corners using:

```python
cv2.findChessboardCorners()
```

If the corners are found, their positions are refined using:

```python
cv2.cornerSubPix()
```

The 3D chessboard points and detected 2D image points are then saved.

Images with detected corners are also saved in the `results` folder.

### 4. Camera Calibration

After processing all images, the program uses:

```python
cv2.calibrateCamera()
```

It returns:

```python
ret, K, dist, rvecs, tvecs
```

These values contain the main camera calibration results.

## Calibration Results

### Camera Matrix K

The camera matrix contains the main intrinsic parameters of the camera.

It has this form:

```text
[ fx   0   cx ]
[  0  fy   cy ]
[  0   0    1 ]
```

Where:

* `fx` - focal length in the X direction
* `fy` - focal length in the Y direction
* `cx` - principal point X coordinate
* `cy` - principal point Y coordinate

The program prints the matrix:

```python
print(K)
```

### Distortion Coefficients

The program also calculates lens distortion coefficients:

```python
dist
```

They describe how much the image is distorted by the camera lens.

They are used by OpenCV when correcting the image.

The program prints them with:

```python
print(dist)
```

### Reprojection Error

The program projects the 3D chessboard points back onto the image using the calculated camera parameters.

The difference between the original detected points and the projected points is the reprojection error.

The program calculates the error for every calibration image and then calculates the mean error:

```python
mean_error = total_error / len(objpoints)
```

The final value is printed in pixels: 0.262884 pixels

```text
Mean reprojection error: 0.18 pixels
```

A lower reprojection error means that the calculated camera parameters fit the detected calibration points more closely. The error should still be interpreted together with the calibration images and detection quality.

## Undistortion

After calibration, the camera matrix and distortion coefficients can be used to correct images.

The project uses:

```python
cv2.undistort()
```

The program creates comparison images:

```text
Original | Undistorted
```

The first three calibration images are used as examples.

The results are saved as:

```text
comparison_1.jpg
comparison_2.jpg
comparison_3.jpg
```

## Coordinate Axes

The project also visualizes 3D coordinate axes on one of the calibration images.

It uses:

```python
cv2.drawFrameAxes()
```

The result is saved as:

```text
axes_visualization.jpg
```

This helps visualize the camera pose relative to the chessboard.

## Saved Calibration Parameters

The main calibration parameters are saved to:

```text
camera_calibration.npz
```

The file contains:

* `K` - camera matrix
* `dist` - distortion coefficients
* `image_width` - image width
* `image_height` - image height

These parameters can be loaded later without running the calibration again.

For example:

```python
data = np.load("camera_calibration.npz")

K = data["K"]
dist = data["dist"]
```

## Final Output

At the end of the program, the main results are printed:

```text
================================
FINAL CALIBRATION PARAMETERS
================================

1. Mean reprojection error:
0.18 pixels

2. Camera matrix K:
[[... ... ...]
 [... ... ...]
 [... ... ...]]

3. Distortion coefficients:
[[... ... ...]]
```

The exact values depend on the camera and the calibration images.

## Results Folder

After running the program, the `results` folder can contain:

* `camera_calibration.npz` - saved camera parameters
* `corners_*.jpg` - images with detected chessboard corners
* `axes_visualization.jpg` - coordinate axes on a calibration image
* `comparison_*.jpg` - original and undistorted images

## How to Run

Put the chessboard images into:

```text
calibration_images/
```

Then run:

python calibration.py
```

The program will detect the chessboard corners, calibrate the camera, calculate the reprojection error, save the calibration parameters, and create visualization results.

## What I Learned

In this project I learned how to:

* Prepare chessboard points for camera calibration
* Detect and refine chessboard corners with OpenCV
* Calibrate a camera with `cv2.calibrateCamera()`
* Understand the camera matrix `K`
* Understand distortion coefficients
* Calculate reprojection error
* Save calibration parameters
* Undistort images
* Visualize 3D coordinate axes on an image

This is a basic camera calibration pipeline that can be used as a starting point for more advanced 3D Computer Vision tasks.
