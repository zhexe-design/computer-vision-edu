# 3D Computer Vision Learning Projects

A collection of small Python projects where I learn the basics of 3D computer vision and camera geometry using OpenCV and NumPy.

Every project is a separate step. I started with simple projection of 3D points and moved on to distortion and full camera calibration. These projects are mainly for learning, so the code is simple and not perfect.

## Why I made this repository

I want to understand how a camera "sees" the world and how 3D points become 2D pixels. This is the base for more advanced topics, for example:

* Structure-from-Motion and COLMAP
* 3D reconstruction
* Photogrammetry
* Augmented Reality
* Robotics and SLAM
* Scan-to-BIM

Each project has its own `README.md` with more details, and this file is a short overview.

## Projects

| Project | What it does | Main OpenCV functions |
| ------- | ------------ | --------------------- |
| [`pinhole_projection`](./pinhole_projection) | Projects a 3D cube and coordinate axes onto a 2D image using the pinhole camera model | `cv2.projectPoints()` |
| [`day3_distortion`](./day3_distortion) | Adds artificial lens distortion to photos and then removes it | `cv2.initUndistortRectifyMap()`, `cv2.remap()`, `cv2.undistort()` |
| [`chessboard_corner_detection`](./chessboard_corner_detection) | Finds chessboard corners in photos and prepares 3D and 2D points for calibration | `cv2.findChessboardCorners()`, `cv2.cornerSubPix()`, `cv2.drawChessboardCorners()` |
| [`day5_calibration`](./day5_calibration) | Calibrates a camera with chessboard images and checks the result | `cv2.calibrateCamera()`, `cv2.undistort()`, `cv2.drawFrameAxes()` |

### 1. Pinhole Projection

I create a simple 3D scene (the vertices of a unit cube and the coordinate axes) and a virtual camera with the matrix `K`, rotation `rvec` and translation `tvec`. Then I project the points onto a white image.

The basic formula I used:

```text
u = fx * X / Z + cx
v = fy * Y / Z + cy
```

The axes are drawn in colors: X is red, Y is green, Z is blue.

### 2. Lens Distortion

Real lenses change the position of points in the image, so straight lines near the edges can look curved. In this project I:

* learned the difference between radial and tangential distortion
* made barrel and pincushion distortion on my own photos (doors, windows, books, buildings)
* removed the distortion with OpenCV
* saved a comparison image: `Original | Distorted | Undistorted`

### 3. Chessboard Corner Detection

This is the first step before calibration. The program finds the inner corners of a chessboard (9 x 6), improves their positions with `cornerSubPix()` and saves two lists:

* `objpoints` - 3D points of the chessboard
* `imgpoints` - 2D points of the same corners in the image

### 4. Camera Calibration

The full pipeline: chessboard images, corner detection, `cv2.calibrateCamera()`. After that the program:

* prints the camera matrix `K` and the distortion coefficients
* calculates the mean reprojection error (about 0.26 pixels in my run)
* saves the parameters to `camera_calibration.npz`
* creates `Original | Undistorted` comparison images
* draws the 3D coordinate axes on a calibration image

## Technologies

* Python 3
* NumPy
* OpenCV (`opencv-python`)

## Repository Structure

```text
.
├── pinhole_projection/
├── day3_distortion/
├── chessboard_corner_detection/
├── day5_calibration/
└── README.md
```

## How to Run

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPOSITORY.git
cd YOUR_REPOSITORY
```

Create a virtual environment:

```bash
python -m venv venv
```

Activate it.

On Windows:

```bash
venv\Scripts\activate
```

On Linux or macOS:

```bash
source venv/bin/activate
```

Install the libraries:

```bash
pip install numpy opencv-python
```

Then go to the folder of the project you want and run its script, for example:

```bash
cd pinhole_projection
python main.py
```

The scripts are named differently in different projects (`main.py`, `day3_distortion.py`, `calibration.py`), so please check the README inside each folder.

Some projects need your own photos:

* `day3_distortion` reads images from the `images/` folder
* `chessboard_corner_detection` and `day5_calibration` read chessboard photos from `calibration_images/`

## What I Learned

* how the pinhole camera model works
* what intrinsic (`K`) and extrinsic (`rvec`, `tvec`) parameters are
* how rotation and translation move points from world to camera coordinates
* what radial and tangential distortion are
* how to prepare 3D and 2D point pairs for calibration
* how to calibrate a camera and how to read the reprojection error
* how to use calibration results to undistort images

The most important thing I understood: if the camera parameters or the distortion are wrong, all later 3D calculations will have errors too.

## Limitations

* The projects are simple and made for learning
* I only used one camera, and calibration quality depends a lot on the photos (angles, sharpness, lighting)
* The code has almost no error handling yet

## Plans

* draw the full 3D cube (edges, not only points)
* try `solvePnP` for pose estimation
* learn about stereo vision and depth
* try 3D reconstruction and Structure-from-Motion (COLMAP)

## License

This repository is created for learning and educational purposes.