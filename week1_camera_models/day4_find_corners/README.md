# Chessboard Corner Detection

A simple Python project for finding chessboard corners in images using OpenCV.

This project is a first step toward camera calibration. It detects the internal corners of a chessboard and saves their 2D image coordinates together with the corresponding 3D points.

## About

Camera calibration is used to find the parameters of a camera.

One common way to do this is to use a chessboard pattern.

In this project, I use several photos of a chessboard. The program tries to find the internal corners on each photo.

The main steps are:

1. Create 3D points for the chessboard.
2. Find calibration images.
3. Convert images to grayscale.
4. Find chessboard corners.
5. Improve the corner positions.
6. Save the detected points.
7. Show the detected corners.

The actual camera calibration is not performed yet. The project only prepares the data needed for the next step.

## Technologies

* Python
* OpenCV
* NumPy

## Project Structure

```text
chessboard_corner_detection/
|
├── main.py
├── calibration_images/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── image3.jpg
├── README.md
└── requirements.txt
```

## Installation

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/chessboard_corner_detection.git
cd chessboard_corner_detection
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

Install the required packages:

```bash
pip install -r requirements.txt
```

## Requirements

The project uses:

```text
numpy
opencv-python
```

The `requirements.txt` file can contain:

```text
numpy
opencv-python
```

## Calibration Images

Put chessboard images into the following folder:

```text
calibration_images/
```

The program supports `.jpg` and `.png` images.

Example:

```text
calibration_images/
|
├── image1.jpg
├── image2.jpg
├── image3.jpg
├── image4.jpg
└── image5.jpg
```

It is better to use several photos of the same chessboard from different positions and angles.

## Chessboard Size

The chessboard size is defined here:

```python
CHESSBOARD_SIZE = (9, 6)
```

This means that the program is looking for:

* 9 internal corners horizontally
* 6 internal corners vertically

It is important that this is the number of internal corners, not the number of squares.

For example, a board with 10 x 7 squares has 9 x 6 internal corners.

## Square Size

The size of one chessboard square is:

```python
SQUARE_SIZE = 0.025
```

The value is given in meters.

This means that one square is:

```text
0.025 m = 2.5 cm
```

The exact value is not very important for just finding the corners. It becomes important when the actual camera calibration is performed.

## 3D Points

The program creates 3D points for the chessboard:

```python
objp = np.zeros(
    (CHESSBOARD_SIZE[0] * CHESSBOARD_SIZE[1], 3),
    np.float32
)
```

The points are located on the same flat plane.

For example:

```text
(0,0,0)  (1,0,0)  (2,0,0)  ...

(0,1,0)  (1,1,0)  (2,1,0)  ...

(0,2,0)  (1,2,0)  (2,2,0)  ...
```

The points are then multiplied by `SQUARE_SIZE` to get their real-world coordinates.

These points represent where the chessboard corners are located in 3D space.

## Finding Images

The program searches for images inside:

```python
calibration_images/
```

It looks for both JPG and PNG files:

```python
images = (
    glob.glob("calibration_images/*.jpg")
    + glob.glob("calibration_images/*.png")
)
```

If the folder is empty, the program does not crash. It simply prints a message that no images were found.

## Finding Chessboard Corners

Each image is first converted to grayscale:

```python
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
```

Then OpenCV tries to find the chessboard corners:

```python
ret, corners = cv2.findChessboardCorners(
    gray,
    CHESSBOARD_SIZE,
    None
)
```

If the corners are found, the program prints:

```text
Углы найдены!
```

If they are not found:

```text
Углы НЕ найдены.
```

## Improving Corner Positions

After finding the corners, the program improves their positions using:

```python
cv2.cornerSubPix()
```

This gives more accurate corner coordinates.

The program uses:

```python
criteria = (
    cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
    30,
    0.001
)
```

These settings define when the corner refinement should stop.

## Saving the Points

If the corners are found, two types of points are stored:

```python
objpoints
```

contains the 3D coordinates of the chessboard corners.

```python
imgpoints
```

contains the 2D coordinates of the same corners in the image.

These two lists will be needed for camera calibration.

The basic idea is:

```text
3D chessboard points
        |
        | camera projection
        v
2D image points
```

## Visualization

The program draws the detected corners on the original image:

```python
cv2.drawChessboardCorners(
    img,
    CHESSBOARD_SIZE,
    corners2,
    ret
)
```

The image is then displayed in an OpenCV window.

The window stays open for about 500 milliseconds for each image:

```python
cv2.waitKey(500)
```

## Run

Run the program with:

```bash
python main.py
```

If there are no images in the folder, the program will show:

```text
Photographs not found.
Folder calibration_images/ is empty.
```

If images are available, the program will process them one by one.

At the end, it prints information about the result:

```text
================================
Result
================================

Total images: 5
Successfully detected boards: 4
```

If at least one chessboard was detected, the program says that the data is ready for camera calibration.

## What I Learned

While working on this project, I learned the basics of:

* chessboard patterns
* camera calibration
* 3D object points
* 2D image points
* OpenCV `findChessboardCorners()`
* OpenCV `cornerSubPix()`
* OpenCV `drawChessboardCorners()`
* grayscale images
* preparing data for camera calibration

## Current Limitations

This project does not perform the final camera calibration yet.

It only finds the chessboard corners and prepares the data.

The next step is to use:

```python
cv2.calibrateCamera()
```

to calculate the camera parameters.

## Possible Improvements

Some possible next steps:

* perform full camera calibration
* calculate the camera matrix
* find lens distortion coefficients
* save calibration results to a file
* use the calibration results to undistort images
* add more image formats
* add better error handling
* save images with detected corners

## Goal

The goal of this project is to understand the first step of camera calibration and learn how OpenCV can be used to find known points on an image.

This project is mainly for learning and experimenting with computer vision.

## License

This project is created for learning and educational purposes.
