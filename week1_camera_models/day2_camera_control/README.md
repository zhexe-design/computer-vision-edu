# Pinhole Projection

A simple Python project that shows how 3D points can be projected onto a 2D image using the pinhole camera model.

The project uses NumPy and OpenCV.

## About

In this project, I create a small 3D scene with points representing a cube and coordinate axes.

Then I create a virtual camera and set its parameters:

* image size
* focal length
* camera center
* camera rotation
* camera translation

After that, the 3D points are projected to 2D points using `cv2.projectPoints()`.

The final result is drawn on a white image and saved as `result.png`.

## How it works

The main steps are:

1. Create 3D points.
2. Create the camera matrix.
3. Set camera rotation and translation.
4. Project 3D points to 2D.
5. Draw the points and axes.
6. Save and display the result.

The basic projection looks like this:

```text
3D points
    |
    v
Camera rotation and translation
    |
    v
Camera coordinates
    |
    v
2D image coordinates
```

## Technologies

* Python
* NumPy
* OpenCV

## Project Structure

```text
pinhole_projection/
|
├── main.py
├── requirements.txt
├── README.md
└── result.png
```

## Installation

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/pinhole_projection.git
cd pinhole_projection
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

The project uses the following Python packages:

```text
numpy
opencv-python
```

## Run

Run the program with:

```bash
python main.py
```

After running the program, an OpenCV window will show the projected points.

The result is also saved to:

```text
result.png
```

The program prints:

```text
Projection completed successfully!
Result saved to: result.png
```

## Camera Parameters

The camera resolution is:

```python
width = 640
height = 480
```

The focal length is:

```python
fx = 800
fy = 800
```

The center of the image is calculated from the image size:

```python
cx = width / 2
cy = height / 2
```

The camera matrix is:

```python
K = np.array([
    [fx, 0, cx],
    [0, fy, cy],
    [0, 0, 1]
], dtype=np.float32)
```

The camera rotation and translation are defined using:

```python
rvec = np.array([0.5, 0.5, 0.0], dtype=np.float32)

tvec = np.array([0.0, 0.0, 4.0], dtype=np.float32)
```

## 3D Points

The project uses several 3D points:

```python
points_3d = np.array([
    [0, 0, 0],
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1],
    [1, 1, 0],
    [1, 0, 1],
    [0, 1, 1],
    [1, 1, 1]
], dtype=np.float32)
```

These points can be used to represent the vertices of a cube.

The first four points are also used to draw the coordinate axes:

* X axis is red
* Y axis is green
* Z axis is blue

## OpenCV Projection

The main OpenCV function used in this project is:

```python
cv2.projectPoints()
```

It takes the 3D points, camera rotation, camera translation, camera matrix and distortion coefficients.

In this project, no lens distortion is used:

```python
None
```

The result is a list of 2D points that can be drawn on an image.

## Result

The program creates a simple image with the projected points and coordinate axes.

The image is saved as:

```text
result.png
```

## What I Learned

While working on this project, I learned the basics of:

* 3D points
* 2D image coordinates
* camera intrinsic parameters
* camera extrinsic parameters
* rotation and translation
* perspective projection
* OpenCV `projectPoints()`