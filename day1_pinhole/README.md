# Pinhole Projection

A simple example of **3D-to-2D point projection** using the **pinhole camera model** and OpenCV.

This project demonstrates how 3D points are projected onto a 2D image using:

* camera **intrinsic parameters**;
* camera **extrinsic parameters**;
* perspective projection;
* OpenCV visualization.

---

## Overview

The project creates a simple 3D scene consisting of the vertices of a unit cube and the origin of the coordinate system.

A virtual camera is then configured using its intrinsic and extrinsic parameters.

The 3D points are projected onto the image plane using OpenCV's `cv2.projectPoints()` function.

The resulting 2D points are visualized on a white image.

### Projection pipeline

```text
3D Point
   │
   ▼
Camera Extrinsics
(Rotation + Translation)
   │
   ▼
Camera Coordinates
   │
   ▼
Perspective Projection
   │
   ▼
Camera Intrinsics
   │
   ▼
2D Image Point
```

---

## Pinhole Camera Model

The pinhole camera model describes how a 3D point is projected onto a 2D image plane.

For a point in camera coordinates:

```text
(X, Y, Z)
```

the ideal perspective projection is:

```text
u = fx * X / Z + cx
v = fy * Y / Z + cy
```

where:

* `u`, `v` — pixel coordinates;
* `X`, `Y`, `Z` — 3D coordinates in the camera coordinate system;
* `fx`, `fy` — focal lengths in pixels;
* `cx`, `cy` — principal point coordinates.

---

## Camera Intrinsics

The camera intrinsic matrix `K` is defined as:

```python
K = np.array([
    [fx, 0, cx],
    [0, fy, cy],
    [0,  0,  1]
], dtype=np.float32)
```

For an image with a resolution of `640 × 480`:

```python
width = 640
height = 480

fx = 800
fy = 800

cx = width / 2
cy = height / 2
```

The resulting camera matrix is:

```text
K =

[ 800    0   320 ]
[   0  800   240 ]
[   0    0     1 ]
```

The matrix `K` describes the internal properties of the virtual camera.

---

## Camera Extrinsics

The camera's position and orientation relative to the scene are defined using rotation and translation.

### Rotation

```python
rvec = np.array(
    [0.5, 0.5, 0.0],
    dtype=np.float32
)
```

`rvec` is a **Rodrigues rotation vector** used by OpenCV to represent the camera rotation.

### Translation

```python
tvec = np.array(
    [0.0, 0.0, 4.0],
    dtype=np.float32
)
```

`tvec` represents the camera translation.

Together, the rotation and translation transform points from the world coordinate system into the camera coordinate system:

```text
X_camera = R · X_world + t
```

where `R` is the rotation matrix obtained from `rvec`.

---

## 3D Scene

The scene contains eight vertices of a unit cube:

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

The first four points also represent the coordinate axes:

```text
[0, 0, 0] → Origin
[1, 0, 0] → X axis
[0, 1, 0] → Y axis
[0, 0, 1] → Z axis
```

---

## 3D → 2D Projection

OpenCV is used to project the 3D points onto the image plane:

```python
points_2d, _ = cv2.projectPoints(
    points_3d,
    rvec,
    tvec,
    K,
    None
)
```

The function takes:

* `points_3d` — 3D points;
* `rvec` — camera rotation;
* `tvec` — camera translation;
* `K` — camera intrinsic matrix;
* `None` — distortion coefficients.

The resulting coordinates are 2D image points:

```python
points_2d = points_2d.reshape(-1, 2)
```

---

## Visualization

A white `640 × 480` image is created:

```python
img = np.ones(
    (height, width, 3),
    dtype=np.uint8
) * 255
```

The projected points and coordinate axes are then drawn using OpenCV.

### Coordinate axes

* **X axis** — red
* **Y axis** — green
* **Z axis** — blue
* **Projected points** — black

The final image is saved as:

```text
result.png
```

and displayed in an OpenCV window.

---

## Project Structure

Recommended project structure:

```text
pinhole_projection/
│
├── main.py
├── result.png
├── README.md
└── requirements.txt
```

### `main.py`

The main Python script that:

1. creates the 3D scene;
2. defines camera intrinsics;
3. defines camera extrinsics;
4. projects 3D points to 2D;
5. visualizes the result;
6. saves the output image.

### `result.png`

The generated visualization of the projected 3D points.

### `requirements.txt`

Contains the Python dependencies required to run the project.

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/pinhole_projection.git
cd pinhole_projection
```

### 2. Create a virtual environment

#### Linux / macOS

```bash
python3 -m venv venv
source venv/bin/activate
```

#### Windows

```powershell
python -m venv venv
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## 📦 Requirements

The project uses:

* **Python 3**
* **NumPy**
* **OpenCV**

Example `requirements.txt`:

```text
numpy
opencv-python
```

---

## ▶Usage

Run the main script:

```bash
python main.py
```

If the projection is completed successfully, the program prints:

```text
Projection completed successfully!
Result saved to: result.png
```

An OpenCV window named:

```text
Pinhole Projection
```

will also display the result.

Press any key to close the window.

---

## Experiments

This project is intentionally simple and can be used to experiment with camera geometry.

### Change the focal length

Try:

```python
fx = 400
fy = 400
```

or:

```python
fx = 1200
fy = 1200
```

Changing the focal length affects the perspective and scale of the projected points.

### Change the camera distance

For example:

```python
tvec = np.array(
    [0.0, 0.0, 6.0],
    dtype=np.float32
)
```

This changes the camera's position relative to the scene.

### Change the camera rotation

For example:

```python
rvec = np.array(
    [0.0, 0.5, 0.0],
    dtype=np.float32
)
```

This changes the camera orientation.

---

## 📚 Key Concepts

| Concept               | Description                                |
| --------------------- | ------------------------------------------ |
| `3D Point`            | A point in 3D space                        |
| `2D Point`            | A point on the image plane                 |
| `K`                   | Camera intrinsic matrix                    |
| `fx`, `fy`            | Focal lengths in pixels                    |
| `cx`, `cy`            | Principal point coordinates                |
| `rvec`                | Rotation vector                            |
| `tvec`                | Translation vector                         |
| `R`                   | Rotation matrix                            |
| `cv2.projectPoints()` | OpenCV 3D-to-2D projection function        |
| `Pinhole Camera`      | Mathematical model of a perspective camera |

---

## Learning Goals

The main goal of this project is to provide a simple and practical introduction to **perspective projection** and **camera geometry**.

It can serve as a starting point for learning:

* Computer Vision
* 3D Geometry
* Camera Models
* Perspective Projection
* OpenCV
* NumPy
* Camera Calibration
* Pose Estimation
* `solvePnP`
* Stereo Vision
* 3D Reconstruction

---

## Possible Extensions

The project can be extended with:

* drawing the complete 3D cube;
* camera distortion;
* interactive camera movement;
* 3D visualization;
* Euler angles;
* rotation matrices;
* homogeneous coordinates;
* camera calibration;
* pose estimation with `solvePnP`;
* multiple cameras;
* stereo projection;
* depth visualization.

---

## License

This project is provided for educational purposes.

You can add a specific open-source license, such as the **MIT License**, depending on how you want others to use, modify, and distribute the project.
