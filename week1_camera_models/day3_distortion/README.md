Description

In this task I learned about camera lens distortion and how to add and remove distortion using OpenCV.

Camera distortion changes the position of points in an image. Because of this, straight lines near the edges of an image can look curved.

I used several photos with straight lines, for example doors, windows, books and buildings.

Radial Distortion

Radial distortion happens because of the camera lens. The further a point is from the center of the image, the stronger the distortion can be.

There are two common types:

Barrel distortion — lines look curved outward from the center. The image looks a little like a barrel.
Pincushion distortion — lines look curved inward toward the center. The image looks like a pincushion.

The main radial distortion coefficients are:

k1
k2
k3

The first coefficient k1 has a strong effect on the distortion.

In my experiment:

dist_coeffs = np.array([-0.3, 0.1, 0.0, 0.0, 0.0])

A negative k1 creates barrel distortion.

A positive k1 creates pincushion distortion.

Tangential Distortion

Tangential distortion happens when the camera lens is not perfectly aligned with the camera sensor.

The main tangential distortion coefficients are:

p1
p2

Radial distortion is related to the distance from the image center, while tangential distortion is related to the position and alignment of the lens.

In this task I mainly worked with radial distortion.

What I Did

The program does the following:

Reads all images from the images folder.
Creates a camera matrix.
Creates artificial radial distortion.
Applies the distortion to the original image.
Removes the distortion using OpenCV.
Creates a comparison of the original, distorted and undistorted images.
Saves the results to the results folder.

The main OpenCV functions used in the project are:

cv2.initUndistortRectifyMap()
cv2.remap()
cv2.undistort()
Results

For each input image, the program creates four files:

original.jpg
distorted.jpg
undistorted.jpg
comparison.jpg

The comparison image contains:

Original | Distorted | Undistorted

This makes it easier to see how the distortion changes the straight lines and how the correction works.

Project Structure
day3_distortion/
│
├── day3_distortion.py
├── images/
│   ├── 1.jpg
│   ├── 2.jpg
│   └── ...
│
└── results/
    ├── 1_original.jpg
    ├── 1_distorted.jpg
    ├── 1_undistorted.jpg
    └── 1_comparison.jpg
How to Run

Install the required libraries:

pip install opencv-python numpy

Then run:

python day3_distortion.py

The program automatically finds all supported images in the images folder and processes them one by one.

Conclusion

In this task I learned the difference between radial and tangential distortion.

I also learned how to artificially add distortion to an image and then remove it using OpenCV.

This is important for 3D reconstruction because camera distortion can change the position of points in an image. If the distortion is ignored, the 3D reconstruction can have incorrect results. Before serious 3D work, camera calibration and distortion correction are therefore important.