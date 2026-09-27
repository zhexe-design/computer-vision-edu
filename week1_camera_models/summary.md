Camera calibration is needed because all 3D methods assume that we know how the camera turns 3D points into pixels. A real camera is not a perfect pinhole. It has its own focal length, its own image center and some lens distortion. Calibration finds these numbers, so the computer can trust the image geometry.

In COLMAP, the goal is to find camera positions and a 3D point cloud from many photos. COLMAP can estimate the camera parameters by itself, but this is a hard problem, and a wrong start can give a bent or broken reconstruction. If you calibrate the camera in advance and give COLMAP good focal length and distortion values, the result is usually more stable and more accurate.

Gaussian Splatting usually starts from COLMAP results, so it depends on them. It also assumes an ideal pinhole camera without distortion, so the images have to be undistorted first. If the calibration is bad, the camera poses are slightly wrong, and the final scene looks blurry, with floaters and ghost edges.

In AR, virtual objects must sit exactly on top of the real world. To draw them in the right place and at the right size, the app needs the intrinsic matrix K and the distortion coefficients. If the calibration is wrong, the virtual object slides or looks like it is floating.

In short, calibration removes a big source of error before any 3D work starts.