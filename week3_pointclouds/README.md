# Point Cloud Processing with Open3D

Practical exercises in 3D point cloud processing using Python, Open3D, and NumPy.

This week focuses on understanding point cloud geometry, visualizing 3D data, and removing noise before more advanced operations such as point cloud registration, ICP, mesh reconstruction, and 3D Gaussian Splatting.

## Project Structure

```text
week3_pointclouds/
├── data/
├── results/
│   ├── statistical/
│   └── radius/
├── day1_pointcloud_basics.py
├── day2_filtering.py
└── README.md
```

## Requirements

- Python 3.10–3.12 recommended
- Open3D
- NumPy
- Matplotlib

Install the dependencies:

```bash
python -m pip install open3d numpy matplotlib
```

---

# Day 1 — Point Cloud Basics

## Overview

The first day introduces point clouds as a representation of 3D geometry.

A point cloud is a collection of points in three-dimensional space. Each point has XYZ coordinates and may also contain color or other attributes.

Point clouds are widely used in photogrammetry, 3D scanning, robotics, autonomous navigation, and digital twin applications.

## Objectives

- Understand the structure of a 3D point cloud.
- Load and visualize point cloud data using Open3D.
- Inspect point coordinates and the number of points.
- Understand basic 3D visualization and coordinate systems.

## Methods and Tools

- Open3D point cloud loading and visualization.
- XYZ coordinate inspection.
- Point cloud rendering.
- Basic 3D geometry analysis.

## Dataset

**Dataset:** Stanford Bunny point cloud.

The Stanford Bunny is a commonly used 3D model for experimenting with geometry processing and visualization.

If another point cloud was used, replace this description with the actual dataset name.

## Implementation

The main script is:

```text
day1_pointcloud_basics.py
```

Run it from the project directory:

```bash
python day1_pointcloud_basics.py
```

## Results

The expected result is a visual representation of a 3D point cloud.

**Point cloud visualization**

<!-- Replace with the actual image path -->

![Day 1 — Point Cloud Visualization](results/day1_pointcloud.png)

**Dataset information**

- Dataset: Stanford Bunny / replace with actual dataset
- Number of points: To be measured
- Visualization library: Open3D

## What I Learned

- How 3D point clouds represent object geometry.
- How to load and visualize point clouds with Open3D.
- How XYZ coordinates describe the spatial position of each point.
- Why point cloud visualization is an important first step in 3D data processing.

---

# Day 2 — Point Cloud Filtering

## Overview

Real-world point clouds often contain noise, isolated points, and irregular sampling density.

Before performing operations such as ICP registration, surface reconstruction, or geometric analysis, it is important to identify and remove unwanted points without destroying useful geometric details.

Day 2 explores three techniques:

1. Voxel Downsampling
2. Statistical Outlier Removal
3. Radius Outlier Removal

## Objectives

- Reduce point cloud density using voxel downsampling.
- Remove statistical outliers.
- Remove points with insufficient local neighbors.
- Compare filtering methods under different parameter settings.
- Evaluate noise reduction and preservation of geometric details.

## Dataset

**Dataset:** [Specify the actual dataset used]

The script can load a point cloud from the `data/` directory.

If no supported file is available, it attempts to load the Stanford Bunny mesh and sample points from its surface.

For the filtering experiments, the script can add synthetic outliers to simulate noisy input data.

The default synthetic outlier ratio is 3%. Set `OUTLIER_RATIO = 0.0` in the script to disable this step.

## Methods Used

### 1. Voxel Downsampling

Voxel downsampling divides 3D space into small cubic regions called voxels.

Points within each voxel are represented by a single point, reducing the total number of points while approximately preserving the object's shape.

```python
downsampled = pcd.voxel_down_sample(
    voxel_size=0.01
)
```

The voxel size controls the degree of downsampling.

- Smaller voxels preserve more details.
- Larger voxels reduce point density more aggressively.

### 2. Statistical Outlier Removal

Statistical Outlier Removal evaluates the average distance from each point to its nearest neighbors.

Points with unusually large average distances are considered potential outliers.

```python
filtered, indices = pcd.remove_statistical_outlier(
    nb_neighbors=20,
    std_ratio=2.0
)
```

Parameters:

- `nb_neighbors`: Number of neighboring points used for the statistical analysis.
- `std_ratio`: Threshold based on the standard deviation of neighbor distances.

Experiments use different values of `nb_neighbors` and `std_ratio`.

A smaller `std_ratio` generally results in more aggressive filtering.

### 3. Radius Outlier Removal

Radius Outlier Removal checks how many neighboring points exist within a specified radius.

Points with too few neighbors are removed.

```python
filtered, indices = pcd.remove_radius_outlier(
    nb_points=16,
    radius=0.05
)
```

Parameters:

- `nb_points`: Minimum number of neighboring points required within the radius.
- `radius`: Search radius around each point.

A larger minimum neighbor count or smaller radius generally makes the filter more aggressive.

The appropriate values depend on the scale and density of the point cloud.

## Implementation

The main script is:

```text
day2_filtering.py
```

Run it from the project directory:

```bash
python day2_filtering.py
```

The script performs the following operations:

1. Loads the point cloud.
2. Optionally adds synthetic outliers.
3. Applies voxel downsampling.
4. Runs Statistical Outlier Removal with multiple parameter combinations.
5. Runs Radius Outlier Removal with multiple parameter combinations.
6. Saves the filtered point clouds.
7. Generates before-and-after comparison images.
8. Writes an experiment summary.

## Experiments

### Statistical Outlier Removal

| Experiment | `nb_neighbors` | `std_ratio` |
|---|---:|---:|
| Conservative | 20 | 3.0 |
| Balanced | 20 | 2.0 |
| Aggressive | 20 | 1.0 |
| Fewer neighbors | 10 | 2.0 |
| More neighbors | 40 | 2.0 |

### Radius Outlier Removal

| Experiment | `nb_points` | `radius` |
|---|---:|---:|
| Conservative | 8 | 0.05 |
| Balanced | 16 | 0.05 |
| Aggressive | 24 | 0.05 |
| Smaller radius | 16 | 0.025 |
| Larger radius | 16 | 0.10 |

**Note:** These are initial experimental settings. They may need to be adjusted according to the point cloud's scale and density.

## Results

### Voxel Downsampling

**Before and after downsampling**

![Voxel Downsampling Comparison](results/voxel_comparison.png)

- Points before downsampling: To be measured
- Points after downsampling: To be measured
- Voxel size: `0.01`

### Statistical Outlier Removal

**Comparison of filtering configurations**

| Experiment | Points Before | Points After | Points Removed |
|---|---:|---:|---:|
| Conservative | — | — | — |
| Balanced | — | — | — |
| Aggressive | — | — | — |

Replace the placeholders with values from `results/experiment_summary.txt`.

Example result image:

![Statistical Filtering Comparison](results/statistical/statistical_balanced_comparison.png)

### Radius Outlier Removal

**Comparison of filtering configurations**

| Experiment | Points Before | Points After | Points Removed |
|---|---:|---:|---:|
| Conservative | — | — | — |
| Balanced | — | — | — |
| Aggressive | — | — | — |

Replace the placeholders with values from `results/experiment_summary.txt`.

Example result image:

![Radius Filtering Comparison](results/radius/radius_balanced_comparison.png)

### Best Parameter Combination

- **Best method:** To be determined after visual inspection.
- **Best parameters:** To be recorded after the experiments.
- **Reason:** The selected configuration should remove unwanted noise while preserving the object's shape and important geometric details.

The most aggressive filter is not necessarily the best one.

## Statistical vs. Radius Outlier Removal

| Characteristic | Statistical Outlier Removal | Radius Outlier Removal |
|---|---|---|
| Main principle | Evaluates average distances to neighboring points | Counts neighbors within a fixed radius |
| Main parameters | `nb_neighbors`, `std_ratio` | `nb_points`, `radius` |
| Useful for | Detecting points with unusually large neighbor distances | Removing isolated points in locally sparse regions |
| Main limitation | Can remove valid points in regions with different sampling densities | Sensitive to the selected radius and local point density |

Both methods can improve point cloud quality, but neither guarantees perfect noise removal.

## What I Learned

- How voxel downsampling reduces point cloud density.
- How Statistical Outlier Removal detects points with unusual neighbor distances.
- How Radius Outlier Removal identifies points with insufficient local support.
- Why parameter tuning is important for point cloud filtering.
- How aggressive filtering can remove useful points and geometric details.
- Why visual inspection is necessary when comparing filtering configurations.

## Conclusion

Day 1 introduced point cloud representation and visualization.

Day 2 extended this foundation with point cloud preprocessing and noise removal.

Voxel downsampling reduces data density, while Statistical and Radius Outlier Removal identify potential outliers using different criteria.

These techniques form an important preprocessing stage for subsequent 3D computer vision tasks, including point cloud registration, ICP, and mesh reconstruction.

---

---

## Day 3 — Normals Estimation, ROI Cropping & Cleaning Pipeline

### Overview

On Day 3, the previous point cloud processing techniques were combined into a single cleaning pipeline. The workflow includes voxel downsampling, two outlier removal methods, surface normal estimation, and Region of Interest (ROI) cropping.

**Technologies:** Python, Open3D, NumPy, Matplotlib.

**Script:** `day3_normals_and_roi.py`

### Pipeline

1. **Load Point Cloud** — Load a point cloud from the `data/` directory. If no supported file is found, the script uses the Stanford Bunny mesh as a fallback.
2. **Voxel Downsampling** — Reduce point density and computational cost.
3. **Statistical Outlier Removal** — Remove points whose average distance to their neighbors is unusually large.
4. **Radius Outlier Removal** — Remove points with too few neighbors within a specified radius.
5. **Normal Estimation** — Estimate local surface orientation using neighboring points.
6. **ROI Cropping** — Extract a region of interest using an axis-aligned bounding box.
7. **Export Results** — Save processed point clouds, visualizations, and a summary of the pipeline parameters.

### Parameters

| Operation | Parameter | Default value |
|---|---|---|
| Voxel Downsampling | `VOXEL_SIZE_RATIO` | `0.005` |
| Statistical Removal | `STATISTICAL_NB_NEIGHBORS` | `20` |
| Statistical Removal | `STATISTICAL_STD_RATIO` | `2.0` |
| Radius Removal | `RADIUS_NB_POINTS` | `8` |
| Radius Removal | `RADIUS_RATIO` | `0.015` |
| Normal Estimation | `NORMAL_RADIUS_RATIO` | `0.03` |
| Normal Estimation | `NORMAL_MAX_NN` | `30` |
| ROI Cropping | `ROI_MIN_RATIO` | `(0.10, 0.10, 0.10)` |
| ROI Cropping | `ROI_MAX_RATIO` | `(0.90, 0.90, 0.90)` |

*Note: Voxel size, radius, and normal search radius are calculated relative to the cloud's bounding-box diagonal. Actual distances therefore depend on the input point cloud.*

### Results

The script saves the processed point clouds and visualizations in the `results/` directory.

| Output | Description |
|---|---|
| `results/01_original.png` | Original point cloud |
| `results/cleaned/02_cleaned.png` | Point cloud after downsampling and outlier removal |
| `results/with_normals/03_normals.png` | Point cloud with estimated surface normals |
| `results/cropped/04_roi.png` | Cropped region of interest |
| `results/05_pipeline_comparison.png` | Comparison of the original, cleaned, and cropped clouds |
| `results/day3_summary.txt` | Point counts, parameter values, and ROI boundaries |

Processed `.ply` files are also saved in `cleaned/`, `with_normals/`, and `cropped/`.

#### Pipeline Visualization

![Original, cleaned, and ROI-cropped point cloud comparison](results/05_pipeline_comparison.png)

#### Surface Normals

![Point cloud with estimated surface normals](results/with_normals/03_normals.png)

#### ROI Cropping

![Region of interest after cropping](results/cropped/04_roi.png)

### Why Normals Matter

Surface normals describe the local orientation of a surface at each point. They are important for:

- **Point-to-plane ICP:** Normals help calculate alignment errors between point clouds during registration.
- **Surface reconstruction:** Normal orientation can improve the reconstruction of meshes from point clouds.
- **Geometry analysis:** Normals provide information about local surface orientation and shape.
- **Visualization:** Normal vectors help inspect the geometric structure of a point cloud.

Normals are estimated using `estimate_normals()` with a hybrid KD-tree search based on a specified radius and maximum number of neighbors.

### ROI Cropping

An Axis-Aligned Bounding Box (AABB) was used to select a region of interest.

Unlike an oriented bounding box, an AABB is aligned with the coordinate axes. It is simple to configure and efficient for rectangular crops, although it may include unwanted points when the target object is rotated.

The current implementation defines ROI boundaries as fractions of the cloud's bounding box. These settings are intended as a demonstration and should be adjusted for real-world scans.

### What I Learned

- How voxel downsampling reduces point density before further processing.
- How Statistical and Radius Outlier Removal complement each other.
- How to estimate surface normals using local neighborhoods.
- Why normal estimation is important for Point-to-Plane ICP and surface reconstruction.
- How to crop a point cloud using an axis-aligned bounding box.
- How to combine preprocessing operations into a reusable pipeline.
- How to export processed point clouds and visualizations for further analysis.

### Conclusion

Day 3 brought together the main preprocessing operations required to prepare point clouds for more advanced 3D computer vision tasks.

The resulting pipeline provides a starting point for point cloud registration, mesh reconstruction, and Scan-to-BIM workflows. Its parameters must be adapted to the density, scale, noise level, and geometry of the input data.