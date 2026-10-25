# Point Cloud Processing with Open3D

Practical exercises in 3D point cloud processing using Python, Open3D, and NumPy.

This week focuses on understanding point cloud geometry, visualizing 3D data, removing noise, estimating surface normals, and extracting regions of interest (ROI).

These techniques are essential preprocessing steps for advanced 3D computer vision tasks, including point cloud registration, Iterative Closest Point (ICP), mesh reconstruction, 3D Gaussian Splatting, and Scan-to-BIM workflows.

## Project Structure

```text
week3_pointclouds/
├── data/
├── results/
│   ├── statistical/
│   ├── radius/
│   ├── cleaned/
│   ├── with_normals/
│   ├── cropped/
│   ├── before.png
│   ├── 01_original.png
│   ├── 05_pipeline_comparison.png
│   └── day3_summary.txt
├── day1_pointcloud_basics.py
├── day2_filtering.py
├── day3_normals_and_roi.py
└── README.md
```

The `results/` directory contains visualizations, filtered point clouds, and experiment summaries generated throughout the week.

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

The Stanford Bunny is a commonly used 3D model for experimenting with geometry processing, visualization, and point cloud algorithms.

## Implementation

Main script:

```text
day1_pointcloud_basics.py
```

Run it from the project directory:

```bash
python day1_pointcloud_basics.py
```

## Results

The script loads and visualizes a 3D point cloud using Open3D.

**Point cloud visualization**

![Day 1 — Point Cloud Visualization](results/before.png)

### Dataset Information

- **Dataset:** Stanford Bunny
- **Data representation:** 3D point cloud
- **Coordinates:** XYZ
- **Visualization library:** Open3D

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

Day 2 explores three point cloud preprocessing techniques:

1. Voxel Downsampling
2. Statistical Outlier Removal
3. Radius Outlier Removal

The experiments compare different parameter configurations to understand how filtering affects point density, noise, and geometry preservation.

## Objectives

- Reduce point cloud density using voxel downsampling.
- Remove statistical outliers.
- Remove points with insufficient local neighbors.
- Compare filtering methods under different parameter settings.
- Evaluate noise reduction and preservation of geometric details.
- Identify a suitable filtering configuration for subsequent processing.

## Dataset

**Dataset:** Stanford Bunny with 3% synthetic noise.

The Stanford Bunny point cloud is used to evaluate filtering methods under controlled noise conditions.

Synthetic outliers are added at a ratio of 3% to simulate noisy input data.

Using the same dataset across experiments makes it possible to compare different filtering configurations under consistent conditions.

## Methods Used

### 1. Voxel Downsampling

Voxel downsampling divides 3D space into small cubic regions called voxels.

Points within each voxel are represented by a single point, reducing the total number of points while approximately preserving the object's overall shape.

```python
downsampled = pcd.voxel_down_sample(
    voxel_size=0.01
)
```

The voxel size controls the degree of downsampling.

- Smaller voxels preserve more geometric details.
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

- `nb_neighbors`: Number of neighboring points used for statistical analysis.
- `std_ratio`: Threshold based on the standard deviation of neighbor distances.

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

Main script:

```text
day2_filtering.py
```

Run it from the project directory:

```bash
python day2_filtering.py
```

The script performs the following operations:

1. Loads the point cloud.
2. Adds synthetic outliers when noise simulation is enabled.
3. Applies voxel downsampling.
4. Runs Statistical Outlier Removal with multiple parameter combinations.
5. Runs Radius Outlier Removal with multiple parameter combinations.
6. Saves filtered point clouds.
7. Generates before-and-after comparison images.
8. Writes an experiment summary.

## Experiments

### Statistical Outlier Removal

The following configurations are used to compare filtering behavior.

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

**Note:** These configurations are experimental starting points. The appropriate parameters depend on the point cloud's scale and density.

## Results

The experiment summary is saved to:

```text
results/experiment_summary.txt
```

### Voxel Downsampling

**Before and after downsampling**

![Voxel Downsampling Comparison](results/voxel_comparison.png)

The purpose of voxel downsampling is to reduce computational cost while retaining the object's overall geometry.

- **Points before downsampling:** See `results/experiment_summary.txt`.
- **Points after downsampling:** See `results/experiment_summary.txt`.
- **Voxel size:** Depends on the configuration used in the experiment.

### Statistical Outlier Removal

**Comparison of filtering configurations**

| Experiment | Points Before | Points After | Points Removed |
|---|---:|---:|---:|
| Conservative | See summary | See summary | See summary |
| Balanced | See summary | See summary | See summary |
| Aggressive | See summary | See summary | See summary |

![Statistical Filtering Comparison](results/statistical/statistical_balanced_comparison.png)

### Radius Outlier Removal

**Comparison of filtering configurations**

| Experiment | Points Before | Points After | Points Removed |
|---|---:|---:|---:|
| Conservative | See summary | See summary | See summary |
| Balanced | See summary | See summary | See summary |
| Aggressive | See summary | See summary | See summary |

![Radius Filtering Comparison](results/radius/radius_balanced_comparison.png)

### Best Parameter Combination

The best configuration should be selected based on the actual experimental results and visual inspection.

The selection criteria are:

- Effective removal of isolated noise points.
- Preservation of the Bunny's overall shape.
- Retention of important geometric details.
- Avoidance of excessive removal of valid surface points.

The most aggressive filter is not necessarily the best one. A configuration that removes more points may also eliminate useful geometric information.

**The selected Day 2 configuration is reused in the Day 3 cleaning pipeline** to maintain a consistent preprocessing approach.

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

Day 2 introduced practical point cloud filtering techniques using a Stanford Bunny dataset with 3% synthetic noise.

Voxel downsampling reduces data density, while Statistical and Radius Outlier Removal identify potential outliers using different criteria.

Comparing these methods provides a foundation for selecting appropriate preprocessing parameters before more advanced 3D computer vision operations.

---

# Day 3 — Normals Estimation, ROI Cropping & Cleaning Pipeline

## Overview

Day 3 combines the point cloud processing techniques from previous exercises into a single cleaning pipeline.

The workflow includes:

1. Voxel downsampling.
2. Statistical Outlier Removal.
3. Radius Outlier Removal.
4. Surface normal estimation.
5. Region of Interest (ROI) cropping.
6. Exporting processed point clouds and visualizations.

The pipeline demonstrates how to prepare a point cloud for subsequent geometry processing and registration tasks.

**Technologies:** Python, Open3D, NumPy, Matplotlib.

**Main script:** `day3_normals_and_roi.py`

## Dataset

**Dataset:** Stanford Bunny point cloud.

The input cloud contains 30,000 points.

The Day 3 pipeline reuses the best filtering configuration selected during Day 2, keeping the preprocessing strategy consistent across both exercises.

## Pipeline

1. **Load Point Cloud** — Load the input point cloud. If no supported file is available, the script uses the Stanford Bunny mesh as a fallback.
2. **Voxel Downsampling** — Reduce point density and computational cost.
3. **Statistical Outlier Removal** — Remove points whose average distance to their neighbors is unusually large.
4. **Radius Outlier Removal** — Remove points with too few neighbors within the specified radius.
5. **Normal Estimation** — Estimate local surface orientation using neighboring points.
6. **ROI Cropping** — Extract a region of interest using an axis-aligned bounding box.
7. **Export Results** — Save processed point clouds, visualizations, and a summary of pipeline parameters.

## Parameters

The following values were recorded in `day3_summary.txt`.

| Operation | Parameter | Actual Value |
|---|---|---:|
| Voxel Downsampling | Voxel size | `0.00124996` |
| Statistical Removal | `nb_neighbors` | `20` |
| Statistical Removal | `std_ratio` | `2.0` |
| Radius Removal | Minimum neighbors | `8` |
| Radius Removal | Radius | `0.00374826` |
| Normal Estimation | Search radius | `0.00749651` |
| Normal Estimation | Maximum neighbors | `30` |
| ROI Cropping | Minimum ratios | `(0.1, 0.1, 0.1)` |
| ROI Cropping | Maximum ratios | `(0.9, 0.9, 0.9)` |

The voxel size, radius, and normal search radius are calculated relative to the point cloud's bounding-box diagonal. Their actual distances therefore depend on the input point cloud.

## Results

### Point Count Summary

The pipeline produced the following results:

| Processing Stage | Number of Points | Points Removed |
|---|---:|---:|
| Original point cloud | 30,000 | — |
| After voxel downsampling | 20,893 | 9,107 |
| After statistical filtering | 20,260 | 633 |
| After radius filtering | 20,180 | 80 |
| After ROI cropping | 11,869 | 8,311 |

The point cloud was reduced from **30,000 to 11,869 points**, a total reduction of approximately **60.44%**.

This reduction reflects both point density reduction and ROI cropping, as well as outlier removal. It should not be interpreted as the percentage of noise removed.

### Best Filter from Day 2

The best filter configuration selected during Day 2 was used in the Day 3 pipeline.

This allows the same preprocessing strategy to be applied before normal estimation and ROI cropping, creating a consistent workflow across the two exercises.

### Pipeline Outputs

The script saves processed point clouds and visualizations in the `results/` directory.

| Output | Description |
|---|---|
| `results/01_original.png` | Original point cloud |
| `results/cleaned/02_cleaned.png` | Point cloud after downsampling and outlier removal |
| `results/with_normals/03_normals.png` | Point cloud with estimated surface normals |
| `results/cropped/04_roi.png` | Cropped region of interest |
| `results/05_pipeline_comparison.png` | Comparison of the original, cleaned, and cropped point clouds |
| `results/day3_summary.txt` | Point counts, parameter values, and ROI boundaries |

Processed `.ply` files are also saved in the corresponding `cleaned/`, `with_normals/`, and `cropped/` directories.

### Pipeline Visualization

![Original, cleaned, and ROI-cropped point cloud comparison](results/05_pipeline_comparison.png)

### Surface Normals

![Point cloud with estimated surface normals](results/with_normals/03_normals.png)

### ROI Cropping

![Region of interest after cropping](results/cropped/04_roi.png)

## ROI Bounding Box

The axis-aligned bounding box used to crop the point cloud has the following boundaries:

**Minimum bound:**

```text
[-0.07909207, 0.04872920, -0.04972257]
```

**Maximum bound:**

```text
[0.04540350, 0.17186362, 0.04672223]
```

These coordinates define the selected region of interest in the point cloud's coordinate system.

An Axis-Aligned Bounding Box (AABB) is aligned with the coordinate axes. It is simple to configure and efficient for rectangular crops, although it may include unwanted points when the target object is rotated.

The current implementation defines ROI boundaries as fractions of the cloud's bounding box. These settings demonstrate the cropping process and should be adapted to the geometry of real-world scans.

## Why Normals Matter

Surface normals describe the local orientation of a surface at each point.

They are important for:

- **Point-to-plane ICP:** Normals help calculate alignment errors between point clouds during registration.
- **Surface reconstruction:** Normal orientation can improve mesh reconstruction from point clouds.
- **Geometry analysis:** Normals provide information about local surface orientation and shape.
- **Visualization:** Normal vectors help inspect the geometric structure of a point cloud.

Normals are estimated using `estimate_normals()` with a hybrid KD-tree search based on a specified radius and maximum number of neighbors.

## What I Learned

- How voxel downsampling reduces point density before further processing.
- How Statistical and Radius Outlier Removal complement each other.
- How to estimate surface normals using local neighborhoods.
- Why normal estimation is important for Point-to-Plane ICP and surface reconstruction.
- How to crop a point cloud using an axis-aligned bounding box.
- How to combine preprocessing operations into a reusable pipeline.
- How to export processed point clouds and visualizations for further analysis.

## Conclusion

Day 3 combined the main point cloud preprocessing operations into a single pipeline.

Starting with 30,000 points, the pipeline applied voxel downsampling, Statistical Outlier Removal, Radius Outlier Removal, normal estimation, and ROI cropping, resulting in a cropped point cloud containing 11,869 points.

The best filtering configuration from Day 2 was reused in the Day 3 pipeline to maintain a consistent preprocessing approach.

This workflow provides a foundation for more advanced 3D computer vision tasks, including point cloud registration, Point-to-Plane ICP, mesh reconstruction, and Scan-to-BIM applications.

The results demonstrate how preprocessing can reduce data size and isolate relevant geometry before subsequent analysis.