# Week 3 — Point Cloud Processing with Open3D

A six-day project on point cloud processing with Open3D: loading, visualization, downsampling, outlier removal, normal estimation, ROI cropping and a reusable cleaning pipeline. The project prepares data for Week 4 (ICP Registration).

## Project Structure

```text
week3_pointclouds/
├── data/
├── results/
│   ├── statistical/
│   ├── radius/
│   ├── cleaned/
│   ├── cleaned_v1/
│   ├── cleaned_v2/
│   ├── comparison/
│   ├── with_normals/
│   └── cropped/
├── day1_pointcloud_basics.py
├── day2_filtering.py
├── day3_normals_and_roi.py
├── day4_advanced_cleaning.py
├── day5_full_pipeline.py
└── README.md
```

## Installation

```bash
pip install open3d
```

---

# Day 1 — Point Cloud Basics with Open3D

## What I did
- Installed Open3D and verified the environment
- Loaded a point cloud (`o3d.io.read_point_cloud`)
- Visualized it (`o3d.visualization.draw_geometries`)
- Applied voxel downsampling (`voxel_down_sample`)
- Removed statistical outliers (`remove_statistical_outlier`)
- Saved the result to `results/`

## Results
- Before / After screenshots: `[add screenshots]`
- Point count: original `...` → downsampled `...` → cleaned `...`

## How to run
```bash
pip install open3d
python day1_pointcloud_basics.py
```

## What I learned
- A point cloud is a set of points with XYZ coordinates, optionally with color and normals
- Unlike a mesh, a point cloud has no connectivity (no edges or faces)
- Downsampling reduces size and evens out density; outlier removal gets rid of noise
- The basic Open3D workflow: load → visualize → process → save

---

# Day 2 — Point Cloud Filtering

## Methods used
- Voxel downsampling
- Statistical Outlier Removal (`nb_neighbors`, `std_ratio`)
- Radius Outlier Removal (`nb_points`, `radius`)

## Experiments
- Statistical: `nb_neighbors` = ..., `std_ratio` = 1.0 / 2.0 / 3.0
- Radius: `nb_points` = ..., `radius` = ...
- Outputs saved to `results/statistical/` and `results/radius/`

## Results
- Before / After comparisons: `[add screenshots]`
- Best parameter combination: `...`

## How to run
```bash
python day2_filtering.py
```

## What I learned
- Statistical removal discards points whose mean distance to neighbors is far from the global average
- Radius removal discards points with too few neighbors within a given radius
- Parameter tuning matters: overly aggressive filtering removes useful points
- Recommended order: downsampling → statistical → radius

---

# Day 3 — Normals and ROI Cropping

## Pipeline
1. Load point cloud
2. Voxel downsampling
3. Statistical / Radius outlier removal
4. Estimate normals (`KDTreeSearchParamHybrid(radius=0.1, max_nn=30)`)
5. Crop region of interest (`AxisAlignedBoundingBox`)
6. Save results

## Results
- Before / After cleaning: `[add screenshots]`
- Point cloud with normals: `results/with_normals/`
- Cropped cloud: `results/cropped/`

## Data notes
- Source: `[Open3D demo / public dataset / own scan]`

## How to run
```bash
python day3_normals_and_roi.py
```

## What I learned
- Normals are needed for Point-to-Plane ICP, shading and surface reconstruction
- `estimate_normals` fits a local plane using neighbors found via a KD-tree
- ROI cropping removes irrelevant areas of real-world scans
- An axis-aligned bounding box is simple and fast; more complex crops handle arbitrary shapes

---

# Day 4 — Advanced Cleaning

## Filtering Experiments
- Combination 1 (`results/cleaned_v1/`): voxel=... + statistical(nb_neighbors=..., std_ratio=...)
- Combination 2 (`results/cleaned_v2/`): voxel=... + statistical(...) + radius(nb_points=..., radius=...)
- Point counts per variant: `...`
- Comparison images: `results/comparison/`

## Best result
- `voxel=0.02 + statistical(nb_neighbors=20, std_ratio=2.0)` (replace with your own)

## Observations
- Too aggressive filtering removes useful points
- The order of operations matters
- Combining methods handles different types of noise

## How to run
```bash
python day4_advanced_cleaning.py
```

## What I learned
- How to combine downsampling, statistical and radius filtering
- How to choose parameters deliberately, not by guessing
- How to compare variants visually and by remaining point count

---

# Day 5 — Full Pipeline

## Overview
A complete, reusable pipeline for loading, cleaning, normal estimation and ROI cropping, runnable with one command.

## Full Pipeline
1. Load point cloud
2. Voxel downsampling
3. Statistical (+ optional Radius) outlier removal
4. Estimate normals
5. Crop region of interest
6. Save intermediate and final results
7. Visualize with and without normals

## Best Parameters
- Voxel size: 0.02
- Statistical: `nb_neighbors=20`, `std_ratio=2.0`
- Radius (optional): `nb_points=...`, `radius=...`
- (replace with your own best values)

## Results
- Before / After cleaning: `[add screenshots]`
- Point cloud with normals: `results/with_normals/`
- Cropped ROI: `results/cropped/`

## How to run
```bash
python day5_full_pipeline.py
```

## What I learned
- How to build a reusable cleaning pipeline (`clean_point_cloud`)
- The importance of normals for later stages (ICP)
- How to prepare clean point clouds for registration

---

# Day 6 — Final Packaging and ICP Preparation

## What I did
- Cleaned up the code and folder structure; removed temporary and failed files
- Prepared visuals: 2–3 before/after comparisons, normals visualization, cropped ROI, optional GIF
- Wrote the final README
- Did a self-check and a light introduction to ICP

## Self-check
1. Can I explain the whole cleaning pipeline in 2 minutes?
2. Do I understand why normals are needed?
3. Can I reuse my script on a new point cloud?
4. Am I ready for ICP?

## Data
- Source: `[Open3D demo / public dataset / own]`
- Notes: real-world noisy data where possible

## How to run
```bash
pip install open3d
python day5_full_pipeline.py
```

## What I learned
- How to combine filtering methods effectively
- The importance of normals for ICP and surface reconstruction
- How to prepare clean point clouds for registration and reconstruction
- ICP in simple terms: iteratively aligning one point cloud to another by minimizing the distance between corresponding points

## Next
Week 4 — ICP Registration
