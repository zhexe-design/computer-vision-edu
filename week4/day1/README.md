# Week 4 — Day 1: ICP Registration & CNN Training

The first day of Week 4 introduces **Iterative Closest Point (ICP)** for 3D point cloud registration and revisits Convolutional Neural Networks (CNNs) using PyTorch.

The goal is to understand the mathematical foundations of point cloud alignment, experiment with Open3D registration methods, and implement a reproducible CNN training pipeline with a separate validation dataset.

## Learning Objectives

- Understand the ICP algorithm and its iterative optimization process.
- Learn how point correspondences are established.
- Compare Point-to-Point and Point-to-Plane ICP.
- Understand registration evaluation metrics: Fitness and Inlier RMSE.
- Analyze how correspondence distance thresholds affect registration.
- Build a CNN training pipeline using PyTorch.
- Introduce a separate validation dataset to monitor overfitting.
- Save model checkpoints, training metrics, and visualization plots.

---

## Part 1 — ICP Registration with Open3D

### 1. Dataset

The experiments use the built-in Open3D demonstration dataset:

`DemoICPPointClouds`

It contains two 3D point clouds:

- Source — the point cloud being transformed.
- Target — the reference point cloud.

The dataset is downloaded automatically by Open3D.

### 2. Registration Methods

Two ICP algorithms are evaluated.

**Point-to-Point ICP**

Minimizes the Euclidean distance between corresponding points.

- Does not require surface normals.
- Uses point-to-point correspondences.
- Provides a basic registration approach.

**Point-to-Plane ICP**

Minimizes the distance between source points and tangent planes of the target surface.

- Requires surface normals.
- Often converges faster on smooth surfaces.
- Can provide better alignment for structured surfaces.

### 3. Experimental Parameters

Maximum ICP iterations: `200`

Correspondence distance thresholds:

| Threshold | Description |
|---|---|
| 0.01 | Very restrictive correspondence distance |
| 0.02 | Restrictive correspondence distance |
| 0.05 | Moderate correspondence distance |
| 0.10 | More permissive correspondence distance |

The threshold is expressed in the coordinate units of the point cloud.

### 4. Evaluation Metrics

**Fitness**

Measures the proportion of source points that have valid correspondences within the specified distance threshold.

Higher values generally indicate greater overlap.

**Inlier RMSE**

Measures the root mean square distance between accepted correspondences.

Lower values indicate smaller registration errors.

Both metrics must be considered together when evaluating registration quality.

### 5. Experiment Results

The following graphs are generated:

- `day1_fitness.png` — Fitness comparison across different thresholds.
- `day1_rmse.png` — Inlier RMSE comparison across different thresholds.

The registration results are saved as PLY point clouds for further visualization.

### 6. ICP Results

| Method | Threshold | Fitness | Inlier RMSE |
|---|---:|---:|---:|
| Initial | 0.01 | — | — |
| Initial | 0.02 | — | — |
| Initial | 0.05 | — | — |
| Initial | 0.10 | — | — |
| Point-to-Point | 0.05 | — | — |
| Point-to-Plane | 0.05 | — | — |

*Actual experimental values will be added after execution.*

---

## Part 2 — CIFAR-10 CNN with PyTorch

### 1. Dataset

The project uses the CIFAR-10 image classification dataset.

The dataset contains 60,000 color images divided into 10 classes.

| Dataset | Images | Purpose |
|---|---:|---|
| Training | 45,000 | Model optimization |
| Validation | 5,000 | Monitoring generalization |
| Testing | 10,000 | Final model evaluation |

The original CIFAR-10 training dataset is split reproducibly using a fixed random seed (`42`).

The test dataset remains separate throughout training.

### 2. CNN Architecture

The implemented CNN contains three convolutional blocks.

| Component | Configuration |
|---|---|
| Input | 3 × 32 × 32 |
| Block 1 | 64 convolutional filters |
| Block 2 | 128 convolutional filters |
| Block 3 | 256 convolutional filters |
| Normalization | Batch Normalization |
| Activation | ReLU |
| Pooling | Max Pooling |
| Regularization | Dropout |
| Classifier | Fully Connected Layers |
| Output | 10 classes |

### 3. Training Configuration

| Parameter | Value |
|---|---|
| Framework | PyTorch |
| Optimizer | Adam |
| Learning Rate | 0.001 |
| Weight Decay | 0.0001 |
| Batch Size | 128 |
| Epochs | 25 |
| Scheduler | CosineAnnealingLR |
| Loss Function | CrossEntropyLoss |
| Random Seed | 42 |

### 4. Data Augmentation

Training images are augmented using:

- Random Cropping with padding.
- Random Horizontal Flipping.
- Normalization using CIFAR-10 channel statistics.

Validation and test images are normalized without random augmentation.

### 5. Model Evaluation

The training pipeline tracks four metrics after every epoch:

- Training Loss.
- Training Accuracy.
- Validation Loss.
- Validation Accuracy.

The best model checkpoint is selected based on validation accuracy.

After training, the saved best checkpoint is evaluated on the independent test dataset.

**Target:** 70%+ test accuracy.

This is an experimental target, not a guaranteed result.

### 6. Training Visualizations

The following graphs are generated:

- `loss.png` — Training and validation loss.
- `accuracy.png` — Training and validation accuracy.

These graphs help identify:

- Overfitting.
- Underfitting.
- Training convergence.
- Differences between training and validation performance.

### 7. CNN Results

| Metric | Result |
|---|---:|
| Best Epoch | — |
| Best Validation Accuracy | — |
| Final Test Accuracy | — |
| Final Test Loss | — |

*Actual values will be added after training.*

---

## Project Structure

```text
computer-vision-edu/
│
├── week4/
│   ├── day1_icp_tutorial.py
│   ├── day1_cifar10.py
│   └── README.md
│
├── results/
│   ├── icp/
│   │   ├── point_to_point_aligned.ply
│   │   ├── point_to_point_target.ply
│   │   ├── point_to_plane_aligned.ply
│   │   └── point_to_plane_target.ply
│   │
│   ├── cifar10/
│   │   ├── models/
│   │   │   └── best_cifar10_cnn.pth
│   │   ├── metrics/
│   │   │   ├── training_history.csv
│   │   │   └── test_metrics.json
│   │   └── plots/
│   │       ├── loss.png
│   │       └── accuracy.png
│   │
│   ├── metrics/
│   │   ├── day1_icp_metrics.csv
│   │   └── day1_icp_metrics.json
│   │
│   └── plots/
│       ├── day1_fitness.png
│       └── day1_rmse.png
│
└── data/
    └── cifar10/
```

## Technologies

- Python
- Open3D
- NumPy
- SciPy
- Matplotlib
- PyTorch
- Torchvision
- CIFAR-10

## Key Takeaways

1. ICP is an iterative local optimization algorithm that requires a reasonable initial alignment.
2. Point-to-Point and Point-to-Plane ICP use different distance minimization strategies.
3. Correspondence distance thresholds directly influence registration fitness and RMSE.
4. Fitness and Inlier RMSE should be interpreted together.
5. Validation data is essential for monitoring generalization and identifying overfitting.
6. A separate test dataset provides an independent evaluation of the trained CNN.

## Next Steps

- **Day 2:** Implement a custom ICP algorithm using NumPy and SciPy, including nearest-neighbor search and SVD-based rigid transformation estimation.
- **Day 3:** Compare custom ICP against Open3D and investigate registration performance.
- **Day 4:** Register personal smartphone scans.
- **Day 5:** Complete experiments, comparison tables, and visualization outputs.