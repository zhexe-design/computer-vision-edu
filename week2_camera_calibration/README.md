## Week 2 — Calibration Quality & Application

| Project | Focus | Key result |
|---------|--------|------------|
| day2_quality_analysis | Per-image error + outlier removal | **1.43 → 0.96 px** (−33%) |
| day3_apply_calibration | Undistort real photos with saved K, dist | Before/After on non-chessboard images |

**Takeaway:** not only run `calibrateCamera`, but measure quality, remove bad frames, and reuse parameters.