# Day 2 — Calibration Quality Analysis

## Calibration Quality

The initial calibration was performed using 23 chessboard images.

The reprojection error was calculated separately for every image. Some images had a noticeably higher error than the others, so they were selected as problematic frames.

The four images with the highest reprojection error were removed from the final calibration.

* Initial reprojection error: **1.43 px**
* Final reprojection error: **0.96 px**
* Initial number of images: **23**
* Final number of images used: **19**
* Images removed due to high error: **4**
* Improvement: **32.94%**
* Final RMS calibration error: **1.32 px**

The mean reprojection error decreased from 1.43 px to 0.96 px after removing the four images with the highest errors.

## Reprojection Error Analysis

The reprojection error was calculated for every calibration image.

The images with the highest initial errors were:

| Image                                                                                        | Initial error |
| -------------------------------------------------------------------------------------------- | ------------: |
| HNLhtJgvYQcRjGTLnMwH5Gm1Br1oC6sUNR2VLuvJ0sYU0ffBRF6RnyqM_69MC7ArBdg834OeJJ1DkY0U6b92x1sK.jpg |       3.25 px |
| xslDQjoye-Xv2wCJz_ZuMA8HrtlR0OZ7Issu0S5_RM5Kp6Tb7kJjve9UBOpL1W7liGOD1B_QcYm_YTfZF8kPFco-.jpg |       3.06 px |
| 8G_75Du01JC8ahHEVDkIXONKf45X2h4RYoZoyij_yW1Ac38yHQ5XZFlpdM1mJL077TFFKoaCpMNmGHb7HwZGNvK6.jpg |       2.83 px |
| wmrtKLb4C778gfd1ZSisRi5sX6uk1D5a1Qo0JcLOFhO_smYlPI02lD9mQ8so2uVAgtubVsH5Cvoxc5ZOPTM0QN-2.jpg |       2.70 px |

These four images were removed from the final calibration because they had the highest reprojection errors.

Other images also had relatively high errors, for example:

* RKSjDE0_4ph-47eGDKW7T1tNRso5G7XejJBCviLn5cL2e3NyS-BN_sdtOF-bG5xDR48whYz5ji1CDkOhJTV76MR1H.jpg — 2.48 px
* uJCS6ZefmQ3j_FKJGUIaXEFj6mLIMUTQIoVevXfIVQmmzPVZrq0lj3geZ9hxNMUCYm-R09C4xgwWlr2fqEjRk3Vp.jpg — 2.35 px
* Bb4BIn-RSGRC2Fk4fGdWdJCBwa21Jo8-TKhDV2LWZ6-NWZXHtuEKQ3NCwI_IAMeyJHtmBsM7d3fVkoa1AEf-xg64.jpg — 2.11 px

These images were kept in the final calibration.

## Good and Problematic Images

Most of the calibration images produced a relatively low reprojection error.

The lowest initial errors were:

* ePbyuZP3H8wp9zoulq6vDmVVtL8q01XpMkhRMt1e_QEhcwCwmB57DwIg8N5BgpsCn3nQkjEj2rk7fjZFNN18bxLr.jpg — 0.39 px
* dgVRTwMRZg4YqskRYU4pdovuiMA1WL05Hp-gvL24hl7DEs2eMMMEm8QM3clJnI46a_F6QWscuCutdP4csI9yVrp1.jpg — 0.47 px
* mId2PTPsV2k0ifhyXhahaK8QnDS1IZRPzHI22IJenpLv3niOICHSG_eH0zyt1lHvJUGJqo83brtdddzOj9xjkLPt.jpg — 0.47 px
* UdsqP3jQ2Zxorps0WzXRZ5GdPcAqV_bEdgW9YrRox7hQED-WWe13sndRdMF4K_lkX9tBntdGU8nrnXw9SnPPjCmo.jpg — 0.49 px

These images gave stable corner detection and low reprojection error.

The problematic images were selected using the reprojection error. A high error can indicate that the chessboard was captured from a difficult angle, had blur or lighting problems, or that the detected corner positions were less accurate.

## Improved Calibration

After removing four images with the highest reprojection errors, the calibration was performed again using 19 images.

### Before removing images

* Images: **23**
* Mean reprojection error: **1.43 px**
* RMS calibration error: **1.97 px**

### After removing images

* Images: **19**
* Mean reprojection error: **0.96 px**
* RMS calibration error: **1.32 px**

The mean reprojection error decreased by **0.47 px**, which corresponds to an improvement of **32.94%**.

The final calibration parameters were calculated using the remaining 19 images.

## Capture Conditions

The calibration images were taken using a real smartphone camera.

* Real smartphone photos
* Indoor lighting, no studio setup
* Different angles, distances and positions
* Chessboard was captured from different viewpoints
* 23 calibration images were initially used
* 19 images were used for the final calibration

## Test Images

The final calibration was also tested on new ordinary images instead of only using the chessboard calibration images.

Four test images were processed with the final camera parameters.

The results were saved as Original | Undistorted comparisons:

* `comparison_1.jpg`
* `comparison_2.jpg`
* `comparison_3.jpg`
* `comparison_4.jpg`

These images are stored in:

```text
results/before_after/
```

The purpose of this test was to check whether the final distortion parameters work on normal images.

## Saved Parameters

The final camera parameters are stored in:

```text
camera_params.yaml
```

The file contains:

* Intrinsic matrix K
* Distortion coefficients
* Image size
* Final reprojection error
* Number of images used for the final calibration

The final parameters were calculated after removing the four images with the highest reprojection errors.

## Axes Visualization

The final camera parameters were also used to visualize 3D coordinate axes on a calibration image.

The result is saved as:

```text
results/axes_visualization.jpg
```

This helps check whether the estimated camera parameters can be used to project 3D coordinates onto the image.

## Output Files

The main results of Day 2 are:

```text
day2_calibration_quality/
├── calibration_images/
├── results/
│   ├── before_after/
│   │   ├── comparison_1.jpg
│   │   ├── comparison_2.jpg
│   │   ├── comparison_3.jpg
│   │   └── comparison_4.jpg
│   └── axes_visualization.jpg
├── camera_params.yaml
├── day2_calibration_quality.py
└── README.md
```

## Conclusion

The initial calibration used 23 images and had a mean reprojection error of 1.43 px.

The reprojection error was calculated for every image. Four images with the highest errors were removed and the calibration was performed again using 19 images.

The final mean reprojection error was **0.96 px**, compared with **1.43 px** before filtering.

The calibration improved by **32.94%** after removing the four highest-error images.

The final camera parameters were saved in `camera_params.yaml`.

The final calibration was also tested on four new ordinary images using Original | Undistorted comparisons.
