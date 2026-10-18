# Day 2 — Calibration Quality Analysis

Analyze camera calibration quality by calculating reprojection errors for individual images, identifying problematic frames, and comparing calibration results before and after filtering.

The initial calibration used 23 chessboard images. The four images with the highest reprojection errors were removed, and calibration was performed again using the remaining 19 images.

## Calibration Quality

| Metric | Initial Calibration | Final Calibration |
|---|---:|---:|
| Number of images | 23 | 19 |
| Mean reprojection error | 1.43 px | 0.96 px |
| RMS calibration error | 1.97 px | 1.32 px |
| Images removed | 0 | 4 |

**Mean reprojection error improvement: 32.94%.**

The mean reprojection error decreased from 1.43 px to 0.96 px after removing the four images with the highest errors.

## Reprojection Error Analysis

The reprojection error was calculated separately for every calibration image. Images with unusually high errors were reviewed as potentially problematic frames.

### Images Removed from Calibration

The following four images had the highest initial reprojection errors and were excluded from the final calibration.

| Image | Initial Reprojection Error |
|---|---:|
| `HNLhtJgvYQcRjGTLnMwH5Gm1Br1oC6sUNR2VLuvJ0sYU0ffBRF6RnyqM_69MC7ArBdg834OeJJ1DkY0U6b92x1sK.jpg` | 3.25 px |
| `xslDQjoye-Xv2wCJz_ZuMA8HrtlR0OZ7Issu0S5_RM5Kp6Tb7kJjve9UBOpL1W7liGOD1B_QcYm_YTfZF8kPFco-.jpg` | 3.06 px |
| `8G_75Du01JC8ahHEVDkIXONKf45X2h4RYoZoyij_yW1Ac38yHQ5XZFlpdM1mJL077TFFKoaCpMNmGHb7HwZGNvK6.jpg` | 2.83 px |
| `wmrtKLb4C778gfd1ZSisRi5sX6uk1D5a1Qo0JcLOFhO_smYlPI02lD9mQ8so2uVAgtubVsH5Cvoxc5ZOPTM0QN-2.jpg` | 2.70 px |

### Additional High-Error Images

The following images also had relatively high reprojection errors but were retained in the final calibration.

- `RKSjDE0_4ph-47eGDKW7T1tNRso5G7XejJBCviLn5cL2e3NyS-BN_sdtOF-bG5xDR48whYz5ji1CDkOhJTV76MR1H.jpg` — 2.48 px
- `uJCS6ZefmQ3j_FKJGUIaXEFj6mLIMUTQIoVevXfIVQmmzPVZrq0lj3geZ9hxNMUCYm-R09C4xgwWlr2fqEjRk3Vp.jpg` — 2.35 px
- `Bb4BIn-RSGRC2Fk4fGdWdJCBwa21Jo8-TKhDV2LWZ6-NWZXHtuEKQ3NCwI_IAMeyJHtmBsM7d3fVkoa1AEf-xg64.jpg` — 2.11 px

These examples show that images were not removed solely because their errors were relatively high. The four images with the highest errors were selected for exclusion.

### Images with the Lowest Reprojection Errors

The following images produced the lowest initial reprojection errors.

- `ePbyuZP3H8wp9zoulq6vDmVVtL8q01XpMkhRMt1e_QEhcwCwmB57DwIg8N5BgpsCn3nQkjEj2rk7fjZFNN18bxLr.jpg` — 0.39 px
- `dgVRTwMRZg4YqskRYU4pdovuiMA1WL05Hp-gvL24hl7DEs2eMMMEm8QM3clJnI46a_F6QWscuCutdP4csI9yVrp1.jpg` — 0.47 px
- `mId2PTPsV2k0ifhyXhahaK8QnDS1IZRPzHI22IJenpLv3niOICHSG_eH0zyt1lHvJUGJqo83brtdddzOj9xjkLPt.jpg` — 0.47 px
- `UdsqP3jQ2Zxorps0WzXRZ5GdPcAqV_bEdgW9YrRox7hQED-WWe13sndRdMF4K_lkX9tBntdGU8nrnXw9SnPPjCmo.jpg` — 0.49 px

These images provided relatively low reprojection errors, indicating good agreement between the observed chessboard corners and their projected positions.

A high reprojection error can indicate challenging viewing angles, motion blur, lighting problems, or inaccurate corner detection. However, reprojection error alone does not establish the cause of an image's poor performance.

## Improved Calibration

Calibration was performed twice to evaluate the effect of removing the four highest-error images.

### Before Filtering

- Images used: **23**
- Mean reprojection error: **1.43 px**
- RMS calibration error: **1.97 px**

### After Filtering

- Images used: **19**
- Mean reprojection error: **0.96 px**
- RMS calibration error: **1.32 px**

The mean reprojection error decreased by **0.47 px**, corresponding to an improvement of approximately **32.94%**.

The RMS calibration error also decreased, from 1.97 px to 1.32 px.

These results indicate that filtering the four highest-error images improved the calibration metrics for this dataset.

## Capture Conditions

The calibration images were captured using a real smartphone camera under uncontrolled indoor conditions.

- Real smartphone photographs.
- Indoor lighting without a dedicated studio setup.
- Different chessboard angles, distances, and positions.
- Multiple viewpoints of the calibration pattern.
- 23 images used for the initial calibration.
- 19 images used for the final calibration.

## Test Images

The final calibration parameters were also tested on four ordinary photographs that were separate from the chessboard calibration images.

The script generated original-versus-undistorted comparisons:

- `comparison_1.jpg`
- `comparison_2.jpg`
- `comparison_3.jpg`
- `comparison_4.jpg`

The results are stored in:

```text
results