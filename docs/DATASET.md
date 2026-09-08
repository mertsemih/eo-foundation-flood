# Sen1Floods11

Bonafilia, Tellman, Anderson, Issenberg — *Sen1Floods11: a georeferenced dataset to train and test deep learning flood algorithms for Sentinel-1*, CVPR Workshops 2020. Public bucket: `gs://sen1floods11`.

## What we use

Only the **hand-labeled** subset: 446 chips of 512×512 px at 10 m over 11 flood events (Bolivia, Ghana, India, Mekong, Nigeria, Pakistan, Paraguay, Somalia, Spain, Sri Lanka, USA).

| folder | content | dtype |
|---|---|---|
| `S2Hand` | Sentinel-2 L1C, 13 bands, TOA reflectance × 10 000 | uint16 |
| `S1Hand` | Sentinel-1 GRD, VV and VH in dB | float32 |
| `LabelHand` | `-1` no data / cloud, `0` no water, `1` water | int16 |
| `JRCWaterHand` | JRC permanent-water mask (optional, for permanent vs flood water analysis) | |
| `S1OtsuLabelHand` | weak labels from Otsu thresholding (optional, weak-supervision ablation) | |

Official splits (`v1.1/splits/flood_handlabeled/`):

| split | chips | note |
|---|---|---|
| train | 252 | |
| valid | 89 | model selection only |
| test | 90 | in-distribution test |
| bolivia | 15 | **held-out region**, our OOD test |

Counts are from the dataset paper; verify after download with `ls LabelHand | wc -l` and the CSV lengths.

## Band mapping for Prithvi-EO

Prithvi-EO was pretrained on six HLS bands. Sentinel-2 equivalents, in the order the model expects:

| Prithvi input | HLS band | Sentinel-2 band | index in 13-band stack |
|---|---|---|---|
| Blue | B02 | B2 | 1 |
| Green | B03 | B3 | 2 |
| Red | B04 | B4 | 3 |
| NIR narrow | B8A | B8A | 8 |
| SWIR 1 | B11 | B11 | 11 |
| SWIR 2 | B12 | B12 | 12 |

`PRITHVI_BANDS` in `eoflood.data` encodes exactly this. U-Net baselines use the same six bands by default so the comparison isolates *pretraining*, not *input information*. A 13-band U-Net and an S1+S2 U-Net are optional extra rows in the results table.

## Normalization

Per-band mean / std over the training split, computed once with `scripts/compute_stats.py`. Prithvi was pretrained on reflectance-scaled HLS with its own statistics; we deliberately re-standardize on Sen1Floods11 rather than reuse those, and note it in the paper. (Ablation idea: Prithvi's original stats vs dataset stats.)

## Known quirks

- Some S2 chips contain NaN / nodata pixels; the loader maps them to 0 after normalization is *not* applied, i.e. before standardization. Watch for chips where this dominates.
- Label `-1` also marks clouds. It is excluded from loss and metrics (`ignore_index`).
- Class imbalance: water is roughly 10–15 % of valid pixels overall but varies wildly per chip. `train.class_weights` exists for a weighted-CE ablation; the primary metric (water IoU) is imbalance-aware anyway.
