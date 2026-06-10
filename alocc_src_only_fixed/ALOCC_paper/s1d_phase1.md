# [S1D-Phase1] S1 + Distortion Go/No-Go (4 cfg x 3 seed = 12 pairs)

Baseline = `--variant alocc` (S1 only; reused from A4-SEED sweep).

Combo    = `--variant alocc_loss` with `--g-outclass-distortion-scale 0.1`, `--d-outclass-loss-scale 0.1`, `--g-outclass-distortion-margin 0.2`, `--out-per-class-count 300`.

Clean-pass = redline_fallback==N AND ssim_oc<=0.15 AND raw_auc>=0.60.


## Per-pair detail

| class | r | p | seed | who | ep | raw_auc | ssim_ic | ssim_oc | gap | rl_fb | clean |
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| 0 | 8 | 0.3 | 42 | base | 2 | 0.8032 | 0.3187 | 0.1479 | 0.1708 | N | Y |
| 0 | 8 | 0.3 | 42 | combo | 9 | 0.9256 | 0.8065 | 0.5192 | 0.2874 | Y | N |
| 0 | 8 | 0.3 | 1337 | base | 2 | 0.8913 | 0.6349 | 0.3773 | 0.2576 | Y | N |
| 0 | 8 | 0.3 | 1337 | combo | 1 | 0.9779 | 0.3323 | 0.1002 | 0.2320 | N | Y |
| 0 | 8 | 0.3 | 2026 | base | 2 | 0.5974 | 0.5204 | 0.2467 | 0.2737 | Y | N |
| 0 | 8 | 0.3 | 2026 | combo | 1 | 0.9804 | 0.3569 | 0.1298 | 0.2271 | N | Y |
| 3 | 8 | 0.3 | 42 | base | 3 | 0.8238 | 0.2833 | 0.1401 | 0.1431 | N | Y |
| 3 | 8 | 0.3 | 42 | combo | 1 | 0.7049 | 0.1509 | 0.0662 | 0.0847 | N | Y |
| 3 | 8 | 0.3 | 1337 | base | 1 | 0.5556 | 0.2257 | 0.0762 | 0.1495 | Y | N |
| 3 | 8 | 0.3 | 1337 | combo | 1 | 0.8969 | 0.2156 | 0.0754 | 0.1402 | N | Y |
| 3 | 8 | 0.3 | 2026 | base | 5 | 0.5863 | 0.4823 | 0.3180 | 0.1643 | Y | N |
| 3 | 8 | 0.3 | 2026 | combo | 10 | 0.8446 | 0.6635 | 0.5496 | 0.1139 | Y | N |
| 7 | 4 | 0.3 | 42 | base | 1 | 0.8921 | 0.2409 | 0.1155 | 0.1254 | N | Y |
| 7 | 4 | 0.3 | 42 | combo | 1 | 0.8354 | 0.2306 | 0.1071 | 0.1235 | N | Y |
| 7 | 4 | 0.3 | 1337 | base | 2 | 0.4706 | 0.3913 | 0.2459 | 0.1454 | Y | N |
| 7 | 4 | 0.3 | 1337 | combo | 6 | 0.9689 | 0.5844 | 0.3962 | 0.1882 | Y | N |
| 7 | 4 | 0.3 | 2026 | base | 1 | 0.6306 | 0.2618 | 0.1322 | 0.1296 | N | Y |
| 7 | 4 | 0.3 | 2026 | combo | 1 | 0.9346 | 0.2174 | 0.1126 | 0.1047 | N | Y |
| 7 | 4 | 0.5 | 42 | base | 1 | 0.7667 | 0.2407 | 0.1123 | 0.1285 | N | Y |
| 7 | 4 | 0.5 | 42 | combo | 1 | 0.9065 | 0.1364 | 0.0616 | 0.0748 | N | Y |
| 7 | 4 | 0.5 | 1337 | base | 1 | 0.8926 | 0.2545 | 0.1415 | 0.1130 | N | Y |
| 7 | 4 | 0.5 | 1337 | combo | 6 | 0.9383 | 0.6440 | 0.4208 | 0.2232 | Y | N |
| 7 | 4 | 0.5 | 2026 | base | 1 | 0.6354 | 0.1475 | 0.0884 | 0.0591 | N | Y |
| 7 | 4 | 0.5 | 2026 | combo | 10 | 0.9106 | 0.7682 | 0.4943 | 0.2739 | Y | N |

## Per-config paired aggregate

| class | r | p | base clean/3 | combo clean/3 | d_raw_auc | d_ssim_ic | d_ssim_oc | d_gap |
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| 0 | 8 | 0.3 | 1/3 | 2/3 | +0.1973 | +0.0072 | -0.0075 | +0.0148 |
| 3 | 8 | 0.3 | 1/3 | 2/3 | +0.1603 | +0.0129 | +0.0523 | -0.0394 |
| 7 | 4 | 0.3 | 2/3 | 2/3 | +0.2485 | +0.0461 | +0.0407 | +0.0053 |
| 7 | 4 | 0.5 | 3/3 | 1/3 | +0.1536 | +0.3019 | +0.2115 | +0.0905 |

## Headline

- paired runs: **12/12**
- clean-pass  base vs combo: **7/12 -> 7/12**  (delta = +0)
- mean delta raw_auc  : **+0.1899**
- mean delta ssim_ic  : **+0.0920**
- mean delta ssim_oc  : **+0.0742**  (negative = further suppressed)
- mean delta ssim_gap : **+0.0178**  (positive = gap widened)
- fragile-pair clean  : base 2/6 -> combo 4/6

## Go/No-Go gates

- (a) fragile-pair clean rescue (+1 required) : **PASS**  (2/6 -> 4/6)
- (b) mean d_raw_auc >= +0.05                : **PASS**  (+0.1899)
- (c) mean d_ssim_gap >= +0.05               : **FAIL**  (+0.0178)

## Decision: **GO to Phase 2**

