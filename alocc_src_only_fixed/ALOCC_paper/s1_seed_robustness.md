# [A4-SEED] 3-seed robustness sweep (4 configs x 3 seeds = 12 runs)

**Configs** = C-2 per-class winners; **seeds** = {42, 1337, 2026}; seed=42 is the historical anchor.

**Clean-pass** = redline fallback NOT triggered AND ssim_oc<=0.15 AND raw_auc>=0.60.


## Per-run detail

| class | rank | dropout | seed | best_ep | auc | raw_auc | ssim_ic | ssim_oc | rl_fb | clean |
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| 0 | 8 | 0.3 | 42 | 2 | 0.7514 | 0.8032 | 0.3187 | 0.1479 | N | Y |
| 0 | 8 | 0.3 | 1337 | 2 | 0.9517 | 0.8913 | 0.6349 | 0.3773 | Y | N |
| 0 | 8 | 0.3 | 2026 | 2 | 0.6204 | 0.5974 | 0.5204 | 0.2467 | Y | N |
| 3 | 8 | 0.3 | 42 | 3 | 0.6915 | 0.8238 | 0.2833 | 0.1401 | N | Y |
| 3 | 8 | 0.3 | 1337 | 1 | 0.7491 | 0.5556 | 0.2257 | 0.0762 | Y | N |
| 3 | 8 | 0.3 | 2026 | 5 | 0.7931 | 0.5863 | 0.4823 | 0.3180 | Y | N |
| 7 | 4 | 0.3 | 42 | 1 | 0.8869 | 0.8921 | 0.2409 | 0.1155 | N | Y |
| 7 | 4 | 0.3 | 1337 | 2 | 0.3866 | 0.4706 | 0.3913 | 0.2459 | Y | N |
| 7 | 4 | 0.3 | 2026 | 1 | 0.7842 | 0.6306 | 0.2618 | 0.1322 | N | Y |
| 7 | 4 | 0.5 | 42 | 1 | 0.6316 | 0.7667 | 0.2407 | 0.1123 | N | Y |
| 7 | 4 | 0.5 | 1337 | 1 | 0.8545 | 0.8926 | 0.2545 | 0.1415 | N | Y |
| 7 | 4 | 0.5 | 2026 | 1 | 0.6705 | 0.6354 | 0.1475 | 0.0884 | N | Y |

## Per-config aggregate

| class | rank | dropout | N | clean/N | raw_auc mean±std | ssim_oc mean±std | auc mean±std |
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| 0 | 8 | 0.3 | 3 | 1/3 | 0.7640 ± 0.1509 | 0.2573 ± 0.1150 | 0.7745 ± 0.1668 |
| 3 | 8 | 0.3 | 3 | 1/3 | 0.6552 ± 0.1468 | 0.1781 ± 0.1253 | 0.7446 ± 0.0509 |
| 7 | 4 | 0.3 | 3 | 2/3 | 0.6645 ± 0.2128 | 0.1646 ± 0.0710 | 0.6859 ± 0.2642 |
| 7 | 4 | 0.5 | 3 | 3/3 | 0.7649 ± 0.1286 | 0.1141 ± 0.0266 | 0.7189 ± 0.1191 |

## Headline

- total runs: **12**
- clean-pass: **7/12**

## Interpretation

Baseline C-2 claim: at seed=42 implicit, the 4 winners all pass redline (3 classes recovered).
Seed-42 column below should reproduce that (modulo code-drift).
Other seeds show **true variance** of the per-class rank choice: if clean-pass << 1 on them,
the C-2 'per-class rank tuning => 10/10' claim is a single-seed artefact and must be downgraded.
