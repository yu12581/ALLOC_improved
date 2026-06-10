# [C-2] S1 Rank-Scan Ablation on Failing Classes {0, 3, 7}

**Grid**: rank ∈ {8, 4} × dropout ∈ {0.3, 0.5} × class ∈ {0, 3, 7} = 12 runs  
**Redline**: ssim_oc ≤ 0.15 AND raw_auc ≥ 0.6 (clean, no fallback)

## Full grid

| class | rank | p | best_ep | auc | raw_auc | ssim_ic | ssim_oc | redline |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 0 | 8 | 0.3 | 2 | 0.751 | 0.803 | 0.319 | 0.148 | ✅ |
| 0 | 8 | 0.5 | 3 | 0.586 | 0.491 | 0.703 | 0.448 | ❌ |
| 0 | 4 | 0.3 | 2 | 0.878 | 0.655 | 0.380 | 0.138 | ✅ |
| 0 | 4 | 0.5 | 1 | 0.979 | 0.599 | 0.358 | 0.119 | ❌ |
| 3 | 8 | 0.3 | 3 | 0.692 | 0.824 | 0.283 | 0.140 | ✅ |
| 3 | 8 | 0.5 | 7 | 0.670 | 0.437 | 0.661 | 0.531 | ❌ |
| 3 | 4 | 0.3 | 7 | 0.767 | 0.716 | 0.669 | 0.514 | ❌ |
| 3 | 4 | 0.5 | 1 | 0.766 | 0.694 | 0.174 | 0.074 | ✅ |
| 7 | 8 | 0.3 | 2 | 0.663 | 0.612 | 0.270 | 0.123 | ✅ |
| 7 | 8 | 0.5 | 2 | 0.460 | 0.460 | 0.463 | 0.317 | ❌ |
| 7 | 4 | 0.3 | 1 | 0.887 | 0.892 | 0.241 | 0.116 | ✅ |
| 7 | 4 | 0.5 | 1 | 0.632 | 0.767 | 0.241 | 0.112 | ✅ |

## Per-class best config vs baselines

| class | OFF ssim_oc / raw_auc / rl | S1 r=16 p=0.3 ssim_oc / raw_auc / rl | BEST rank-scan config | BEST ssim_oc / raw_auc / rl |
|:---:|:---:|:---:|:---:|:---:|
| 0 | 0.284 / 0.727 / ❌ | 0.394 / 0.688 / ❌ | r=8 p=0.3 | 0.148 / 0.803 / ✅ |
| 3 | 0.319 / 0.800 / ❌ | 0.366 / 0.687 / ❌ | r=8 p=0.3 | 0.140 / 0.824 / ✅ |
| 7 | 0.153 / 0.927 / ❌ | 0.406 / 0.700 / ❌ | r=4 p=0.3 | 0.116 / 0.892 / ✅ |

**Clean-pass count**: 7/12 runs pass redline without fallback
**Per-class pass**: class 0 = 2/4, class 3 = 2/4, class 7 = 3/4

> Classes previously labeled 'S1 ceiling fail' all have at least one rank/dropout config that passes the redline.
> The prior conclusion ("S1 hits a ceiling for {0,3,7} → need Contractive AE") was driven by a fixed rank=16, not a fundamental limitation of S1. See §3.2 for revised Round 2 framing.