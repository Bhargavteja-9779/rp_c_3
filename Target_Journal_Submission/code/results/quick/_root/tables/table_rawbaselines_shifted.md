**Table: comparison with Arsenal and QUANT on the official split — shifted**

*Metric: nll* (n = 6 datasets)

| Method | Median | Mean | Mean rank | W/D/L (LOOM vs method) | Median Δ (method − LOOM) [95% CI] | r_rb | p (Holm) |
|---|---|---|---|---|---|---|---|
| LOOM | 0.379 | 0.396 | 1.50 | – | –  | – | – |
| QUANT | 0.389 | 0.382 | 1.83 | 4.0/0.0/2.0 | 0.013 [-0.092, 0.037] | -0.05 | 1.000 |
| PreVal | 2.764 | 3.602 | 3.17 | 5.0/0.0/1.0 | 2.419 [0.008, 7.191] | 0.90 | 0.125 |
| Arsenal | 3.509 | 4.032 | 3.50 | 6.0/0.0/0.0 | 3.151 [2.649, 5.108] | 1.00 | 0.094 |

*Metric: ece* (n = 6 datasets)

| Method | Median | Mean | Mean rank | W/D/L (LOOM vs method) | Median Δ (method − LOOM) [95% CI] | r_rb | p (Holm) |
|---|---|---|---|---|---|---|---|
| PreVal | 0.123 | 0.142 | 2.17 | 2.0/0.0/4.0 | -0.010 [-0.050, 0.036] | -0.33 | 1.000 |
| LOOM | 0.142 | 0.150 | 2.33 | – | –  | – | – |
| Arsenal | 0.139 | 0.152 | 2.50 | 4.0/0.0/2.0 | 0.029 [-0.075, 0.052] | 0.24 | 1.000 |
| QUANT | 0.151 | 0.160 | 3.00 | 4.0/0.0/2.0 | 0.015 [-0.041, 0.057] | 0.33 | 1.000 |

*Metric: acc* (n = 6 datasets)

| Method | Median | Mean | Mean rank | W/D/L (LOOM vs method) | Median Δ (method − LOOM) [95% CI] | r_rb | p (Holm) |
|---|---|---|---|---|---|---|---|
| QUANT | 0.854 | 0.854 | 2.00 | 2.0/0.0/4.0 | 0.010 [-0.006, 0.055] | 0.62 | 0.656 |
| LOOM | 0.847 | 0.835 | 2.25 | – | –  | – | – |
| PreVal | 0.841 | 0.822 | 2.42 | 2.0/3.0/1.0 | 0.000 [-0.039, 0.001] | -0.67 | 0.656 |
| Arsenal | 0.843 | 0.831 | 3.33 | 5.0/0.0/1.0 | -0.004 [-0.008, 0.002] | -0.62 | 0.656 |

*Metric: shift_auroc* (n = 6 datasets)

| Method | Median | Mean | Mean rank | W/D/L (LOOM vs method) | Median Δ (method − LOOM) [95% CI] | r_rb | p (Holm) |
|---|---|---|---|---|---|---|---|
| LOOM | 0.769 | 0.781 | 1.00 | – | –  | – | – |
| QUANT | 0.724 | 0.729 | 2.00 | 6.0/0.0/0.0 | -0.055 [-0.086, -0.016] | -1.00 | 0.094 |
| Arsenal | 0.548 | 0.550 | 3.33 | 6.0/0.0/0.0 | -0.223 [-0.337, -0.134] | -1.00 | 0.094 |
| PreVal | 0.523 | 0.536 | 3.67 | 6.0/0.0/0.0 | -0.245 [-0.390, -0.101] | -1.00 | 0.094 |
