**Table: comparison with Arsenal and QUANT on the official split — shifted**

*Metric: nll* (n = 6 datasets; methods with fewer datasets show their n)

| Method | n | Median | Mean | Mean rank | W/D/L (LOOM vs method) | Median Δ (method − LOOM) [95% CI] | r_rb | p (Holm) |
|---|---|---|---|---|---|---|---|---|
| LOOM | 6 | 0.379 | 0.396 | 1.50 | – | –  | – | – |
| QUANT | 6 | 0.389 | 0.382 | 1.83 | 4.0/0.0/2.0 | 0.013 [-0.092, 0.037] | -0.05 | 1.000 |
| PreVal | 6 | 2.764 | 3.602 | 2.83 | 5.0/0.0/1.0 | 2.419 [0.008, 7.191] | 0.90 | 0.125 |
| Arsenal | 6 | 5.341 | 6.175 | 3.83 | 6.0/0.0/0.0 | 4.910 [3.989, 8.438] | 1.00 | 0.094 |

*Metric: ece* (n = 6 datasets; methods with fewer datasets show their n)

| Method | n | Median | Mean | Mean rank | W/D/L (LOOM vs method) | Median Δ (method − LOOM) [95% CI] | r_rb | p (Holm) |
|---|---|---|---|---|---|---|---|---|
| PreVal | 6 | 0.123 | 0.142 | 1.67 | 2.0/0.0/4.0 | -0.010 [-0.050, 0.036] | -0.33 | 1.000 |
| LOOM | 6 | 0.142 | 0.150 | 2.17 | – | –  | – | – |
| QUANT | 6 | 0.151 | 0.160 | 2.33 | 4.0/0.0/2.0 | 0.015 [-0.041, 0.057] | 0.33 | 1.000 |
| Arsenal | 6 | 0.229 | 0.232 | 3.83 | 5.0/0.0/1.0 | 0.088 [0.011, 0.146] | 0.90 | 0.188 |

*Metric: acc* (n = 6 datasets; methods with fewer datasets show their n)

| Method | n | Median | Mean | Mean rank | W/D/L (LOOM vs method) | Median Δ (method − LOOM) [95% CI] | r_rb | p (Holm) |
|---|---|---|---|---|---|---|---|---|
| QUANT | 6 | 0.854 | 0.854 | 1.67 | 2.0/0.0/4.0 | 0.010 [-0.006, 0.055] | 0.62 | 0.438 |
| LOOM | 6 | 0.847 | 0.835 | 2.08 | – | –  | – | – |
| PreVal | 6 | 0.841 | 0.822 | 2.25 | 2.0/3.0/1.0 | 0.000 [-0.039, 0.001] | -0.67 | 0.500 |
| Arsenal | 6 | 0.740 | 0.736 | 4.00 | 6.0/0.0/0.0 | -0.110 [-0.118, -0.068] | -1.00 | 0.094 |

*Metric: shift_auroc* (n = 6 datasets; methods with fewer datasets show their n)

| Method | n | Median | Mean | Mean rank | W/D/L (LOOM vs method) | Median Δ (method − LOOM) [95% CI] | r_rb | p (Holm) |
|---|---|---|---|---|---|---|---|---|
| LOOM | 6 | 0.769 | 0.781 | 1.00 | – | –  | – | – |
| QUANT | 6 | 0.724 | 0.729 | 2.00 | 6.0/0.0/0.0 | -0.055 [-0.086, -0.016] | -1.00 | 0.094 |
| Arsenal | 6 | 0.584 | 0.585 | 3.17 | 6.0/0.0/0.0 | -0.191 [-0.301, -0.096] | -1.00 | 0.094 |
| PreVal | 6 | 0.523 | 0.536 | 3.83 | 6.0/0.0/0.0 | -0.245 [-0.390, -0.101] | -1.00 | 0.094 |
