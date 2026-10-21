**Table: comparison with Arsenal and QUANT on the official split — clean**

*Metric: nll* (n = 6 datasets; methods with fewer datasets show their n)

| Method | n | Median | Mean | Mean rank | W/D/L (LOOM vs method) | Median Δ (method − LOOM) [95% CI] | r_rb | p (Holm) |
|---|---|---|---|---|---|---|---|---|
| PreVal | 6 | 0.133 | 0.136 | 1.92 | 2.0/0.0/4.0 | -0.015 [-0.076, 0.067] | -0.24 | 0.875 |
| LOOM | 6 | 0.110 | 0.144 | 2.33 | – | –  | – | – |
| Arsenal | 6 | 0.507 | 0.954 | 2.58 | 3.0/0.0/3.0 | 0.406 [-0.069, 2.095] | 0.43 | 0.875 |
| QUANT | 6 | 0.176 | 0.194 | 3.17 | 5.0/0.0/1.0 | 0.043 [0.012, 0.095] | 0.90 | 0.188 |

*Metric: ece* (n = 6 datasets; methods with fewer datasets show their n)

| Method | n | Median | Mean | Mean rank | W/D/L (LOOM vs method) | Median Δ (method − LOOM) [95% CI] | r_rb | p (Holm) |
|---|---|---|---|---|---|---|---|---|
| PreVal | 6 | 0.009 | 0.018 | 1.42 | 0.0/0.0/6.0 | -0.054 [-0.076, -0.036] | -1.00 | 0.094 |
| Arsenal | 6 | 0.018 | 0.036 | 1.58 | 0.0/0.0/6.0 | -0.030 [-0.074, -0.008] | -1.00 | 0.094 |
| LOOM | 6 | 0.075 | 0.073 | 3.50 | – | –  | – | – |
| QUANT | 6 | 0.101 | 0.103 | 3.50 | 3.0/0.0/3.0 | 0.026 [-0.008, 0.072] | 0.43 | 0.438 |

*Metric: acc* (n = 6 datasets; methods with fewer datasets show their n)

| Method | n | Median | Mean | Mean rank | W/D/L (LOOM vs method) | Median Δ (method − LOOM) [95% CI] | r_rb | p (Holm) |
|---|---|---|---|---|---|---|---|---|
| Arsenal | 6 | 0.980 | 0.959 | 2.00 | 1.0/4.0/1.0 | 0.000 [-0.016, 0.002] | -0.33 | 1.000 |
| LOOM | 6 | 0.978 | 0.964 | 2.17 | – | –  | – | – |
| PreVal | 6 | 0.978 | 0.964 | 2.58 | 1.0/5.0/0.0 | 0.000 [-0.001, 0.000] | -1.00 | 1.000 |
| QUANT | 6 | 0.976 | 0.956 | 3.25 | 4.0/1.0/1.0 | -0.004 [-0.023, 0.002] | -0.73 | 0.562 |
