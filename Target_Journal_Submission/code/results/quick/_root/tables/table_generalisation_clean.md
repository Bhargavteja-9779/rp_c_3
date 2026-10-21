**Table: other random-kernel extractors — clean**

*Metric: nll* (n = 6 datasets; methods with fewer datasets show their n)

| Method | n | Median | Mean | Mean rank | W/D/L (LOOM vs method) | Median Δ (method − LOOM) [95% CI] | r_rb | p (Holm) |
|---|---|---|---|---|---|---|---|---|
| PreVal | 6 | 0.126 | 0.130 | 1.83 | 1.0/0.0/5.0 | -0.022 [-0.106, 0.080] | -0.43 | 0.875 |
| LOOM | 6 | 0.131 | 0.146 | 2.33 | – | –  | – | – |
| Holdout-TS | 6 | 0.146 | 0.196 | 2.50 | 3.0/0.0/3.0 | -0.007 [-0.101, 0.257] | -0.14 | 0.875 |
| Ridge-softmax | 6 | 0.205 | 0.219 | 3.33 | 6.0/0.0/0.0 | 0.072 [0.039, 0.108] | 1.00 | 0.094 |

*Metric: ece* (n = 6 datasets; methods with fewer datasets show their n)

| Method | n | Median | Mean | Mean rank | W/D/L (LOOM vs method) | Median Δ (method − LOOM) [95% CI] | r_rb | p (Holm) |
|---|---|---|---|---|---|---|---|---|
| PreVal | 6 | 0.007 | 0.020 | 1.50 | 1.0/0.0/5.0 | -0.059 [-0.100, -0.014] | -0.90 | 0.125 |
| Holdout-TS | 6 | 0.017 | 0.035 | 1.83 | 1.0/0.0/5.0 | -0.036 [-0.100, 0.008] | -0.71 | 0.156 |
| LOOM | 6 | 0.069 | 0.078 | 2.67 | – | –  | – | – |
| Ridge-softmax | 6 | 0.150 | 0.149 | 4.00 | 6.0/0.0/0.0 | 0.072 [0.037, 0.103] | 1.00 | 0.094 |

*Metric: acc* (n = 6 datasets; methods with fewer datasets show their n)

| Method | n | Median | Mean | Mean rank | W/D/L (LOOM vs method) | Median Δ (method − LOOM) [95% CI] | r_rb | p (Holm) |
|---|---|---|---|---|---|---|---|---|
| Ridge-softmax | 6 | 0.983 | 0.971 | 1.92 | 0.0/5.0/1.0 | 0.000 [0.000, 0.001] | 1.00 | 1.000 |
| LOOM | 6 | 0.982 | 0.971 | 2.25 | – | –  | – | – |
| PreVal | 6 | 0.980 | 0.969 | 2.83 | 2.0/4.0/0.0 | 0.000 [-0.006, 0.000] | -1.00 | 1.000 |
| Holdout-TS | 6 | 0.982 | 0.955 | 3.00 | 2.0/4.0/0.0 | 0.000 [-0.048, 0.000] | -1.00 | 1.000 |
