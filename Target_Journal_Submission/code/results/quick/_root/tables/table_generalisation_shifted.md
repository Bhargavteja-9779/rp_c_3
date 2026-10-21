**Table: other random-kernel extractors — shifted**

*Metric: nll* (n = 6 datasets; methods with fewer datasets show their n)

| Method | n | Median | Mean | Mean rank | W/D/L (LOOM vs method) | Median Δ (method − LOOM) [95% CI] | r_rb | p (Holm) |
|---|---|---|---|---|---|---|---|---|
| LOOM | 6 | 0.514 | 0.542 | 1.00 | – | –  | – | – |
| Ridge-softmax | 6 | 0.580 | 1.588 | 2.17 | 6.0/0.0/0.0 | 0.076 [0.037, 3.026] | 1.00 | 0.094 |
| Holdout-TS | 6 | 4.133 | 4.515 | 3.17 | 6.0/0.0/0.0 | 3.643 [0.192, 8.086] | 1.00 | 0.094 |
| PreVal | 6 | 5.805 | 5.783 | 3.67 | 6.0/0.0/0.0 | 5.254 [0.255, 10.215] | 1.00 | 0.094 |

*Metric: ece* (n = 6 datasets; methods with fewer datasets show their n)

| Method | n | Median | Mean | Mean rank | W/D/L (LOOM vs method) | Median Δ (method − LOOM) [95% CI] | r_rb | p (Holm) |
|---|---|---|---|---|---|---|---|---|
| LOOM | 6 | 0.155 | 0.176 | 1.50 | – | –  | – | – |
| Ridge-softmax | 6 | 0.196 | 0.216 | 2.67 | 5.0/0.0/1.0 | 0.034 [0.008, 0.079] | 0.90 | 0.188 |
| PreVal | 6 | 0.262 | 0.247 | 2.83 | 5.0/0.0/1.0 | 0.058 [0.002, 0.152] | 0.90 | 0.188 |
| Holdout-TS | 6 | 0.230 | 0.244 | 3.00 | 5.0/0.0/1.0 | 0.063 [0.018, 0.123] | 0.90 | 0.188 |

*Metric: acc* (n = 6 datasets; methods with fewer datasets show their n)

| Method | n | Median | Mean | Mean rank | W/D/L (LOOM vs method) | Median Δ (method − LOOM) [95% CI] | r_rb | p (Holm) |
|---|---|---|---|---|---|---|---|---|
| Ridge-softmax | 6 | 0.726 | 0.727 | 2.08 | 1.0/4.0/1.0 | 0.000 [-0.000, 0.010] | 0.33 | 1.000 |
| Holdout-TS | 6 | 0.726 | 0.730 | 2.25 | 2.0/1.0/3.0 | 0.001 [-0.015, 0.033] | 0.20 | 1.000 |
| LOOM | 6 | 0.726 | 0.724 | 2.33 | – | –  | – | – |
| PreVal | 6 | 0.712 | 0.719 | 3.33 | 3.0/3.0/0.0 | -0.000 [-0.016, 0.000] | -1.00 | 0.750 |
