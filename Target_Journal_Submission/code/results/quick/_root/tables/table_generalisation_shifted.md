**Table: other random-kernel extractors — shifted**

*Metric: nll* (n = 6 datasets)

| Method | Median | Mean | Mean rank | W/D/L (LOOM vs method) | Median Δ (method − LOOM) [95% CI] | r_rb | p (Holm) |
|---|---|---|---|---|---|---|---|
| LOOM | 0.391 | 0.419 | 1.00 | – | –  | – | – |
| Ridge-softmax | 0.424 | 1.064 | 2.17 | 6.0/0.0/0.0 | 0.053 [0.034, 1.848] | 1.00 | 0.094 |
| Holdout-TS | 2.339 | 2.981 | 3.17 | 6.0/0.0/0.0 | 1.985 [0.102, 5.599] | 1.00 | 0.094 |
| PreVal | 3.573 | 3.957 | 3.67 | 6.0/0.0/0.0 | 3.159 [0.098, 7.356] | 1.00 | 0.094 |

*Metric: ece* (n = 6 datasets)

| Method | Median | Mean | Mean rank | W/D/L (LOOM vs method) | Median Δ (method − LOOM) [95% CI] | r_rb | p (Holm) |
|---|---|---|---|---|---|---|---|
| LOOM | 0.157 | 0.163 | 2.00 | – | –  | – | – |
| PreVal | 0.158 | 0.165 | 2.33 | 4.0/0.0/2.0 | 0.005 [-0.039, 0.043] | 0.14 | 1.000 |
| Holdout-TS | 0.143 | 0.165 | 2.33 | 3.0/0.0/3.0 | 0.009 [-0.045, 0.045] | 0.05 | 1.000 |
| Ridge-softmax | 0.173 | 0.189 | 3.33 | 5.0/0.0/1.0 | 0.023 [-0.000, 0.056] | 0.81 | 0.281 |

*Metric: acc* (n = 6 datasets)

| Method | Median | Mean | Mean rank | W/D/L (LOOM vs method) | Median Δ (method − LOOM) [95% CI] | r_rb | p (Holm) |
|---|---|---|---|---|---|---|---|
| Ridge-softmax | 0.821 | 0.817 | 1.67 | 0.0/4.0/2.0 | 0.000 [0.000, 0.003] | 1.00 | 0.875 |
| LOOM | 0.821 | 0.815 | 2.25 | – | –  | – | – |
| Holdout-TS | 0.813 | 0.810 | 3.00 | 4.0/0.0/2.0 | -0.003 [-0.025, 0.013] | -0.43 | 0.875 |
| PreVal | 0.813 | 0.804 | 3.08 | 3.0/3.0/0.0 | -0.000 [-0.033, 0.000] | -1.00 | 0.750 |
