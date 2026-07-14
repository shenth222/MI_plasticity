# A.1 Positive control: criterion vs gain

> gain_log10 = log10(uniform_mse / goodput_mse). Dynamic allocation should
> help (gain>0) once planted_rank > uniform per-module rank (budget/M).

| planted_rank | uni_per_mod | n_seeds | uniform_mse | gora_mse | goodput_mse | oracle_mse | gain_log10 | persistence | planted_recall |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 2 | 3 | 4.230e-07 | 1.379e-07 | 7.678e-08 | 7.503e-07 | 0.741 | 1.000 | 1.000 |
| 2 | 2 | 3 | 5.266e-08 | 3.914e-08 | 3.767e-08 | 5.266e-08 | 0.145 | 1.000 | 1.000 |
| 3 | 2 | 3 | 0.030 | 1.958e-08 | 4.337e-08 | 7.207e-13 | 5.835 | 1.000 | 1.000 |
| 4 | 2 | 3 | 0.046 | 1.661e-08 | 5.150e-08 | 1.019e-14 | 5.954 | 1.000 | 1.000 |
| 6 | 2 | 3 | 0.061 | 4.974e-09 | 8.786e-09 | 9.810e-15 | 6.843 | 1.000 | 1.000 |
| 8 | 2 | 3 | 0.070 | 9.463e-15 | 9.069e-12 | 9.694e-15 | 9.885 | 1.000 | 1.000 |
