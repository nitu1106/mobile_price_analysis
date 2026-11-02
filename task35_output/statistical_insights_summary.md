# Statistical insights summary

Data: 2000 rows, 20 numeric features, target `price_range`.

## 1. Key findings

- **Hypothesis tests (Task 31).** 1 of 20 features have a significant AND meaningful ANOVA effect across the target groups (strongest: ram, eta-squared 0.84); 3 differ clearly between the lowest and highest group.
- **Chi-square (Task 32).** 3 features are significantly and non-trivially related to the target when cut into groups (strongest: ram, Cramer's V 0.70).
- **Confidence intervals (Task 33).** The averages are precise for 20 of 20 features (interval within +/-5% of the mean); 2 feature(s) are strongly skewed and 0 outlier-heavy.
- **Regression (Task 34).** The linear model has R-squared 0.92 (cross-validated 0.92); 4 predictor(s) are significant with a non-negligible effect; the highest VIF is 1.7 (pc).
- **Agreement.** 3 feature(s) are supported by at least 3 of the 4 methods: ram, battery_power, px_width. 16 feature(s) have no support from any method.

## 2. Evidence table (top features)

```
               anova_eta2  t_test_d  chi2_cramers_v  reg_std_beta    VIF  evidence_score           verdict
ram                 0.841    -7.044           0.697         0.919  1.008               4  STRONG predictor
battery_power       0.045    -0.637           0.116         0.200  1.010               3  STRONG predictor
px_width            0.033    -0.523           0.102         0.108  1.362               3  STRONG predictor
px_height           0.028    -0.483           0.086         0.110  1.369               1     some evidence
mobile_wt           0.005     0.119           0.029        -0.028  1.005               0       no evidence
int_memory          0.004    -0.156           0.039         0.014  1.009               0       no evidence
n_cores             0.004     0.043           0.000         0.004  1.008               0       no evidence
sc_h                0.003    -0.085           0.026         0.004  1.356               0       no evidence
sc_w                0.003    -0.100           0.046        -0.001  1.354               0       no evidence
talk_time           0.002    -0.090           0.005         0.002  1.011               0       no evidence
m_dep               0.002    -0.041           0.017        -0.003  1.006               0       no evidence
touch_screen        0.002     0.056           0.021        -0.003  1.006               0       no evidence
```

Evidence score = number of methods (ANOVA, t-test, chi-square, regression) that find a significant AND meaningful relation.

## 3. Implications

### Feature engineering
- Interactions are worth testing between the strongest predictors (ram, battery_power, px_width); the others add little to build on (Task 17).
- 16 feature(s) show no evidence of a relation with the target alone (mobile_wt, int_memory, n_cores, sc_h, sc_w, talk_time, m_dep, touch_screen...). Keep them out of the first model; test them again inside interactions/ratios before discarding completely (they can matter in combination).

### Data preprocessing
- SKEWED features (|skew| > 1): fc, three_g. Apply log / Box-Cox / Yeo-Johnson (Task 27); the mean of a skewed column and its t-interval are less reliable.
- SCALING: features have very different units; scale before distance- or gradient-based models (Tasks 14, 26). Tree models do not need it.
- MULTICOLLINEARITY: 0 feature(s) with VIF > 10. No action needed for coefficient stability.
- Fit every scaler / transformer / selector on the training rows only (use a Pipeline) to avoid data leakage.

### Model development
- A plain LINEAR model explains the target with cross-validated R-squared 0.92: a linear/logistic regression is a strong, explainable baseline.
- Start with the 3 strongest predictor(s) (ram, battery_power, px_width) and check by cross-validation whether adding the 'some evidence' features helps (Tasks 20, 22, 25).
- Only a few features carry the signal: expect simple models to perform nearly as well as complex ones; prefer the simple, explainable one.
- Report effect sizes (eta-squared, Cohen's d, Cramer's V, standardised coefficients) next to every p-value: with this many rows almost any difference becomes 'significant'.
- Use confidence intervals (Task 33/36) on the final scores and on the key coefficients: a single accuracy number hides how much it could vary.
- Statistical tests show association, not cause: do not read the coefficients as 'what to change to raise the price'.

## 4. Limits of these findings

- Tests look at one feature at a time (ANOVA, t-test, chi-square) or assume a linear relation (regression); combined or curved effects can be missed.
- p-values depend on sample size; effect sizes (not p-values) decide whether a result matters.
- Many tests were run; q-values (Benjamini-Hochberg) control false alarms but not all of them.
- The target `price_range` is ordered (0-3): the linear model treats it as a number, which is an approximation.