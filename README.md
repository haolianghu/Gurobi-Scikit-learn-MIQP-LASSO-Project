# Sparse regression: LASSO and MIQP

The notebook compares indirect feature selection with LASSO against direct
subset selection formulated as a mixed-integer quadratic program. The bundled
`training_data.csv` and `test_data.csv` contain 250 and 50 observations,
respectively, with 50 predictors and a continuous target.

## Reproducible LASSO baseline

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python model_baseline.py
python -m unittest discover -s tests -v
```

The baseline scales predictors within a pipeline, selects regularization
strength using cross-validation on training data only, and reports held-out
MAE, RMSE, R², and number of nonzero coefficients. Optional positional
arguments select alternate training and test files; columns and their order
must match. The script does not use or overwrite the test set for model tuning.

The notebook's MIQP formulation requires Gurobi and a valid license; it is not
reproduced by the LASSO baseline. Its cross-validation should be independently
reviewed before comparing reported results. Neither method establishes causal
effects. The held-out sample is small, so metrics are uncertain and may not
generalize beyond these data.
