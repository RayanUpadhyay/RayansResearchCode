# Fairness and Dependency Preservation in Synthetic Credit Data

Code for my paper **[Fairness and Dependency Preservation in Synthetic Credit Data](https://nhsjs.com/2026/fairness-and-dependency-preservation-in-synthetic-credit-data/)**, published in *The National High School Journal of Science* (NHSJS), April 2026. [Read the PDF](https://nhsjs.com/wp-content/uploads/2026/07/Fairness-and-Dependency-Preservation-in-Synthetic-Credit-Data.pdf).

## What the study asks

Synthetic data is often used to rebalance datasets before training models, including loan-approval models. But credit features are tangled together: income, age, employment and personal status move with each other, and those links can carry information about protected groups even after sensitive columns are removed. If a synthetic-data method breaks or distorts those links, it can make a model less fair rather than more.

This project trains loan-approval models on the **German Credit dataset** (UCI Statlog) and compares five versions of the training data:

| Method | Type |
|---|---|
| **Baseline** | Original data, no synthetic augmentation |
| **SMOTE** | Oversampling by interpolating minority-class points |
| **CTGAN** | Conditional GAN for tabular data |
| **TVAE** | Variational autoencoder for tabular data |
| **Gaussian Copula** | Statistical model of linear correlations |

Each one is scored on predictive performance, fairness and how well it keeps the original data's feature dependencies.

## Key findings

| Method | ROC AUC | Equalized Odds ↓ | Disparate Impact (→1) | Predictive Parity ↓ | Dependency Preservation |
|---|---|---|---|---|---|
| Baseline | 0.781 ± 0.012 | 0.081 ± 0.007 | 0.912 ± 0.022 | 0.035 ± 0.006 | 1.0000 |
| SMOTE | 0.764 ± 0.018 | 0.118 ± 0.013 | 0.841 ± 0.029 | 0.052 ± 0.010 | 0.5365 |
| CTGAN | 0.751 ± 0.021 | 0.102 ± 0.011 | 0.857 ± 0.026 | 0.045 ± 0.009 | 0.3986 |
| TVAE | n/a | n/a | n/a | n/a | n/a |
| Gaussian Copula | n/a | n/a | n/a | n/a | n/a |

Mean ± standard deviation over 10 runs with different random seeds.

- The **baseline** kept the most stable fairness/utility trade-off.
- **SMOTE** balanced the classes but disrupted dependencies and widened fairness gaps the most.
- **CTGAN** partly preserved dependencies but made fairness metrics unstable across runs.
- **TVAE** and **Gaussian Copula** did not produce evaluable synthetic data in this pipeline (degenerate columns and empty subgroups made every metric undefined), so they are excluded from the comparison.

See the paper for the full discussion and limitations, including the small dataset, the imbalanced baseline versus class-balanced synthetic sets, and known label issues in the UCI encoding.

## How it works

1. **Data:** the German Credit dataset is downloaded straight from the UCI repository. The target is mapped to good credit = 0, bad credit = 1, categorical features are one-hot encoded, and a copy of the raw sensitive columns is kept for fairness analysis.
2. **Synthetic data:** SMOTE (`imbalanced-learn`), CTGAN (`ctgan`), and TVAE and Gaussian Copula (`sdv`) generate class-balanced training sets. CTGAN and TVAE use 300 epochs and a batch size of 500 with otherwise default settings.
3. **Model:** AutoGluon `TabularPredictor` with ROC AUC as the evaluation metric and an 80/20 holdout split. The selected model is XGBoost.
4. **Fairness metrics:** Equalized Odds difference (TPR/FPR gaps), Disparate Impact (ratio of favourable outcomes against a reference group) and Predictive Parity (PPV gaps), computed for attributes such as `personal_status_sex`, `foreign_worker`, `credit_history` and `employment_since`.
5. **Dependency preservation:** Spearman correlation matrices are computed for the original and each synthetic dataset, and the score is `1 − ‖R_orig − R_synth‖_F / ‖R_orig‖_F` (Frobenius norm). 1.0 means the dependency structure is identical.
6. **Robustness:** the pipeline is re-run with standardisation plus small Gaussian noise, tweaked generator hyperparameters and an out-of-sample 80/20 split.

## Repository contents

Everything lives in **`finalCode.ipynb`**. The cells, in order:

| Cell | Purpose |
|---|---|
| 0 | Earlier draft, kept commented out |
| 1 | SMOTE pipeline and fairness metrics |
| 2 | CTGAN pipeline |
| 3 | Baseline (no synthetic data) |
| 4 | TVAE pipeline |
| 5 | Gaussian Copula pipeline |
| 6 | Radar chart for the summary figure |
| 7 | Repeated evaluation across seeds (means and standard deviations) |
| 8 | Summary table for TVAE and Gaussian Copula |
| 9–11 | Dependency-preservation scoring |
| 12 | Robustness and sensitivity check |
| 13 | Helper functions for the correlation and preservation scores |

Later cells read variables created by earlier ones (`X_encoded`, `X_resampled`, `X_synthetic` and so on), so run the notebook top to bottom.

## Running it

Python 3.10+ is recommended; AutoGluon is a large install.

```bash
pip install autogluon.tabular imbalanced-learn ctgan sdv scikit-learn scipy pandas numpy matplotlib seaborn jupyter
jupyter notebook finalCode.ipynb
```

The dataset downloads automatically, so you'll need an internet connection on the first run.

## Citation

```bibtex
@article{upadhyay2026fairness,
  title   = {Fairness and Dependency Preservation in Synthetic Credit Data},
  author  = {Upadhyay, Rayan},
  journal = {The National High School Journal of Science},
  year    = {2026},
  url     = {https://nhsjs.com/2026/fairness-and-dependency-preservation-in-synthetic-credit-data/}
}
```

## Author

Rayan Upadhyay · [rayanupadhyay.com](https://rayanupadhyay.com)
