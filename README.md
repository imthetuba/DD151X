# DD151X

# How to run  
  # 1. Set up environment
  python3 -m venv .venv
  source .venv/bin/activate
  pip install -r requirements.txt
  brew install libomp  # needed for XGBoost on macOS

  # 2. Run single-seed experiment (40 runs)
  python3 src/run_experiments.py --config configs/experiment_config.yaml

  # 3. Run repeated-seed experiment (400 runs)
  python3 src/run_experiments_repeated.py --config configs/experiment_config.yaml --n-seeds 10 --seed-start 0

  # 4. Run dataset diagnostics
  python3 scripts/generate_modeling_dataset_diagnostics.py

  # 5. Generate figures and LaTeX tables
  python3 scripts/generate_repeated_seed_figures.py

  Steps 2-4 produce the raw results. Step 5 generates the bar charts, ROC curves, feature importance plots,
  confusion matrices, and LaTeX tables from those results.



Degree project at KTH (Industrial Engineering and Management, Computer Science track) focused on ESG-integrated machine learning for corporate credit risk assessment in the Nordic bond market.

## Project Goal

This project evaluates whether adding ESG variables to credit risk models improves predictive performance compared with traditional financial-only approaches.

The modelling setup follows a supervised multi-class classification problem where the target is engineered into three credit risk classes:

- Investment Grade Low Risk
- Investment Grade High Risk
- High Yield

The feature space combines financial factors and ESG metrics to test whether this improves predictive power for credit risk assessment applications.
D
## Research Focus

The project addresses two dimensions:

- Technical: Compare statistical and machine learning models for credit rating prediction using metrics such as F1-score and ROC-AUC.
- Economic/managerial: Assess implications for capital allocation, risk exposure, and institutional feasibility in a Nordic context.

## Data Sources and Current Data Artifacts

Data is collected from Stamdata feeds and merged into a curated modelling table.

For experiment execution, the active runtime dataset is:

- [data/modeling_dataset.csv](data/modeling_dataset.csv)

Legacy/raw/generated artifacts have been moved to:

- [.archive/2026-03-25_cleanup](.archive/2026-03-25_cleanup)
- Data collection and curation script remains available in [download_curated_rated_esg_dataset.py](download_curated_rated_esg_dataset.py)

The current curated CSV contains issuer-level observations with:

- Target-related fields: OrganizationNumber, siI_Rating, siI_RatingNormalized, siI_RatingCompany, ratingDate
- ESG/financial feature columns such as: CarbonTarget, EnterpriseValue, Revenue, BookValueEquity, BookValueDebt, TotalGHGEmission, Scope1, Scope2Location, board composition, climate/sector flags, and related variables.

## Methodological Direction

Models considered in the thesis framework include:

- Logistic Regression
- Naive Bayes
- Random Forest
- Gradient Boosting / XGBoost

Evaluation emphasizes:

- Accuracy
- Precision/Recall/F1
- ROC-AUC
- Explainability via feature importance (and optionally SHAP)

## Step 1 Plan: Build the 3-Class Target Dataset

### Objective

Create a modelling-ready dataset where normalized ratings are mapped into three target classes:

- Investment Grade Low Risk
- Investment Grade High Risk
- High Yield

### Proposed class mapping (initial baseline)

Using siI_RatingNormalized (CQS-like scale):

- 1-2 -> Investment Grade Low Risk
- 3 -> Investment Grade High Risk
- 4-6 -> High Yield

This mapping should be validated against rating-agency conventions used in the sample and adjusted if needed.

### Step 1 tasks

1. Data quality check
- Confirm non-null coverage for siI_RatingNormalized.
- Remove or flag rows missing essential target/feature values.
- Check duplicate issuer-year combinations.

2. Target engineering
- Implement deterministic mapping from siI_RatingNormalized to the three classes.
- Add a new categorical target column, for example risk_class_3.

3. Feature set definition
- Finalize a factor set combining financial and ESG variables.
- Keep transparent naming and a feature dictionary for reproducibility.

4. Dataset output
- Export a clean modelling table to data/ with:
  - ID columns
  - time columns
  - selected factor features
  - engineered 3-class target

5. Initial class balance diagnostics
- Report class counts and class proportions.
- Decide whether class weighting or resampling is needed before model training.

### Step 1 deliverables

- A documented target mapping rule.
- A clean 3-class dataset ready for train/validation split.
- A short data profiling summary (missingness, duplicates, class distribution).

## Why Explainability Is Central

Feature importance is treated as a core output, not only an add-on, because credit risk applications require transparency, accountability, and regulatory interpretability.

In line with prior related work discussed in the report, key influential predictors are expected to include (in reported order):

1. Debt-to-Equity Ratio  
2. Carbon Intensity  
3. ESG Score  
4. Interest Coverage Ratio  
5. Climate Policy Uncertainty  
6. Current Ratio  
7. Renewable Energy Share  
8. ROA

This supports an explainable AI perspective where both financial and ESG drivers are made explicit in model decisions.

## Repository Structure (current)

- [download_curated_rated_esg_dataset.py](download_curated_rated_esg_dataset.py): Pulls Stamdata feeds, matches ratings with ESG rows, exports curated CSV.
- [data](data): Active runtime dataset for experiments.
- [.archive/2026-03-25_cleanup](.archive/2026-03-25_cleanup): Legacy/raw/generated artifacts not needed to run experiments.
- [Report](Report): Thesis manuscript sections (introduction, literature, theory, methodology).
- [README.md](README.md): Project overview and execution plan.

## Reproducing Data Pull and Curation

Set environment variables (in .env or shell):

- STAMDATA_API_URL
- STAMDATA_API_KEY

Run:

```bash
python3 download_curated_rated_esg_dataset.py \
  --coverage-mode max_rows \
  --require-normalized-target \
  --output data/curated_rated_esg_dataset.csv
```

## Step 1 Data cleaning and prep (Completed)

Run the Step 1 preprocessing pipeline:

```bash
python3 step1_build_three_class_dataset.py
```

This script performs all Step 1 tasks:

1. Confirms non-null coverage for `siI_RatingNormalized`.
2. Removes rows missing required target/feature inputs.
3. Checks and resolves duplicate issuer-year rows.
4. Maps normalized ratings to a deterministic 3-class target.
5. Finalizes a transparent financial+ESG factor set.
6. Exports cleaned modelling data and diagnostics.

Generated artifacts (archived):

- [.archive/2026-03-25_cleanup/data/modeling_dataset_step1.csv](.archive/2026-03-25_cleanup/data/modeling_dataset_step1.csv): Step 1 base dataset (pre-Yahoo enrichment).
- [data/modeling_dataset.csv](data/modeling_dataset.csv): Canonical modeling dataset used by experiments.
- [.archive/2026-03-25_cleanup/data/modeling_dataset_step1_diagnostics.json](.archive/2026-03-25_cleanup/data/modeling_dataset_step1_diagnostics.json): Step 1 quality diagnostics.
- [.archive/2026-03-25_cleanup/data/modeling_dataset_diagnostics.json](.archive/2026-03-25_cleanup/data/modeling_dataset_diagnostics.json): Canonical dataset diagnostics snapshot.
- [.archive/2026-03-25_cleanup/data/feature_dictionary_step1.json](.archive/2026-03-25_cleanup/data/feature_dictionary_step1.json): Feature dictionary snapshot (including `yf_*` enriched ratio features).

Current run summary:

- Input rows: 438
- Non-null `siI_RatingNormalized`: 438 (100.0%)
- Rows after required-field filtering: 382
- Duplicate issuer-year rows detected: 0
- Final modelling rows: 382

Class balance (3-class target):

- Investment Grade Low Risk: 188 (49.2%)
- Investment Grade High Risk: 157 (41.1%)
- High Yield: 37 (9.7%)

Imbalance decision:

- Use `class_weight='balanced'` for baseline model training.
- Evaluate oversampling (for example SMOTE) in cross-validated experiments.

## Step 2 Implementation of Models, Training, Evaluation (Completed)

Step 2 is now implemented as a config-driven training pipeline for all required model families and feature views.

### What is implemented

- Models:
  - Logistic Regression
  - Naive Bayes
  - Random Forest
  - XGBoost
- Feature views:
  - `financial_only`
  - `esg_only`
  - `esg_financial`
- Split strategies:
  - Stratified 80/20 split (`random_state=42`)
  - Grouped 80/20 split by `organization_number`

### Folder structure

- `src/data`: dataset loading and split logic
- `src/features`: feature-view construction
- `src/models`: model factory and training logic
- `src/evaluation`: metrics, confusion matrix, feature importance
- `src/utils`: IO helpers and run naming
- `configs`: experiment configuration
- `outputs`: generated artifacts

### Key files

- [configs/experiment_config.yaml](configs/experiment_config.yaml): single source of truth for all experiments (financial, ESG, and Yahoo-enriched feature sets)
- [src/run_experiments.py](src/run_experiments.py)
- [src/data/load_dataset.py](src/data/load_dataset.py)
- [src/data/split_data.py](src/data/split_data.py)
- [src/features/build_feature_views.py](src/features/build_feature_views.py)
- [src/models/model_factory.py](src/models/model_factory.py)
- [src/models/train_models.py](src/models/train_models.py)
- [src/evaluation/evaluate_models.py](src/evaluation/evaluate_models.py)

### Partner workflow (minimal)

1. Place the canonical dataset in [data/modeling_dataset.csv](data/modeling_dataset.csv).
2. Install dependencies.
3. Run experiments with the main config.

```bash
pip install -r requirements.txt
python src/run_experiments.py --config configs/experiment_config.yaml
```

### Run Step 2 experiments

```bash
python src/run_experiments.py --config configs/experiment_config.yaml
```

### Output naming convention

All run artifacts follow:

`run_{date}_{feature_set}_{model_name}_{split_type}.{ext}`

Example:

- `run_2026-03-25_esg_financial_xgboost_stratified.json`

### Generated outputs

- `outputs/01_splits`: split index files
- `outputs/02_trained_models`: serialized model pipelines (`.joblib`)
- `outputs/03_metrics`: per-run metrics JSON + `run_summary_all.csv`
- `outputs/04_plots`: confusion matrix plots
- `outputs/05_feature_importance`: coefficients/importances
- `outputs/06_predictions`: per-row predictions and class probabilities
- `outputs/07_reports`: run comparison reports

### Current Step 2 run status

- Current repository is prepared for execution from [data/modeling_dataset.csv](data/modeling_dataset.csv).
- Historical run outputs are archived under [.archive/2026-03-25_cleanup/outputs](.archive/2026-03-25_cleanup/outputs).

### Data and preprocessing notes

- Per-row deterministic ratio features (e.g., `debt_to_equity_ratio`, `carbon_intensity`) are already included from Step 1.
- Imputation is performed with median values.
- Standardization is applied for Logistic Regression, Naive Bayes, and XGBoost.
- `class_weight='balanced'` is applied where supported, and weighted fitting is used for XGBoost.

### Extension note

The Step 2 framework is ready for additional variables (Interest Coverage Ratio, ROA, ESG Score, Renewable Energy Share, Climate Policy Uncertainty) once added to the curated dataset.

## Step 2.1 
We added complementary figures for the financial data through yahoo finance. These are known as "enriched" features henceforth. We explain at the end how missing data points are dealt with. 



## Step 3: Repeated Evaluation to limit the effects of small sample

Step 3 adds a lightweight repeated-seed evaluation to reduce dependence on one split/seed.

What it does:

- Runs the same Step 2 experiment grid across multiple seeds (default: 10 seeds, 0..9).
- Uses each seed for both split generation and model random states (where supported).
- Stores compact metrics-only outputs (no extra model binaries/plots/prediction files).

Run Step 3:

```bash
python src/run_experiments_repeated.py --config configs/experiment_config.yaml --n-seeds 10 --seed-start 0
```

Step 3 outputs:

- `outputs/03_metrics/repeated_seed_run_summary.csv`
  - One row per `(seed, split_type, feature_set, model_name)` with all metrics.
- `outputs/03_metrics/repeated_seed_aggregate_mean_std.csv`
  - Mean and standard deviation per `(split_type, feature_set, model_name)`.
- `outputs/07_reports/repeated_seed_top_by_f1_mean.md`
  - Top 10 runs ranked by `f1_macro_mean`, including spread (`std`).

This keeps the output structure consistent with Step 2 while adding robustness reporting (average + spread) with minimal artifact growth.


# Explaining the code

## Incomplete datapoints
How we handle missing data points for the enriched financial dataset: 
the code does not drop those datapoints. It imputes missing enriched values and keeps all rows in training/testing.

Where this happens:

    Feature views are created by column selection only, with no missing-value filtering in build_feature_views.py:19.
    Training always starts with a median imputer in the pipeline in train_models.py:35.
    The model is fit on split rows directly (not a filtered subset) in run_experiments.py:101 and run_experiments.py:110.

What that means in practice:

    If a row is missing some enriched features, those missing cells are replaced by the training-set median for each feature.
    If a row is missing all enriched-only features, it still stays in the dataset and gets median-filled values for those columns.
    Since the imputer is inside the sklearn pipeline and fit during model training, test-set missing values are also filled using medians learned from train data only (good leakage hygiene).

## Current structure of data 
Total rows: 382
Rows complete on all financial_enriched features: 134
Rows with any missing financial_enriched feature: 248
(134 + 248 = 382)
Rows with at least one enriched-only value present: 199
Rows with all enriched-only values missing: 183
(199 + 183 = 382)

## roc_auc_ovr_macro

roc_auc_ovr_macro means:

  roc_auc: Area Under the ROC Curve.
  ovr: One-vs-Rest for multiclass classification.
  macro: Unweighted average across classes.

How it is computed for three classes (A, B, C):

  Build three binary problems: A vs not-A, B vs not-B, C vs not-C.
  Compute ROC AUC for each class using predicted probabilities.
  Take the plain mean: (AUC_A + AUC_B + AUC_C) / 3.

Interpretation:

  1.0 means perfect ranking/separation.
  0.5 means random ranking.
  Below 0.5 means systematically worse-than-random ranking for part of the decision surface.

Why it matters in this project:

  It evaluates probability ranking quality, not only hard class labels.
  With macro averaging, each class is weighted equally, so minority classes still count fully.
  It complements f1_macro:
    f1_macro depends on final class decisions.
    roc_auc_ovr_macro uses the full probability distribution.




