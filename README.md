# DD151X

Degree project at KTH (Industrial Engineering and Management, Computer Science track) focused on ESG-integrated machine learning for corporate credit risk assessment in the Nordic bond market.

## Project Goal

This project evaluates whether adding ESG variables to credit risk models improves predictive performance compared with traditional financial-only approaches.

The modelling setup follows a supervised multi-class classification problem where the target is engineered into three credit risk classes:

- Investment Grade Low Risk
- Investment Grade High Risk
- High Yield

The feature space combines financial factors and ESG metrics to test whether this improves predictive power for credit risk assessment applications.

## Research Focus

The project addresses two dimensions:

- Technical: Compare statistical and machine learning models for credit rating prediction using metrics such as F1-score and ROC-AUC.
- Economic/managerial: Assess implications for capital allocation, risk exposure, and institutional feasibility in a Nordic context.

## Data Sources and Current Data Artifacts

Data is collected from Stamdata feeds and merged into a curated modelling table.

Current repository artifacts include:

- Curated dataset: [data/curated_rated_esg_dataset_max_rows.csv](data/curated_rated_esg_dataset_max_rows.csv)
- Latest raw feed downloads: [data/run_latest](data/run_latest)
- Data collection and curation script: [download_curated_rated_esg_dataset.py](download_curated_rated_esg_dataset.py)

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
- [data](data): Curated dataset and latest run artifacts.
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

## Step 1 Implementation (Completed)

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

Generated artifacts:

- [data/modeling_dataset_step1.csv](data/modeling_dataset_step1.csv): Clean dataset with ID columns, time columns, selected factors, and `risk_class_3` target.
- [data/modeling_dataset_step1_diagnostics.json](data/modeling_dataset_step1_diagnostics.json): Data quality and class balance diagnostics.
- [data/feature_dictionary_step1.json](data/feature_dictionary_step1.json): Feature definitions and source mapping for reproducibility.

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


Step 2 Goal
Build and compare four model families on the same 3-class target using three feature views:

ESG only
Financial only
ESG + Financial
Models to include:

Logistic Regression
Naive Bayes
Random Forest
XGBoost (Gradient Boosting family)
Proposed Folder Structure
Keep it minimal and numbered so execution flow is obvious.

src
src/data
src/features
src/models
src/evaluation
src/utils
configs
outputs
outputs/01_splits
outputs/02_trained_models
outputs/03_metrics
outputs/04_plots
outputs/05_feature_importance
outputs/06_predictions
outputs/07_reports
File Naming Convention
Use one consistent pattern everywhere:
run_{date}{feature_set}{model_name}_{split_type}.{ext}

Examples:

run_2026-03-25_esg_only_logreg_stratified.json
run_2026-03-25_financial_only_rf_grouped.csv
run_2026-03-25_esg_fin_xgb_stratified.png
This makes results searchable and easy to explain in the thesis.

Planned Python Modules

src/data/load_dataset.py
Loads data/modeling_dataset_step1.csv
Basic schema check
src/features/build_feature_views.py
Creates ESG-only, financial-only, combined matrices
Applies final feature list and missing-value policy
src/data/split_data.py
Creates train/test splits
Saves split indices to outputs/01_splits
src/models/train_models.py
Trains all four models for each feature set
Handles class weighting and optional resampling
src/evaluation/evaluate_models.py
Computes metrics and confusion matrices
Writes standardized output files
src/evaluation/feature_importance.py
Model explainability outputs (coefficients/importances/SHAP)
src/run_experiments.py
Orchestrator that runs everything end-to-end
configs/experiment_config.yaml
Central config for features, split seed, model params, output paths
Feature Sets to Train On

From current available columns in data/modeling_dataset_step1.csv:

Financial only
enterprise_value
revenue
book_value_equity
book_value_debt
debt_to_equity_ratio
ESG only
total_ghg_emission
scope1
scope2_location
carbon_intensity
carbon_target
high_impact_climate_sector
fossil_fuel_sector
report_biodiversity
negative_affect_biodiversity
exp_controversial_weapons
exp_controversial_products
exp_debt_collection_or_loans
female_board
male_board
female_board_share
ESG + Financial
Union of both lists above
Target:

risk_class_3
ID/time/meta kept but not used as training features:

organization_number
issuer_name
period_from
period_to
period_year
rating_agency
rating_symbol
rating_normalized
Important Data Note
Some variables highlighted in your thesis narrative are not currently in Step 1 data (for example Interest Coverage Ratio, ROA, ESG Score, Renewable Energy Share, Climate Policy Uncertainty).
Plan should include a later data-enrichment task so importance rankings can include them explicitly.

Ratios and Transformations: Before or After Split
Yes, ratio engineering is needed and should be explicit.

Recommended:

Per-row deterministic ratios (for example debt_to_equity_ratio, carbon_intensity): compute before split (no leakage).
Any operation that learns from global distribution (scaling, imputation statistics, PCA): fit on train only, then apply to test.
Use Z-score standardization for models that benefit from scaling:
Logistic Regression
Naive Bayes
XGBoost optional
Random Forest not required but okay for consistency
Train/Test Split Plan
Use two split strategies:

Primary simple split for easy explanation
80/20 stratified split on risk_class_3
random_state = 42
Robustness split to reduce issuer leakage risk
Grouped split by organization_number
Keep train/test issuers disjoint
Report both results, use grouped result as stronger evidence
This keeps communication simple while addressing a major methodological risk.

Class Imbalance Handling
From data/modeling_dataset_step1_diagnostics.json, High Yield is minority.
Plan:

Baseline: class_weight = balanced where supported
Secondary: train-set-only oversampling (SMOTE or random oversampling)
Compare both and keep the better validated setup
Evaluation Outputs (Per Model x Feature Set x Split Type)

Metrics JSON/CSV
accuracy
macro precision
macro recall
macro F1
weighted F1
one-vs-rest ROC-AUC (macro)
Confusion matrix image

Per-row predictions CSV

true label
predicted label
class probabilities
Explainability artifacts
Logistic Regression coefficients
Random Forest feature importances
XGBoost feature importances
SHAP summary for tree models (optional in first pass, recommended in second pass)
Run summary markdown
Best model by macro F1
Best model by ROC-AUC
Performance delta across ESG-only vs Financial-only vs Combined
Execution Matrix
Total planned runs:

3 feature sets x 4 models x 2 split types = 24 runs
Start simple:

Do 12 runs on primary stratified split
Add 12 grouped robustness runs after baseline is stable
Definition of Done for Step 2

All 24 runs completed without failures
Reproducible outputs saved to structured folders
One comparison table across all runs
One explainability table listing top drivers per best model
Clear answer to:
Does ESG-only beat financial-only?
Does ESG + financial beat both?
Which model family is strongest under class imbalance and grouped split?