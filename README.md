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