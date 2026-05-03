# Reclassification Summary (Repeated Seeds)

Model: `random_forest`
Baseline feature set (A): `financial_enriched`
Comparison feature set (B): `esg_financial_enriched`
HY detection regex: `high yield`
Requested seeds: `0..9` (n=10)

A reclassification is counted when a test-row prediction differs between A and B.
HY->IG means A predicts High Yield and B predicts Investment Grade.
IG->HY means A predicts Investment Grade and B predicts High Yield.

## Average counts per seed

| split_type | seeds_compared | avg_n_compared | avg_reclassified | avg_HY_to_IG | avg_IG_to_HY | avg_reclass_rate |
|---|---:|---:|---:|---:|---:|---:|
| grouped | 10 | 77.4 | 14.5 | 1.7 | 1.6 | 0.188 |
| stratified | 10 | 77.0 | 11.6 | 1.2 | 1.6 | 0.151 |

## Seed-level details

| split_type | seed | n_compared | dropped_due_to_misalignment | total_reclassified | HY_to_IG | IG_to_HY | reclass_rate |
|---|---:|---:|---:|---:|---:|---:|---:|
| grouped | 0 | 76 | 0 | 19 | 4 | 1 | 0.250 |
| grouped | 1 | 76 | 0 | 14 | 4 | 0 | 0.184 |
| grouped | 2 | 75 | 0 | 15 | 0 | 1 | 0.200 |
| grouped | 3 | 77 | 0 | 12 | 1 | 2 | 0.156 |
| grouped | 4 | 77 | 0 | 15 | 2 | 2 | 0.195 |
| grouped | 5 | 77 | 0 | 16 | 1 | 2 | 0.208 |
| grouped | 6 | 81 | 0 | 13 | 0 | 1 | 0.160 |
| grouped | 7 | 81 | 0 | 14 | 3 | 1 | 0.173 |
| grouped | 8 | 76 | 0 | 10 | 0 | 2 | 0.132 |
| grouped | 9 | 78 | 0 | 17 | 2 | 4 | 0.218 |
| stratified | 0 | 77 | 0 | 10 | 2 | 1 | 0.130 |
| stratified | 1 | 77 | 0 | 12 | 3 | 1 | 0.156 |
| stratified | 2 | 77 | 0 | 15 | 0 | 1 | 0.195 |
| stratified | 3 | 77 | 0 | 12 | 1 | 1 | 0.156 |
| stratified | 4 | 77 | 0 | 10 | 0 | 1 | 0.130 |
| stratified | 5 | 77 | 0 | 9 | 2 | 1 | 0.117 |
| stratified | 6 | 77 | 0 | 9 | 2 | 1 | 0.117 |
| stratified | 7 | 77 | 0 | 13 | 0 | 6 | 0.169 |
| stratified | 8 | 77 | 0 | 10 | 0 | 1 | 0.130 |
| stratified | 9 | 77 | 0 | 16 | 2 | 2 | 0.208 |