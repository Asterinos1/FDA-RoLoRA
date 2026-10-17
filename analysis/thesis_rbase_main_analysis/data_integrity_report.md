# Data Integrity & Verification Report: THESIS-RBASE-MAIN Evaluation Suite

- **Total Target Runs Analyzed**: 36
- **Total Runs Successfully Verified**: 36
- **Integrity Issues / Anomalies**: 0

All 36 runs completed successfully with valid telemetry, contiguous round counters, consistent schema, and zero data corruption.

## Verified Scenario Matrix

| dataset   |   clients |   configured_rounds |   rank | method                 | seeds        | actual_rounds   | early_stopping   |
|:----------|----------:|--------------------:|-------:|:-----------------------|:-------------|:----------------|:-----------------|
| rte       |        20 |                  30 |      4 | FDA-RoLoRA (DualTrack) | [42, 43, 44] | [30, 30, 30]    | True             |
| rte       |        20 |                  30 |      4 | FedAvg-LoRA            | [42, 43, 44] | [30, 30, 30]    | False            |
| rte       |        20 |                  30 |      4 | FedAvg-RoLoRA          | [42, 43, 44] | [30, 30, 30]    | False            |
| rte       |        50 |                  12 |      4 | FDA-RoLoRA (DualTrack) | [42, 43, 44] | [12, 12, 12]    | True             |
| rte       |        50 |                  12 |      4 | FedAvg-LoRA            | [42, 43, 44] | [12, 12, 12]    | False            |
| rte       |        50 |                  12 |      4 | FedAvg-RoLoRA          | [42, 43, 44] | [12, 12, 12]    | False            |
| rte       |        50 |                  12 |      8 | FDA-RoLoRA (DualTrack) | [42, 43, 44] | [12, 12, 12]    | True             |
| rte       |        50 |                  12 |      8 | FedAvg-LoRA            | [42, 43, 44] | [12, 12, 12]    | False            |
| rte       |        50 |                  12 |      8 | FedAvg-RoLoRA          | [42, 43, 44] | [12, 12, 12]    | False            |
| sst2      |        20 |                  75 |      4 | FDA-RoLoRA (DualTrack) | [42, 43, 44] | [14, 16, 20]    | True             |
| sst2      |        50 |                  30 |      4 | FDA-RoLoRA (DualTrack) | [42, 43, 44] | [16, 22, 22]    | True             |
| sst2      |        50 |                  30 |      8 | FDA-RoLoRA (DualTrack) | [42, 43, 44] | [16, 15, 20]    | True             |

## Verification Notes on Early Stopping

- **RTE Experiments (27 runs)**: Configured with `early_stopping=True` for FDA and `False` for baselines. Since RTE test accuracy did not surpass the `early_stopping_start_accuracy=0.90` trigger threshold, all RTE runs completed their full round budgets (30 rounds for C20, 12 rounds for C50).
- **SST-2 Experiments (9 runs)**: Configured with `early_stopping=True` (trigger threshold $\ge 0.90$, patience=5, $\Delta=0.01$, window=3). All 9 SST-2 FDA runs rapidly surpassed 90% accuracy and successfully triggered early stopping between rounds 14 and 22, saving up to 81% of communication rounds compared to max round budgets.
