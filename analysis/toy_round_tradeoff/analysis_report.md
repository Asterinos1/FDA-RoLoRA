# FDA-LoRA Experiment Analysis Report

## 1. Dataset/Run Selection
All runs in `results/round_metrics` were processed. Old FDA runs were included as requested by the user, and are distinguished by the `(Old1)` and `(Old2)` suffixes. Corrected FDA runs are suffixed with `(NEW2FDA)`. Baseline runs were properly identified.

## 2. Valid Experiments
The experiments span `LIN` and `NONLIN` architectures across `5c2` (5 clients, 2 classes) and `10c1` (10 clients, 1 class) scenarios. Each valid experiment ran for 100 communication rounds.

## 3. Data-Integrity Findings
Refer to `data_integrity_report.md` for full details. We performed checks for missing rounds, missing values, and identical LIN/NONLIN trajectories (which can indicate incorrect model instantiations in older runs).

## 4. Fixed-Round Performance
Refer to `fixed_round_accuracy_summary.md` and the associated plots. These metrics provide the test accuracy at rounds 25, 50, 75, and 100 for each method.

## 5. Peak Accuracy and Best-Round
Refer to `best_round_summary.csv`. Note that achieving peak accuracy in fewer rounds does not automatically mean communication efficiency if the peak itself is very low compared to baselines.

## 6. Rounds-to-Target
Refer to `rounds_to_target_summary.md`.

## 7. Computation-Efficiency
Refer to `computation_efficiency_summary.csv` and the generated plots comparing Accuracy vs Cumulative Computation Steps. This directly exposes the trade-off FDA makes: performing more local computation per round to potentially save communication rounds.

## 8. FDA Adaptive-Epoch Behavior & Variance/Threshold Observations
Refer to `fda_epoch_statistics.csv` and the individual FDA diagnostic plots. We observed how variance and selected epochs fluctuate over rounds for both the A and B matrices.

## 9. Conclusion
Based on the generated plots and data:
- **FDA vs RoLoRA**: FDA implementations show competitive accuracy. The updated `(NEW2FDA)` runs display corrected behaviors, where NONLIN generally outperforms LIN as expected.
- **Compute vs Communication**: While FDA may reach target accuracies in fewer communication rounds, the `accuracy_vs_compute` plots reveal that FDA accumulates more local compute steps. It trades increased local computation for reduced communication overhead.
- **A/B Matrix Alternation**: The active matrix successfully alternates and often shows differing variance scales, supporting the dual-track threshold mechanism.
