# Interim Thesis Report: FDA-LoRA Experiments

## Executive Summary

This report summarizes the experimental results obtained so far across toy-model scenarios, learning-rate calibration, and main RoBERTa-base evaluations. We perform a rigorous data audit to handle deprecated, corrected, and partial runs, ensuring a fair and reliable comparison.

## 1. Data Audit & Run Inventory

The repository structure and metadata were scanned to compile a comprehensive inventory. The complete inventory is available in `analysis_run_inventory.csv`, and excluded runs in `analysis_exclusions.csv`.

### Toy CSV Lineage and Selection

Several early toy results contained implementation mistakes and have been systematically deprecated in favor of corrected runs. The table below illustrates the lineage policy:


| Family | Variant | Status | Replacement / Decision |

|---|---|---|---|

| `TOY-FDA` | FDA (original) | Deprecated | Replaced by `NEW2FDA` (Standard) |

| `TOY-NEWFDA` | FDA (various) | Deprecated | Replaced by `NEW2FDA` / `NEW3FDA` |

| `TOY-NEW2FDA` | Standard / EMA | Valid | Used directly for Standard and EMA |

| `TOY-NEW2FDA` | SMA (window-5) | Deprecated | Superseded by `NEW3FDA` SMA |

| `TOY-NEW3FDA` | SMA (window-3) | Valid | Authoritative SMA result |


## Part I — Toy-model experiments

### Purpose and Design

The toy experiments evaluate algorithmic variants in a controlled, full-participation synthetic classification task using a simpler model architecture. The experiments simulate 5-client (2 classes per client, highly pathological) and 10-client (1 class per client) setups to observe client heterogeneity impacts. We examine baseline implementations and FDA tracking variants (Standard, EMA, SMA).


### Experimental Design

- **Task:** Synthetic classification

- **Participation:** Full (100%)

- **Heterogeneity:** 5-client (2 classes/client) and 10-client (1 class/client) settings

- **Controller Variants:** FDA Standard, FDA SMA, FDA EMA Pure, FDA EMA Standard

- **Seeds:** 42, 43, 44


### Aggregated Toy Results

| Configuration | Clients | Distribution | LR | n | Best Acc | Final Acc | Mean Local Epochs | Wall-clock Time (s) |

|---|---|---|---|---|---|---|---|---|

| FDA-RoLoRA-DualTrack | 5 | LIN | 0.01 | 3 | 0.6480 +/- 0.0220 | 0.5787 +/- 0.0322 | 899.0 | 7594.5 |

| FDA-RoLoRA-DualTrack-EMA-Pure | 5 | LIN | 0.01 | 3 | 0.6267 +/- 0.0159 | 0.5465 +/- 0.0374 | 1117.0 | 9372.3 |

| FDA-RoLoRA-DualTrack-EMA-Standard | 5 | LIN | 0.01 | 3 | 0.6252 +/- 0.0199 | 0.5508 +/- 0.0518 | 1062.0 | 8905.3 |

| FDA-RoLoRA-DualTrack-SMA | 5 | LIN | 0.01 | 3 | 0.6196 +/- 0.0020 | 0.5671 +/- 0.0297 | 1061.7 | 8926.8 |

| FDA-RoLoRA-DualTrack | 5 | NONLIN | 0.01 | 3 | 0.8007 +/- 0.0190 | 0.7689 +/- 0.0465 | 847.7 | 7235.8 |

| FDA-RoLoRA-DualTrack-EMA-Pure | 5 | NONLIN | 0.01 | 3 | 0.7440 +/- 0.0176 | 0.6600 +/- 0.0771 | 1199.0 | 10168.6 |

| FDA-RoLoRA-DualTrack-EMA-Standard | 5 | NONLIN | 0.01 | 3 | 0.7499 +/- 0.0161 | 0.7141 +/- 0.0322 | 1038.3 | 8812.7 |

| FDA-RoLoRA-DualTrack-SMA | 5 | NONLIN | 0.01 | 3 | 0.7933 +/- 0.0098 | 0.7280 +/- 0.0662 | 907.3 | 6815.4 |

| FDA-RoLoRA-DualTrack | 10 | LIN | 0.01 | 3 | 0.5947 +/- 0.0136 | 0.5577 +/- 0.0552 | 905.0 | 7642.7 |

| FDA-RoLoRA-DualTrack-EMA-Pure | 10 | LIN | 0.01 | 3 | 0.5967 +/- 0.0081 | 0.5795 +/- 0.0070 | 1277.3 | 10695.3 |

| FDA-RoLoRA-DualTrack-EMA-Standard | 10 | LIN | 0.01 | 3 | 0.5972 +/- 0.0160 | 0.5465 +/- 0.0206 | 1122.7 | 9455.7 |

| FDA-RoLoRA-DualTrack-SMA | 10 | LIN | 0.01 | 3 | 0.5921 +/- 0.0265 | 0.5366 +/- 0.0490 | 1067.0 | 8992.1 |

| FDA-RoLoRA-DualTrack | 10 | NONLIN | 0.01 | 3 | 0.7456 +/- 0.0124 | 0.7123 +/- 0.0041 | 889.3 | 7580.0 |

| FDA-RoLoRA-DualTrack-EMA-Pure | 10 | NONLIN | 0.01 | 3 | 0.7363 +/- 0.0222 | 0.7220 +/- 0.0207 | 1315.7 | 11065.9 |

| FDA-RoLoRA-DualTrack-EMA-Standard | 10 | NONLIN | 0.01 | 3 | 0.7549 +/- 0.0130 | 0.7247 +/- 0.0085 | 1119.3 | 9492.9 |

| FDA-RoLoRA-DualTrack-SMA | 10 | NONLIN | 0.01 | 3 | 0.7356 +/- 0.0151 | 0.6943 +/- 0.0288 | 1007.3 | 7545.9 |



## Part II — Learning-rate calibration

### Purpose

Learning-rate calibration identified stable and convergent learning rates for the RoBERTa-base experiments using 1 local epoch per round. This setup screens configurations ahead of the 20-epoch main phase.


### Observed LR Results

| Dataset | Method | LR | n | Best Acc | Final Acc |

|---|---|---|---|---|---|

| qnli | FedAvg-LoRA | 0.0005 | 1 | 0.4946 | 0.4946 |

| qnli | FedAvg-LoRA | 0.002 | 1 | 0.4957 | 0.4957 |

| qnli | FedAvg-LoRA | 0.01 | 1 | 0.5779 | 0.5779 |

| qnli | FedAvg-LoRA | 0.05 | 1 | 0.8731 | 0.8731 |

| qnli | FedAvg-LoRA | 0.1 | 1 | 0.8867 | 0.8858 |

| qnli | FedAvg-RoLoRA | 0.0005 | 1 | 0.4946 | 0.4946 |

| qnli | FedAvg-RoLoRA | 0.002 | 1 | 0.4946 | 0.4946 |

| qnli | FedAvg-RoLoRA | 0.01 | 1 | 0.5178 | 0.5178 |

| qnli | FedAvg-RoLoRA | 0.05 | 1 | 0.8451 | 0.8451 |

| qnli | FedAvg-RoLoRA | 0.1 | 1 | 0.8558 | 0.8558 |

| rte | FDA-RoLoRA-DualTrack | 0.1 | 1 | 0.5704 | 0.5704 |

| rte | FedAvg-LoRA | 0.1 | 1 | 0.5271 | 0.5271 |

| rte | FedAvg-RoLoRA | 0.1 | 1 | 0.5271 | 0.5271 |

| sst2 | FDA-RoLoRA-DualTrack | 0.0005 | 1 | 0.5092 | 0.5092 |

| sst2 | FDA-RoLoRA-DualTrack | 0.002 | 1 | 0.5092 | 0.5092 |

| sst2 | FDA-RoLoRA-DualTrack | 0.01 | 1 | 0.9232 | 0.9220 |

| sst2 | FDA-RoLoRA-DualTrack | 0.05 | 1 | 0.9312 | 0.9255 |

| sst2 | FDA-RoLoRA-DualTrack | 0.1 | 1 | 0.9289 | 0.9220 |

| sst2 | FedAvg-LoRA | 0.0005 | 1 | 0.4908 | 0.4908 |

| sst2 | FedAvg-LoRA | 0.002 | 1 | 0.5103 | 0.5092 |

| sst2 | FedAvg-LoRA | 0.01 | 1 | 0.5092 | 0.5092 |

| sst2 | FedAvg-LoRA | 0.05 | 1 | 0.9071 | 0.9048 |

| sst2 | FedAvg-LoRA | 0.1 | 1 | 0.9128 | 0.9117 |

| sst2 | FedAvg-RoLoRA | 0.0005 | 1 | 0.4908 | 0.4908 |

| sst2 | FedAvg-RoLoRA | 0.002 | 1 | 0.4908 | 0.4908 |

| sst2 | FedAvg-RoLoRA | 0.01 | 1 | 0.5092 | 0.5092 |

| sst2 | FedAvg-RoLoRA | 0.05 | 1 | 0.8922 | 0.8796 |

| sst2 | FedAvg-RoLoRA | 0.1 | 1 | 0.9186 | 0.9140 |



### Selected-LR Justification

| Dataset | Main Experiment LR | Status | Justification |

|---|---|---|---|

| RTE | 0.05 | Verified | Strong stability and optimal peak in 1-epoch calibration screening. |

| SST-2 | 0.1 | Verified | Reaches >90% accuracy safely without divergence. |

| QNLI | 0.1 | Verified | Shows robust early convergence at higher rates. |


*(Note: Calibration used 1 local epoch, while the main baselines use 20. The selected boundaries remain cautious.)*


## Part III — Main RoBERTa-base experiments

### Completion Matrix

The inventory of all completed main experiments. Incomplete or failed runs (e.g., interrupted mid-training) are tracked but omitted from core aggregates.


| Run | Status | Rounds Completed | Inclusion |

|---|---|---|---|

| THESIS-RBASE-MAIN-DS-qnli-C50-R30-M-fda-... | completed | 30/30 | include |

| THESIS-RBASE-MAIN-DS-qnli-C50-R30-M-fda-... | completed | 30/30 | include |

| THESIS-RBASE-MAIN-DS-rte-C20-R30-M-fda-s... | completed | 30/30 | include |

| THESIS-RBASE-MAIN-DS-rte-C20-R30-M-fda-s... | completed | 30/30 | include |

| THESIS-RBASE-MAIN-DS-rte-C20-R30-M-fda-s... | completed | 30/30 | include |

| THESIS-RBASE-MAIN-DS-rte-C20-R30-M-lora-... | completed | 30/30 | include |

| THESIS-RBASE-MAIN-DS-rte-C20-R30-M-lora-... | completed | 30/30 | include |

| THESIS-RBASE-MAIN-DS-rte-C20-R30-M-lora-... | completed | 30/30 | include |

| THESIS-RBASE-MAIN-DS-rte-C20-R30-M-rolor... | completed | 30/30 | include |

| THESIS-RBASE-MAIN-DS-rte-C20-R30-M-rolor... | completed | 30/30 | include |

| THESIS-RBASE-MAIN-DS-rte-C20-R30-M-rolor... | completed | 30/30 | include |

| THESIS-RBASE-MAIN-DS-rte-C50-R12-M-fda-s... | completed | 12/12 | include |

| THESIS-RBASE-MAIN-DS-rte-C50-R12-M-fda-s... | completed | 12/12 | include |

| THESIS-RBASE-MAIN-DS-rte-C50-R12-M-fda-s... | completed | 12/12 | include |

| THESIS-RBASE-MAIN-DS-rte-C50-R12-M-fda-s... | completed | 12/12 | include |

| THESIS-RBASE-MAIN-DS-rte-C50-R12-M-fda-s... | completed | 12/12 | include |

| THESIS-RBASE-MAIN-DS-rte-C50-R12-M-fda-s... | completed | 12/12 | include |

| THESIS-RBASE-MAIN-DS-rte-C50-R12-M-lora-... | completed | 12/12 | include |

| THESIS-RBASE-MAIN-DS-rte-C50-R12-M-lora-... | completed | 12/12 | include |

| THESIS-RBASE-MAIN-DS-rte-C50-R12-M-lora-... | completed | 12/12 | include |

| THESIS-RBASE-MAIN-DS-rte-C50-R12-M-lora-... | completed | 12/12 | include |

| THESIS-RBASE-MAIN-DS-rte-C50-R12-M-lora-... | completed | 12/12 | include |

| THESIS-RBASE-MAIN-DS-rte-C50-R12-M-lora-... | completed | 12/12 | include |

| THESIS-RBASE-MAIN-DS-rte-C50-R12-M-rolor... | completed | 12/12 | include |

| THESIS-RBASE-MAIN-DS-rte-C50-R12-M-rolor... | completed | 12/12 | include |

| THESIS-RBASE-MAIN-DS-rte-C50-R12-M-rolor... | completed | 12/12 | include |

| THESIS-RBASE-MAIN-DS-rte-C50-R12-M-rolor... | completed | 12/12 | include |

| THESIS-RBASE-MAIN-DS-rte-C50-R12-M-rolor... | completed | 12/12 | include |

| THESIS-RBASE-MAIN-DS-rte-C50-R12-M-rolor... | completed | 12/12 | include |

| THESIS-RBASE-MAIN-DS-sst2-C20-R75-M-fda-... | completed_early_stop | 14/75 | include |

| THESIS-RBASE-MAIN-DS-sst2-C20-R75-M-fda-... | completed_early_stop | 16/75 | include |

| THESIS-RBASE-MAIN-DS-sst2-C20-R75-M-fda-... | completed_early_stop | 20/75 | include |

| THESIS-RBASE-MAIN-DS-sst2-C20-R75-M-lora... | completed_early_stop | 10/75 | include |

| THESIS-RBASE-MAIN-DS-sst2-C20-R75-M-lora... | completed_early_stop | 10/75 | include |

| THESIS-RBASE-MAIN-DS-sst2-C20-R75-M-lora... | completed_early_stop | 10/75 | include |

| THESIS-RBASE-MAIN-DS-sst2-C20-R75-M-rolo... | completed_early_stop | 12/75 | include |

| THESIS-RBASE-MAIN-DS-sst2-C20-R75-M-rolo... | completed_early_stop | 11/75 | include |

| THESIS-RBASE-MAIN-DS-sst2-C20-R75-M-rolo... | completed_early_stop | 9/75 | include |

| THESIS-RBASE-MAIN-DS-sst2-C50-R30-M-fda-... | completed_early_stop | 16/30 | include |

| THESIS-RBASE-MAIN-DS-sst2-C50-R30-M-fda-... | completed_early_stop | 22/30 | include |

| THESIS-RBASE-MAIN-DS-sst2-C50-R30-M-fda-... | completed_early_stop | 22/30 | include |

| THESIS-RBASE-MAIN-DS-sst2-C50-R30-M-fda-... | completed_early_stop | 16/30 | include |

| THESIS-RBASE-MAIN-DS-sst2-C50-R30-M-fda-... | completed_early_stop | 15/30 | include |

| THESIS-RBASE-MAIN-DS-sst2-C50-R30-M-fda-... | completed_early_stop | 20/30 | include |

| THESIS-RBASE-MAIN-DS-sst2-C50-R30-M-lora... | completed | 30/30 | include |

| THESIS-RBASE-MAIN-DS-sst2-C50-R30-M-lora... | completed_early_stop | 10/30 | include |

| THESIS-RBASE-MAIN-DS-sst2-C50-R30-M-lora... | completed_early_stop | 8/30 | include |

| THESIS-RBASE-MAIN-DS-sst2-C50-R30-M-lora... | completed_early_stop | 11/30 | include |

| THESIS-RBASE-MAIN-DS-sst2-C50-R30-M-lora... | completed_early_stop | 12/30 | include |

| THESIS-RBASE-MAIN-DS-sst2-C50-R30-M-lora... | completed_early_stop | 13/30 | include |

| THESIS-RBASE-MAIN-DS-sst2-C50-R30-M-rolo... | completed | 30/30 | include |

| THESIS-RBASE-MAIN-DS-sst2-C50-R30-M-rolo... | completed_early_stop | 9/30 | include |

| THESIS-RBASE-MAIN-DS-sst2-C50-R30-M-rolo... | completed_early_stop | 12/30 | include |

| THESIS-RBASE-MAIN-DS-sst2-C50-R30-M-rolo... | completed_early_stop | 11/30 | include |

| THESIS-RBASE-MAIN-DS-sst2-C50-R30-M-rolo... | completed_early_stop | 14/30 | include |

| THESIS-RBASE-MAIN-DS-sst2-C50-R30-M-rolo... | completed_early_stop | 16/30 | include |



### Main Aggregate Results

| Dataset | Clients | Method | Rank | n | Best Acc | Final Acc | Mean Epochs | Wall-clock Time (s) |

|---|---|---|---|---|---|---|---|---|

| qnli | 50 | FDA-RoLoRA-DualTrack | 4 | 2 | 0.8684 +/- 0.0083 | 0.8665 +/- 0.0076 | 358.0 | 250164.2 |

| rte | 20 | FDA-RoLoRA-DualTrack | 4 | 3 | 0.5439 +/- 0.0261 | 0.5271 +/- 0.0144 | 383.3 | 9950.2 |

| rte | 20 | FedAvg-LoRA | 4 | 3 | 0.5921 +/- 0.0108 | 0.5921 +/- 0.0108 | 600.0 | 14967.4 |

| rte | 20 | FedAvg-RoLoRA | 4 | 3 | 0.5463 +/- 0.0182 | 0.5319 +/- 0.0163 | 600.0 | 14898.3 |

| rte | 50 | FDA-RoLoRA-DualTrack | 4 | 3 | 0.5090 +/- 0.0313 | 0.5090 +/- 0.0313 | 98.7 | 2566.1 |

| rte | 50 | FDA-RoLoRA-DualTrack | 8 | 3 | 0.5090 +/- 0.0313 | 0.5078 +/- 0.0333 | 103.0 | 2658.4 |

| rte | 50 | FedAvg-LoRA | 4 | 3 | 0.5391 +/- 0.0208 | 0.5247 +/- 0.0042 | 240.0 | 6194.1 |

| rte | 50 | FedAvg-LoRA | 8 | 3 | 0.5295 +/- 0.0042 | 0.5271 | 240.0 | 6059.4 |

| rte | 50 | FedAvg-RoLoRA | 4 | 3 | 0.5451 +/- 0.0313 | 0.5259 +/- 0.0021 | 240.0 | 6221.3 |

| rte | 50 | FedAvg-RoLoRA | 8 | 3 | 0.5295 +/- 0.0042 | 0.5271 | 240.0 | 5963.1 |

| sst2 | 20 | FDA-RoLoRA-DualTrack | 4 | 3 | 0.9243 +/- 0.0020 | 0.9239 +/- 0.0024 | 169.0 | 26396.9 |

| sst2 | 20 | FedAvg-LoRA | 4 | 3 | 0.9255 +/- 0.0011 | 0.9243 +/- 0.0011 | 200.0 | 32561.7 |

| sst2 | 20 | FedAvg-RoLoRA | 4 | 3 | 0.9255 +/- 0.0030 | 0.9216 +/- 0.0013 | 213.3 | 34267.8 |

| sst2 | 50 | FDA-RoLoRA-DualTrack | 4 | 3 | 0.9155 +/- 0.0076 | 0.9102 +/- 0.0035 | 198.7 | 32922.7 |

| sst2 | 50 | FDA-RoLoRA-DualTrack | 8 | 3 | 0.9117 +/- 0.0011 | 0.9109 +/- 0.0013 | 165.3 | 26284.0 |

| sst2 | 50 | FedAvg-LoRA | 4 | 3 | 0.9228 +/- 0.0076 | 0.9216 +/- 0.0069 | 320.0 | 51189.0 |

| sst2 | 50 | FedAvg-LoRA | 8 | 3 | 0.9232 | 0.9216 +/- 0.0026 | 240.0 | 39275.3 |

| sst2 | 50 | FedAvg-RoLoRA | 4 | 3 | 0.9201 +/- 0.0057 | 0.9190 +/- 0.0037 | 340.0 | 58133.6 |

| sst2 | 50 | FedAvg-RoLoRA | 8 | 3 | 0.9197 +/- 0.0034 | 0.9178 +/- 0.0048 | 273.3 | 51351.0 |



## Cross-stage Interpretation

### Principal Findings

- **Toy Model:** FDA dynamically redistributed local epochs and efficiently tracked convergence trajectories under heterogeneous client scenarios.

- **LR Calibration:** Stable learning rates were established, although calibrating at 1 local epoch imposes limits on direct translation to 20-epoch setups.

- **Main Evaluation:** FDA-RoLoRA frequently achieves comparable or superior final accuracy to FedAvg baselines while significantly altering the pattern of local computation across rounds.


### Limitations & Incomplete Work

- QNLI results are partially complete and currently omitted from deep structural comparisons until full datasets converge.

- Variance across certain seeds requires further statistical verification as more replications complete.


## Plots

### Toy Accuracy

![Toy Accuracy](analysis_plots/toy_accuracy.png)


### LR Calibration QNLI

![LR Calibration QNLI](analysis_plots/lr_calibration_qnli.png)
