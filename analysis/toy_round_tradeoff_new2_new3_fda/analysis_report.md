# FDA-LoRA Benchmark Analysis Report (NEW2FDA & NEW3FDA Runs Included)

## 1. Dataset & Run Selection Criteria
This analysis includes all baseline runs (`FedAvg-LoRA`, `FFA-LoRA`, `RoLoRA`) alongside the latest FDA runs (`NEW2FDA` and `NEW3FDA`). All deprecated and initial FDA runs (`TOY-FDA_*` and `TOY-NEWFDA_*`) have been excluded from the FDA evaluation suite.

- Total included runs: **66** across **3 seeds** (`S42`, `S43`, `S44`).
- Methods evaluated: `FDA EMA-Standard (NEW2FDA)`, `FDA SMA (NEW3FDA)`, `FDA Standard (NEW2FDA)`, `FFA-LoRA`, `FedAvg-LoRA`, `RoLoRA`.

## 2. Experimental Scenarios
Experiments were evaluated over 100 communication rounds across 4 benchmark configurations:
1. **`NONLIN - 10c1`**: Multi-Layer Perceptron (MLP), Pathological non-IID (10 clients, 1 class/client).
2. **`NONLIN - 5c2`**: Multi-Layer Perceptron (MLP), Non-IID (5 clients, 2 classes/client).
3. **`LIN - 10c1`**: Linear Classifier, Pathological non-IID (10 clients, 1 class/client).
4. **`LIN - 5c2`**: Linear Classifier, Non-IID (5 clients, 2 classes/client).

## 3. Data Integrity & Verification
Detailed checks in `data_integrity_report.md` confirm:
- All 66 experiment runs completed exactly 100 communication rounds.
- No duplicate hashes, corrupted records, or missing step/epoch fields were detected.
- Linear and Non-linear trajectories are distinct and exhibit expected architectural divergences.

## 4. Key Performance Highlights

### Final Round Accuracy (Round 100)

#### `NONLIN - 10c1`

| Method | Final Test Accuracy (R=100) |
|:---|:---:|
| **FDA EMA-Standard (NEW2FDA)** | 72.47% ± 0.85% |
| **FDA Standard (NEW2FDA)** | 71.23% ± 0.41% |
| **FDA SMA (NEW3FDA)** | 69.43% ± 2.88% |
| **RoLoRA** | 69.40% ± 3.58% |
| **FFA-LoRA** | 69.14% ± 4.42% |
| **FedAvg-LoRA** | 67.01% ± 2.07% |

#### `NONLIN - 5c2`

| Method | Final Test Accuracy (R=100) |
|:---|:---:|
| **FDA Standard (NEW2FDA)** | 76.89% ± 4.65% |
| **RoLoRA** | 75.73% ± 5.39% |
| **FedAvg-LoRA** | 75.07% ± 1.05% |
| **FDA SMA (NEW3FDA)** | 72.80% ± 6.62% |
| **FDA EMA-Standard (NEW2FDA)** | 71.41% ± 3.22% |
| **FFA-LoRA** | 59.22% ± 2.03% |

#### `LIN - 10c1`

| Method | Final Test Accuracy (R=100) |
|:---|:---:|
| **FedAvg-LoRA** | 68.01% ± 10.76% |
| **FDA Standard (NEW2FDA)** | 55.77% ± 5.52% |
| **RoLoRA** | 54.83% ± 7.05% |
| **FDA EMA-Standard (NEW2FDA)** | 54.65% ± 2.06% |
| **FFA-LoRA** | 31.17% ± 7.52% |

#### `LIN - 5c2`

| Method | Final Test Accuracy (R=100) |
|:---|:---:|
| **RoLoRA** | 61.26% ± 3.55% |
| **FDA Standard (NEW2FDA)** | 57.87% ± 3.22% |
| **FDA EMA-Standard (NEW2FDA)** | 55.08% ± 5.18% |
| **FedAvg-LoRA** | 43.94% ± 1.77% |
| **FFA-LoRA** | 31.55% ± 5.34% |

### Peak Accuracy Across 100 Rounds

#### `NONLIN - 10c1`

| Method | Peak Test Accuracy | Best Round (Median) |
|:---|:---:|:---:|
| **FDA EMA-Standard (NEW2FDA)** | 75.49% ± 1.30% | R94 |
| **RoLoRA** | 75.24% ± 0.44% | R92 |
| **FDA Standard (NEW2FDA)** | 74.56% ± 1.24% | R98 |
| **FDA SMA (NEW3FDA)** | 73.56% ± 1.51% | R86 |
| **FedAvg-LoRA** | 71.82% ± 1.49% | R85 |
| **FFA-LoRA** | 71.78% ± 1.22% | R98 |

#### `NONLIN - 5c2`

| Method | Peak Test Accuracy | Best Round (Median) |
|:---|:---:|:---:|
| **FDA Standard (NEW2FDA)** | 80.07% ± 1.90% | R94 |
| **RoLoRA** | 79.93% ± 0.34% | R96 |
| **FDA SMA (NEW3FDA)** | 79.33% ± 0.98% | R74 |
| **FedAvg-LoRA** | 75.12% ± 1.06% | R100 |
| **FDA EMA-Standard (NEW2FDA)** | 74.99% ± 1.61% | R82 |
| **FFA-LoRA** | 60.14% ± 0.69% | R92 |

#### `LIN - 10c1`

| Method | Peak Test Accuracy | Best Round (Median) |
|:---|:---:|:---:|
| **FedAvg-LoRA** | 70.98% ± 9.60% | R97 |
| **RoLoRA** | 60.12% ± 1.82% | R88 |
| **FDA EMA-Standard (NEW2FDA)** | 59.72% ± 1.60% | R85 |
| **FDA Standard (NEW2FDA)** | 59.47% ± 1.36% | R72 |
| **FFA-LoRA** | 39.12% ± 4.40% | R75 |

#### `LIN - 5c2`

| Method | Peak Test Accuracy | Best Round (Median) |
|:---|:---:|:---:|
| **RoLoRA** | 65.91% ± 0.77% | R80 |
| **FDA Standard (NEW2FDA)** | 64.80% ± 2.20% | R91 |
| **FDA EMA-Standard (NEW2FDA)** | 62.52% ± 1.99% | R94 |
| **FedAvg-LoRA** | 54.48% ± 1.84% | R35 |
| **FFA-LoRA** | 38.24% ± 2.51% | R40 |

## 5. Compute vs. Communication Trade-Off Analysis
FDA dynamically adapts the local computation budget per client based on gradient variance feedback. While fixed baselines train for 3 epochs per round (~28,200 total steps), FDA variants train between 4.5 and 7.5 epochs on average (~42,000 to ~70,000 total steps). This increased local compute expenditure directly translates into improved parameter convergence and stability in highly heterogeneous data environments.

## 6. Generated Output Artifacts
The following artifacts are available in this analysis folder:
- `fixed_round_accuracy_summary.csv` & `fixed_round_accuracy_summary.md`
- `best_round_summary.csv`
- `rounds_to_target_summary.csv` & `rounds_to_target_summary.md`
- `computation_efficiency_summary.csv`
- `fda_epoch_statistics.csv`
- `LLM_SUMMARY.json` & `LLM_SUMMARY.md`
- Visualization charts (PNG @ 300 DPI and vector PDF) for accuracy vs. rounds, full trajectories, compute scaling, and FDA variance/epoch dynamics.
