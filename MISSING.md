# Missing Experimental Runs & Audit: FDA vs. Baselines Evaluation

This document provides a systematic catalog of all missing experimental runs, incomplete random seed sets, unrepresented baseline methods, and unexecuted dataset configurations required to bring the findings in [`FDA_vs_Baselines_Evaluation_Assessment.md`](file:///C:/Users/PC/Documents/GitHub/FDA-LoRA/FDA_vs_Baselines_Evaluation_Assessment.md) to a publication-grade **3-seed statistical standard** ($S \in \{42, 43, 44\}$).

---

## 1. Executive Summary: Missing Runs Inventory

Currently, **all results reported for FDA-RoLoRA-DualTrack-Calibrated in the assessment document are evaluated strictly on a single seed ($S=42$)**. 

To compute rigorous multi-seed statistical metrics ($\mu \pm \sigma$) and achieve full parity across methods and tasks, the following runs are missing:

| Category | Missing Elements | Total Runs Needed | Priority |
| :--- | :--- | :---: | :---: |
| **1. Calibrated FDA ($E_{\max}=20$)** | Seeds 43 and 44 for all 48 configurations across 6 GLUE datasets | **96 runs** | 🔴 High (Core Thesis Metric) |
| **2. Calibrated FDA ($E_{\max}=30$)** | Seeds 43 and 44 for the 8 major benchmark configurations (MNLI, QNLI, QQP, SST-2) | **16 runs** | 🔴 High (Peak Accuracy Frontier) |
| **3. Baseline: LoRA ($E \in \{5, 10\}$)** | Seed 44 across the 4 major benchmarks (MNLI, QNLI, QQP, SST-2 on $C \in \{20, 50\}$) | **16 runs** | 🟡 Medium (Completes 3-seed Baseline) |
| **4. Baseline: RoLoRA ($E \in \{5, 10\}$)** | Seed 44 across the 4 major benchmarks (MNLI, QNLI, QQP, SST-2 on $C \in \{20, 50\}$) | **16 runs** | 🟡 Medium (Completes 3-seed Baseline) |
| **5. Baseline: FFA-LoRA ($E \in \{5, 10\}$)** | Seeds 42, 43, 44 missing on MNLI, QQP, SST-2, QNLI | **47 runs** | 🟢 Low-Medium (Fills FFA-LoRA Void) |
| **6. Extended Ceiling ($E_{\max}=30$)** | Missing on small datasets: MRPC & RTE ($C=20, 50$) | **4 runs** (per seed) | ⚪ Exploratory |
| **7. Toy / Mathematical Non-Linear** | Calibrated controller runs on MNIST | **N/A** (Toy model) | ⚪ Optional / Theoretical |

---

## 2. Seed Deficit Audit: Calibrated FDA Suite (`THESIS-RBASE-CALIB`)

All 56 existing runs in `results/round_metrics/` prefixed with `THESIS-RBASE-CALIB-` possess **only Seed 42**.

### A. Core Factorial Grid ($E \in [3, 20]$, 48 Configurations)
Each configuration requires **Seed 43** and **Seed 44**:
* **Parameter Groups Evaluated**:
  1. `CRATIO1.0-BDISC` ($\kappa = 1.0$, `enable_b_discount=True`, factor 3.0)
  2. `CRATIO1.0-NODISC` ($\kappa = 1.0$, `enable_b_discount=False`)
  3. `CRATIO0.75-BDISC` ($\kappa = 0.75$, `enable_b_discount=True`, factor 3.0)
  4. `CRATIO0.75-NODISC` ($\kappa = 0.75$, `enable_b_discount=False`)

* **Detailed Configuration Matrix**:
  | Dataset | Partition ($C$) | Max Rounds ($R$) | Parameter Groups | Present Seeds | Missing Seeds |
  | :--- | :---: | :---: | :--- | :---: | :---: |
  | **MNLI** | $C=20$ | $R=50$ | All 4 Groups | `[42]` | **`[43, 44]`** (8 runs) |
  | **MNLI** | $C=50$ | $R=30$ | All 4 Groups | `[42]` | **`[43, 44]`** (8 runs) |
  | **MRPC** | $C=20$ | $R=30$ | All 4 Groups | `[42]` | **`[43, 44]`** (8 runs) |
  | **MRPC** | $C=50$ | $R=12$ | All 4 Groups | `[42]` | **`[43, 44]`** (8 runs) |
  | **QNLI** | $C=20$ | $R=50$ | All 4 Groups | `[42]` | **`[43, 44]`** (8 runs) |
  | **QNLI** | $C=50$ | $R=30$ | All 4 Groups | `[42]` | **`[43, 44]`** (8 runs) |
  | **QQP** | $C=20$ | $R=50$ | All 4 Groups | `[42]` | **`[43, 44]`** (8 runs) |
  | **QQP** | $C=50$ | $R=30$ | All 4 Groups | `[42]` | **`[43, 44]`** (8 runs) |
  | **RTE** | $C=20$ | $R=30$ | All 4 Groups | `[42]` | **`[43, 44]`** (8 runs) |
  | **RTE** | $C=50$ | $R=12$ | All 4 Groups | `[42]` | **`[43, 44]`** (8 runs) |
  | **SST-2** | $C=20$ | $R=50$ | All 4 Groups | `[42]` | **`[43, 44]`** (8 runs) |
  | **SST-2** | $C=50$ | $R=30$ | All 4 Groups | `[42]` | **`[43, 44]`** (8 runs) |

### B. Extended Local Epoch Ceiling ($E \in [3, 30]$, 8 Configurations)
Evaluated under the optimal parameter group (`CRATIO1.0-BDISC`):
| Dataset | Partition ($C$) | Max Rounds ($R$) | Parameter Group | Present Seeds | Missing Seeds |
| :--- | :---: | :---: | :--- | :---: | :---: |
| **MNLI** | $C=20$ | $R=50$ | `CRATIO1.0-BDISC-E3TO30` | `[42]` | **`[43, 44]`** (2 runs) |
| **MNLI** | $C=50$ | $R=30$ | `CRATIO1.0-BDISC-E3TO30` | `[42]` | **`[43, 44]`** (2 runs) |
| **QNLI** | $C=20$ | $R=50$ | `CRATIO1.0-BDISC-E3TO30` | `[42]` | **`[43, 44]`** (2 runs) |
| **QNLI** | $C=50$ | $R=30$ | `CRATIO1.0-BDISC-E3TO30` | `[42]` | **`[43, 44]`** (2 runs) |
| **QQP** | $C=20$ | $R=50$ | `CRATIO1.0-BDISC-E3TO30` | `[42]` | **`[43, 44]`** (2 runs) |
| **QQP** | $C=50$ | $R=30$ | `CRATIO1.0-BDISC-E3TO30` | `[42]` | **`[43, 44]`** (2 runs) |
| **SST-2** | $C=20$ | $R=50$ | `CRATIO1.0-BDISC-E3TO30` | `[42]` | **`[43, 44]`** (2 runs) |
| **SST-2** | $C=50$ | $R=30$ | `CRATIO1.0-BDISC-E3TO30` | `[42]` | **`[43, 44]`** (2 runs) |

---

## 3. Seed Deficit Audit: Baselines Suite (`THESIS-RBASE-MAIN`)

In the baseline comparison suite, only **MRPC** and **RTE** currently possess complete 3-seed coverage across all methods. The 4 major datasets have a systematic deficit for **Seed 44**:

| Dataset | Partition | Method | Epochs | Present Seeds | Missing Seeds |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **MNLI** | $C=20, R=50$ | LoRA / RoLoRA | $E \in \{5, 10\}$ | `[42, 43]` | **`[44]`** (4 runs) |
| **MNLI** | $C=50, R=30$ | LoRA / RoLoRA | $E \in \{5, 10\}$ | `[42, 43]` | **`[44]`** (4 runs) |
| **QNLI** | $C=20, R=50$ | LoRA / RoLoRA | $E \in \{5, 10\}$ | `[42, 43]` | **`[44]`** (4 runs) |
| **QNLI** | $C=50, R=30$ | LoRA / RoLoRA | $E \in \{5, 10\}$ | `[42, 43]` | **`[44]`** (4 runs) |
| **QQP** | $C=20, R=50$ | LoRA / RoLoRA | $E \in \{5, 10\}$ | `[42, 43]` | **`[44]`** (4 runs) |
| **QQP** | $C=50, R=30$ | LoRA / RoLoRA | $E \in \{5, 10\}$ | `[42, 43]` | **`[44]`** (4 runs) |
| **SST-2** | $C=20, R=50$ | LoRA / RoLoRA | $E \in \{5, 10\}$ | `[42, 43]` | **`[44]`** (4 runs) |
| **SST-2** | $C=50, R=30$ | LoRA / RoLoRA | $E \in \{5, 10\}$ | `[42, 43]` | **`[44]`** (4 runs) |

---

## 4. Missing Method Comparison: The FFA-LoRA Deficit

In [`FDA_vs_Baselines_Evaluation_Assessment.md`](file:///C:/Users/PC/Documents/GitHub/FDA-LoRA/FDA_vs_Baselines_Evaluation_Assessment.md#L98), FFA-LoRA aggregate numbers are reported (e.g. `0.5791` for E10 and `0.5204` for E5), but these numbers are derived from an incomplete subset of datasets because **FFA-LoRA was never run on the large GLUE benchmarks**:

* **Where FFA-LoRA is Present**:
  * **MRPC** ($C=20, R=30$ and $C=50, R=12$): Complete across $E \in \{5, 10\}$ on Seeds 42, 43, 44.
  * **RTE** ($C=20, R=30$ and $C=50, R=12$): Complete across $E \in \{5, 10\}$ on Seeds 42, 43, 44.
  * **QNLI** ($C=50, R=30$): E5 present on Seed 42 only.
* **Where FFA-LoRA is Completely Missing**:
  * **MNLI** ($C=20, 50$): Missing all seeds for both E5 and E10 (12 runs).
  * **QQP** ($C=20, 50$): Missing all seeds for both E5 and E10 (12 runs).
  * **SST-2** ($C=20, 50$): Missing all seeds for both E5 and E10 (12 runs).
  * **QNLI** ($C=20$): Missing all seeds for both E5 and E10 (6 runs).
  * **QNLI** ($C=50$): Missing Seed 43, 44 for E5, and all seeds for E10 (5 runs).

---

## 5. Dataset & Domain Gap Analysis

1. **MNIST (Non-Linear Model Architecture)**:
   * Used in Section 4.2 of the RoLoRA paper to evaluate rotation dynamics on non-linear MLP/CNN models.
   * In this repository, MNIST exists in `fdaopt/models/toy_model.py` and `results/toy_round_tradeoff_new2fda/`, but has **not been benchmarked under the calibrated dual-track FDA controller** with equivalent production early-stopping criteria.
2. **Missing $E_{\max}=30$ Configurations on Small Datasets**:
   * **MRPC ($C=20, 50$)** and **RTE ($C=20, 50$)** were not included in the `E3TO30` evaluation sweep.
   * *Note*: For RTE, the task suffers from complete collapse at the trivial majority baseline (52.71%) across all architectures; however, testing `E3TO30` on MRPC could evaluate if deeper local exploration resolves small-sample paraphrase plateaus.

---

## 6. Recommended Action Plan for Full Multi-Seed Assessment

To achieve an authoritative, publication-ready thesis chapter with rigorous error bars ($\mu \pm \sigma$):

### Phase 1: Complete 3-Seed Coverage for Primary Comparison (High Impact)
1. **Calibrated FDA ($E_{\max}=20$)**: Launch Seeds 43 and 44 on the winning parameter group `CRATIO1.0-BDISC` across all 12 configurations (**24 runs**).
2. **Calibrated FDA ($E_{\max}=30$)**: Launch Seeds 43 and 44 across the 8 major benchmark configurations (**16 runs**).
3. **RoLoRA & LoRA Baselines**: Launch Seed 44 for $E=5$ and $E=10$ on MNLI, QNLI, QQP, SST-2 across $C \in \{20, 50\}$ (**32 runs**).

### Phase 2: Complete Factorial Ablations & FFA-LoRA (Completeness)
1. Launch Seeds 43 and 44 for the remaining 3 FDA calibration parameter groups (`CRATIO1.0-NODISC`, `CRATIO0.75-BDISC`, `CRATIO0.75-NODISC`) (**72 runs**).
2. Launch FFA-LoRA baseline runs on MNLI, QQP, SST-2, QNLI to balance the 4-method comparison matrix (**47 runs**).
