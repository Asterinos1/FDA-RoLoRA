# Phase 4: Prioritized Experiment Plan for Future GPU Runs

**Author:** Thesis Revision & Audit Team  
**Date:** October 2026  
**Context:** Technical University of Crete Diploma Thesis – *Investigating the Applicability of Federated Dynamic Averaging in Federated Low-Rank Adaptation Architectures*  
**Branch:** `thesis-audit`  
**Location:** `analysis/thesis_final_audit/experiment_plan.md`

---

## 1. Executive Summary & Resource Constraints

This document outlines and ranks potential GPU experiments to address the remaining empirical limitations identified during the thesis audit. 

Given the **immediate thesis deadline (October 2026)** and limited compute availability:
- **Rule of Engagement:** No further GPU experiments are required for thesis submission. The thesis text has been audited and rewritten to honestly represent the existing empirical data ($N=133$ valid runs across 66 comparison cells) with zero claims that exceed the data.
- **Triage Recommendation:** Large-scale re-runs (>100 GPU-hours) must be **deferred to post-thesis journal/conference extensions**. If an emergency compute window of 15–25 GPU-hours opens before camera-ready defense slides, execute strictly **Priority 1** or **Priority 2**.

---

## 2. Empirical Cost Model (from `results/round_metrics/*.csv`)

Based on our empirical analysis of 475 `THESIS-*` runs across 12-layer `roberta-base` on NVIDIA RTX / A100 GPUs:

| Task | Partition | Clients ($C$) | Rounds ($R$) | Method | Mean Sec/Round | Total Time / Run |
|:---|:---|:---:|:---:|:---|:---:|:---:|
| **SST-2** | IID / Non-IID | 20 | 20–30 | RoLoRA | ~965 s | 3.5 – 5.5 hrs |
| **SST-2** | IID / Non-IID | 20 | 20 | FDA-RoLoRA | ~625 s | **3.0 – 3.5 hrs** |
| **QQP** | IID / Non-IID | 20 | 20–30 | RoLoRA | ~1100 s | 5.0 – 6.5 hrs |
| **QQP** | IID / Non-IID | 20 | 20 | FDA-RoLoRA | ~1336 s | **5.5 – 7.0 hrs** |
| **QNLI** | Non-IID | 20 | 30–50 | RoLoRA | ~1764 s | 13.0 – 18.0 hrs |
| **QNLI** | Non-IID | 20 | 30 | FDA-RoLoRA | ~3500 s | **18.0 – 22.0 hrs** |
| **MNLI** | Non-IID | 20 | 42–50 | RoLoRA | ~2355 s | 18.0 – 24.0 hrs |
| **MNLI** | Non-IID | 20 | 42–50 | FDA-RoLoRA | ~2419 s | **20.0 – 25.0 hrs** |

---

## 3. Ranked Experiment Proposals

### Priority 1: Single-Threshold Ablation of FDA-RoLoRA (Dual-Track Validation)
- **Scientific Motivation:** A primary architectural claim of Chapter 3 is that matrix $A$ and matrix $B$ exhibit distinct gradient geometries and require independent threshold trackers ($\theta_A$ and $\theta_B$). While our empirical analysis confirmed factor symmetry in realized epochs ($\bar{e}_A = 10.63 \approx \bar{e}_B = 10.82$), an explicit ablation comparing independent tracking against a shared single threshold $\theta_{\text{shared}} = \frac{1}{2}(\theta_A + \theta_B)$ would formally substantiate whether dual tracking prevents cross-talk or if a simpler single-track controller suffices.
- **Configuration:**
  - Tasks: SST-2 ($C=20$, Non-IID $\alpha=0.5$) and QQP ($C=20$, Non-IID $\alpha=0.5$).
  - Rounds: $R=20$, Seeds: 42, 43.
  - Variant: Single threshold $\theta$ updated every round regardless of active track.
- **Estimated Runtime:**
  - SST-2: $2 \text{ runs} \times 3.2 \text{ hrs} = 6.4 \text{ GPU-hrs}$
  - QQP: $2 \text{ runs} \times 6.0 \text{ hrs} = 12.0 \text{ GPU-hrs}$
  - **Total:** **18.4 GPU-hours**.
- **Value per GPU-Hour:** **Very High (4.8/5.0)**. Directly defends against a committee question on whether the dual-track mechanism is necessary or redundant.
- **Deadline Status:** **Optional if compute permits; otherwise cite empirical tracking symmetry in Chapter 5.5 and leave ablation to future work.**

---

### Priority 2: Missing Second Seed for QNLI $C=50$ RoLoRA $E=20$ (Cell Balance)
- **Scientific Motivation:** In Table 5.4, the QNLI $C=50$ Non-IID cell for RoLoRA $E=20$ currently has $n=1$ (Seed 42 only: final accuracy 61.16%), whereas all FDA-RoLoRA configurations have $n=2$. Completing Seed 43 will balance the cell and provide an exact variance estimate.
- **Configuration:**
  - Task: QNLI ($C=50$, Non-IID $\alpha=0.5$).
  - Setup: RoLoRA $E=20$, Seed 43, $R=30$.
- **Estimated Runtime:** 1 run $\times$ 18.5 hrs = **18.5 GPU-hours**.
- **Value per GPU-Hour:** **High (4.0/5.0)**. Cleans up the single remaining $n=1$ baseline cell in the non-IID tables.
- **Deadline Status:** **Skip for diploma thesis.** The $n=1$ is transparently reported in Table 5.4 and footnotes; the qualitative finding (RoLoRA reaching ~61.16% while FDA-RoLoRA reaches 62.46% at $k=0.5$ and 49.51% at $k=1.0$) is already fully supported.

---

### Priority 3: Re-Running Cells with Mismatched Round Budgets ($R$)
- **Scientific Motivation:** Certain exploratory runs in Non-IID MNLI $C=20$ were executed with $R=42$ (FDA-RoLoRA) vs. $R=30$ or $R=50$ (RoLoRA), with early stopping halting several runs at intermediate rounds ($R \approx 20\text{--}30$). Aligning all runs to an exact fixed round ceiling (e.g., strictly $R=40$) without early-stopping divergence would provide perfectly uniform round-by-round trajectory plots.
- **Configuration:**
  - Tasks: MNLI $C=20$ (2 seeds $\times$ 2 methods = 4 runs).
  - Rounds: Fixed $R=40$.
- **Estimated Runtime:** $4 \text{ runs} \times 22.0 \text{ hrs} = \mathbf{88.0\text{ GPU-hours}}$.
- **Value per GPU-Hour:** **Medium (2.5/5.0)**. Chapter 5 already resolves this cleanly through matched-budget milestone analysis (Table 5.5 and Table 5.6) and matched-round slices.
- **Deadline Status:** **SKIP (Deadline Prohibitive).** The milestone analysis already provides a fair, audit-proof comparison.

---

### Priority 4: Third Seed ($N=3$) for All Primary Benchmark Cells
- **Scientific Motivation:** Moving from $N=2$ (Seeds 42, 43) to $N=3$ (adding Seed 44) across all headline cells (MNLI, QNLI, QQP, SST-2 for both $C=20$ and $C=50$) would shrink error bars and strengthen statistical significance claims.
- **Configuration:**
  - 8 scenario cells $\times$ 2 methods (RoLoRA, FDA-RoLoRA) = 16 full runs.
- **Estimated Runtime:**
  - SST-2: $2 \times 3.5 \text{ hrs} = 7 \text{ hrs}$
  - QQP: $2 \times 6.0 \text{ hrs} = 12 \text{ hrs}$
  - QNLI: $4 \times 18.0 \text{ hrs} = 72 \text{ hrs}$
  - MNLI: $4 \times 24.0 \text{ hrs} = 96 \text{ hrs}$
  - **Total:** **187.0 GPU-hours** (~8 solid GPU-days).
- **Value per GPU-Hour:** **Low to Medium (2.0/5.0)**. $N=2$ with full reporting of standard deviations and individual seed milestones is sufficient for a Master's/Diploma thesis.
- **Deadline Status:** **SKIP.** Must be deferred to post-thesis journal preparation.

---

### Priority 5: Evaluation under Alternative LoRA Ranks ($r \in \{8, 16\}$)
- **Scientific Motivation:** Reviewers may ask whether the variance scaling and controller dynamics observed at $r=4$ hold identically for wider adapters ($r=8, 16$).
- **Configuration:**
  - SST-2 and QQP ($C=20$, Non-IID $\alpha=0.5$, $r \in \{8, 16\}$).
- **Estimated Runtime:** $\approx 40\text{--}60\text{ GPU-hours}$.
- **Value per GPU-Hour:** **Medium (3.0/5.0)**.
- **Deadline Status:** **SKIP.** Scope of this thesis is strictly defined as compact edge LoRA ($r=4, \alpha_{\text{lora}}=8$), as clarified throughout Chapters 1, 2, 4, and 6.

---

## 4. Analytical Formulation of Proposed Controller Fix (Future Work)

Rather than running expensive ad-hoc GPU tuning, Chapter 6 outlines an analytical controller refinement to resolve the sub-linear setpoint tracking ratio ($\mathcal{S}_{\text{IID}} \approx 0.64\text{--}0.79$) and early-round over-tightening.

### The Algorithm: Predictive-Gain Dynamic Averaging (PG-FDA)
In the current implementation:
$$\theta_{\text{next}} = \frac{T}{e^*} \mathrm{Var}(e^*)$$
This rule suffers from two pathologies:
1. **Lagging Sensitivity:** It computes the threshold assuming the subsequent round will exhibit identical per-epoch variance growth as the current round. In non-convex alternating optimization, gradient magnitudes decay as the sub-manifold is approached.
2. **Censored Tightening:** When $e^* = e_{\max}$, the threshold can shrink artificially if $\mathrm{Var}(e_{\max})$ happens to be small, trapping subsequent rounds at low $e$.

### Proposed Refinements:
1. **Same-Track Gradient Drift Factor ($\hat{g}$):**
   Track the epoch-1 coordinate variance ratio between consecutive activations of the *same* track ($A$ or $B$):
   $$\hat{g}_{\text{track}} = \frac{\mathrm{Var}_{\text{track}, r}(e=1)}{\mathrm{Var}_{\text{track}, r-2}(e=1)}$$
   The predictive update rule becomes:
   $$\theta_{\text{next}} = \left(\frac{T}{e^*}\right) \mathrm{Var}(e^*) \cdot \hat{g}_{\text{track}}$$
2. **Monotonic Guard on Censored Rounds:**
   When an agent trains up to $e_{\max}$ without breaching the threshold ($\tau_{\text{halt}} = e_{\max}$):
   $$\theta_{\text{next}} = \max\left(\theta_{\text{current}},\, \frac{T}{e_{\max}} \mathrm{Var}(e_{\max})\right)$$
   This guarantees that non-breaching rounds never contract the client's permissible exploration volume.

*Note: This algorithm is formalized as future theoretical work in Chapter 6 (Section 6.3) and requires zero GPU budget for current submission.*

---

## 5. Summary Recommendation for October 2026 Submission

| Experiment | GPU-Hours | Recommendation |
|:---|:---:|:---|
| **Priority 1: Single-Threshold Ablation** | 18.4 hrs | **Optional** (only if idle GPU exists) |
| **Priority 2: QNLI C50 RoLoRA Seed 43** | 18.5 hrs | **Skip** (already transparent in text) |
| **Priority 3: Re-align Mismatched $R$** | 88.0 hrs | **Skip** (handled via milestone tables) |
| **Priority 4: Third Seed Battery** | 187.0 hrs | **Skip** (defer to conference paper) |
| **Priority 5: Alternative Ranks ($r=8, 16$)** | 50.0 hrs | **Skip** (out of thesis scope) |
| **PG-FDA Analytical Controller Formulation** | 0.0 hrs | **Included in Chapter 6** |
