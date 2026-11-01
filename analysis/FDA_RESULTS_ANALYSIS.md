# FDA-LoRA Results Analysis

## Part I: Overall FDA versus LoRA, RoLoRA, and FFA-LoRA

### Big verdict

FDA is not an overall accuracy winner yet.

Best defensible claim:

> FDA is useful as an adaptive-compute version of RoLoRA. It performs best under severe client heterogeneity with nonlinear models. On real SST-2, it approaches baseline accuracy using fewer total local steps. It does not beat LoRA on low-resource RTE or MRPC.

Claims about FDA outperforming baselines on linear toy models are currently unsafe.

### Data audit

I inspected all 421 CSVs.

- 25 different schemas exist because pilots, sweeps, historical runs, and final experiments are mixed together.
- 240 older FDA toy files have exactly duplicated accuracy trajectories between LIN and NONLIN runs.
- Current dual-track FDA constructors do not pass the `linear` option to `ToyModel`; the model therefore defaults to nonlinear. See [`fdaopt/training/fda_opt.py`](fdaopt/training/fda_opt.py), compared with the correct baseline construction in [`fdaopt/training/fed_opt.py`](fdaopt/training/fed_opt.py) and the default in [`fdaopt/models/toy_model.py`](fdaopt/models/toy_model.py).
- Therefore, FDA files labelled LIN cannot support linear-model conclusions until rerun.
- The existing [`analysis/thesis_rbase_main_analysis/MASTER_ANALYSIS_REPORT.md`](analysis/thesis_rbase_main_analysis/MASTER_ANALYSIS_REPORT.md) is stale: it analyzes 36 runs and omits 18 newer SST-2 baseline CSVs.
- SST-2 stopping configurations are inconsistent. For example, C50/r4 seed 42 LoRA and RoLoRA ran 30 rounds without early stopping; seeds 43–44 used early stopping and ended after 8–12 rounds.
- Only three seeds are available. Small differences are not strong evidence.

### Real RoBERTa-base results

Peak accuracy, mean ± sample standard deviation over three paired seeds:

| Setting | FDA | LoRA | RoLoRA | Result |
|---|---:|---:|---:|---|
| RTE, C20, r4 | 54.39 ± 2.61 | **59.21 ± 1.08** | 54.63 ± 1.82 | FDA ties RoLoRA, loses LoRA by 4.81 points |
| RTE, C50, r4 | 50.90 ± 3.13 | 53.91 ± 2.08 | **54.51 ± 3.13** | FDA fails to learn beyond chance/class-prior behavior |
| RTE, C50, r8 | 50.90 ± 3.13 | **52.95 ± 0.42** | **52.95 ± 0.42** | Same failure |
| SST-2, C20, r4 | 92.43 ± 0.20 | **92.55 ± 0.12** | **92.55 ± 0.30** | Practical tie |
| SST-2, C50, r4 | 91.55 ± 0.76 | **92.28 ± 0.76** | 92.01 ± 0.57 | FDA trails by 0.46–0.73 points |
| SST-2, C50, r8 | 91.17 ± 0.11 | **92.32 ± 0.00** | 91.97 ± 0.34 | FDA consistently trails |

Other evidence:

- QNLI FDA reaches 86.65 ± 0.76 final accuracy, but only two FDA seeds exist and there are no matched main baselines.
- In the one-seed MRPC pilot, LoRA wins decisively: F1 88.81 versus 82.48 FDA-EMA, 82.14 FDA-Standard, 81.83 FFA-LoRA, and 81.22 RoLoRA.
- Therefore, there is currently no credible real-task evidence that FDA beats LoRA.
- There is mild evidence that FDA improves over plain RoLoRA while reducing work.

#### Efficiency result

On SST-2 C50:

- FDA reaches its peak using about 3.75k local steps for r4 versus 5.13k LoRA and 6.89k RoLoRA.
- For r8: 2.98k FDA versus 4.11k LoRA and 5.43k RoLoRA.
- But FDA’s peak is 0.5–1.2 accuracy points lower.
- LoRA reaches 90% much sooner: roughly 2–3 communication rounds versus 12–14 for FDA.
- FDA needs about the same or fewer steps than RoLoRA to reach 90%, but many more communication rounds.

Thus FDA offers a compute/accuracy trade against RoLoRA. It does not improve communication efficiency against LoRA.

### Valid nonlinear toy results

These are the strongest FDA results:

| Nonlinear setting | Best FDA final | LoRA | RoLoRA | FFA-LoRA | FDA local-step cost |
|---|---:|---:|---:|---:|---:|
| 10 clients, 1 class/client | **FDA-EMA: 72.47 ± 0.85** | 67.01 ± 2.07 | 69.40 ± 3.58 | 69.14 ± 4.42 | 2.24× baseline |
| 5 clients, 2 classes/client | **FDA-Standard: 76.89 ± 4.65** | 75.07 ± 1.05 | 75.73 ± 5.39 | 59.22 ± 2.03 | 1.70× baseline |

Important interpretation:

- FDA clearly beats FFA-LoRA in the 5-client setting: +17.67 points.
- FDA-EMA is stable in 10c1: standard deviation 0.85 versus 2.07–4.42 for baselines.
- FDA does not materially improve peak accuracy over RoLoRA:
  - 75.49 FDA versus 75.24 RoLoRA in 10c1.
  - 80.07 FDA versus 79.93 RoLoRA in 5c2.
- Its final advantage mainly comes from less late-round degradation.
- At equal local-step budgets, FDA-EMA loses by approximately 3.3–5.7 points in 10c1.
- At equal compute in 5c2, FDA-Standard loses LoRA/RoLoRA by approximately 3.4–4.0 points. FDA-SMA is approximately tied, but has poor late stability.

So FDA wins per communication round only because it performs 1.7–2.24 times more local work.

### Where FDA fails and why

1. **Low-resource RTE**

   With 50 clients and only 12 rounds, each FDA factor is updated only six times because matrices alternate. FDA uses about 197–206 total steps versus 480 for baselines. Accuracy and test loss remain near chance and 0.693. This is undertraining.

2. **Nearly IID real experiments**

   Every main RoBERTa experiment uses $\alpha=1000$, full client participation, and all clients every round. This removes much of the heterogeneity and sampling noise FDA is designed to react to. FDA then pays for alternating updates and controller complexity without receiving the expected drift-control benefit.

3. **LoRA learns faster**

   LoRA updates both factors every round. FDA updates one factor at a time. On SST-2, LoRA reaches 90% in 2–3 rounds; FDA needs 9–14 rounds.

4. **EMA is not universally better**

   EMA is best in extreme 10c1 heterogeneity, where smoothing likely reduces noisy threshold changes. In 5c2, FDA-EMA uses about 194k steps but finishes at 71.41%; FDA-Standard uses 159k and reaches 76.89%. EMA lag appears to allocate excessive or mistimed local work.

5. **Pure EMA and SMA can collapse late**

   Clean pure-EMA 5c2 runs peak at 74.40 but finish at 66.00 ± 7.71 after 224k steps. FDA-SMA has a 6.54-point mean peak-to-final drop; one seed loses 13.94 points after its peak.

6. **Linear evidence is invalid**

   The dual-track implementation does not forward the linear-model flag. Do not use the current LIN FDA tables in a paper.

### Experiments that can credibly demonstrate FDA’s value

1. **Fix and rerun the toy suite**

   Pass `linear=hyperparams.get("linear", False)` in all four dual-track constructors. Add an assertion recording the executed model architecture. Rerun the 66-run benchmark from clean output directories.

2. **Test the problem FDA was built for**

   Use real tasks with:

   - $\alpha \in \{0.1, 1, 1000\}$
   - 50 clients
   - 10 clients per round
   - SST-2 and QNLI
   - RoBERTa-base, rank 4
   - LoRA, RoLoRA, FFA-LoRA, FDA-Standard, FDA-EMA
   - At least five paired seeds

   The current full-participation, almost-IID setup is a weak test of FDA.

3. **Use three separate fairness views**

   Report:

   - Accuracy versus communication rounds
   - Accuracy versus cumulative local steps/examples
   - Accuracy versus wall time and transmitted bytes

   FDA may be valuable when communication is expensive but local computation is cheap. That should be the explicit claim.

4. **Run the decisive budget ablation**

   Compare:

   - Fixed-epoch RoLoRA
   - RoLoRA using FDA’s exact recorded epoch schedule
   - FDA with adaptive schedule
   - FDA schedule shuffled across rounds
   - FDA with fixed mean epochs

   This reveals whether gains come from intelligent timing or simply extra computation.

5. **Run a distribution-shift experiment**

   Change client label distributions halfway through training, or replace participating clients at round 30. Measure recovery time and accuracy loss. This is probably FDA’s strongest potential experiment because a fixed epoch policy cannot react, while FDA should change its local budget.

6. **Repair the RTE experiment**

   Use 30–60 rounds rather than 12 and test $E_{\min}\in\{3,5,10\}$. Twelve rounds give each factor only six communication updates. Keep the current result as a negative control showing FDA fails under an insufficient horizon.

7. **Disable early stopping for the core comparison**

   First run every method for the same maximum horizon. Apply one identical stopping rule afterward. Then perform a separate early-stopping experiment. This removes the current seed-dependent configuration mismatch.

Bottom line: FDA has promising stability and adaptive-compute behavior against RoLoRA under severe heterogeneity. Current evidence does not support “FDA beats LoRA generally.” The strongest honest research direction is “FDA gives a better accuracy/communication trade under non-IID drift,” followed by real non-IID, partial-participation, equal-budget experiments.

---

## Part II: SST-2 Learning-Rate Experiments

The LR sweep tests a different claim: baselines are forced to one local epoch, while FDA can grow its epoch budget from 3 up to 30. FDA is winning because adapting the local epoch budget is part of the method. The baselines are fixed at one epoch and undertrain. Under a fixed communication budget, FDA is clearly better.

### SST-2 LR sweep

All runs use 45 rounds and seed 42.

| Learning rate | FDA peak | LoRA E=1 | RoLoRA E=1 | FDA mean epochs |
|---:|---:|---:|---:|---:|
| 0.0005 | 50.92 | 49.08 | 49.08 | 11.53 |
| 0.002 | 50.92 | **51.03** | 49.08 | 12.64 |
| 0.01 | **92.32** | 50.92 | 50.92 | 12.51 |
| 0.05 | **93.12** | 90.71 | 89.22 | 12.76 |
| 0.10 | **92.89** | 91.28 | 91.86 | 13.09 |

Best tuned result:

- FDA: 93.12% peak and 92.55% final at LR 0.05.
- LoRA: 91.28% peak and 91.17% final at LR 0.10.
- RoLoRA: 91.86% peak and 91.40% final at LR 0.10.

Therefore, after tuning LR independently, FDA gains:

- +1.84 peak points over LoRA.
- +1.26 peak points over RoLoRA.
- +1.38 final points over LoRA.
- +1.15 final points over RoLoRA.

That is a real win under the experiment’s intended setting.

### FDA is also more LR-robust

This is possibly more interesting than the best accuracy.

- FDA learns successfully at LR 0.01, 0.05, and 0.10.
- LoRA succeeds only at 0.05 and 0.10.
- RoLoRA succeeds only at 0.10.
- At LR 0.01, FDA reaches 92.32%, while both baselines remain at 50.92%.

Why? Local optimization exposure scales roughly with $\eta E$. At LR 0.01:

- FDA uses approximately 12.5 epochs per round, giving $\eta E \approx 0.125$.
- Baselines use one epoch, giving $\eta E=0.01$.

FDA automatically compensates for a smaller learning rate by allocating more local work. The fixed-E baselines cannot escape undertraining.

### Communication efficiency

At LR 0.10:

- FDA reaches 90% at round 9.
- LoRA reaches 90% at round 13.
- RoLoRA reaches 90% at round 31.

FDA therefore uses:

- 31% fewer communication rounds than LoRA.
- 71% fewer communication rounds than RoLoRA.

At LR 0.05:

- FDA reaches 90% at round 11 and 92% at round 17.
- LoRA reaches 90% at round 25 and never reaches 92%.
- RoLoRA never reaches 90%.

Only FDA reaches 92% anywhere in this sweep.

That supports this claim:

> FDA automatically increases local computation when one fixed local epoch is insufficient, improving both final accuracy and accuracy per communication round.

### The compute tradeoff

FDA is not more optimizer-step efficient in this experiment:

- Baselines: 2,385 total steps.
- FDA: 27,507–31,217 total steps.
- FDA therefore uses about 11.5–13.1 times more local steps.

At the same 2,385-step budget:

- FDA at LR 0.05 has only completed four rounds and remains near chance.
- FDA at LR 0.10 has completed six rounds and reaches 81.54%.
- LoRA at LR 0.10 completes all 45 rounds and reaches 91.17%.

So both statements are true:

- FDA wins under equal communication rounds.
- Fixed-E LoRA wins under equal local-step compute.

That is not automatically unfair. It means FDA targets a communication-expensive, local-compute-cheap setting.

### What the controller is doing

FDA’s early epoch schedules change with LR:

- LR 0.01 spikes to 26, 25, and 30 epochs during early rounds.
- LR 0.05 jumps to 30 epochs at round 4.
- LR 0.10 jumps to 24 epochs at round 4.

This is good evidence that the controller detects insufficient progress or drift and increases local work.

However, all LR configurations eventually settle around 15 epochs. Since $E_{\max}=30$, this is exactly the controller midpoint $E_{\max}/2$. That raises the key scientific question:

> Is FDA’s timing important, or would fixed E=13–15 perform just as well?

Current CSVs cannot answer that.

### Strongest experiment to run next

Run the same SST-2 LR sweep with:

- LoRA fixed $E\in\{1,3,5,10,13,15,20\}$
- RoLoRA fixed $E\in\{1,3,5,10,13,15,20\}$
- FDA dynamic $E\in[3,30]$
- RoLoRA replaying FDA’s exact per-round epoch schedule
- Five paired seeds

Report both accuracy-versus-rounds and accuracy-versus-steps.

Possible outcomes:

- If FDA beats fixed E=13–15 at comparable steps, the controller timing works.
- If replaying FDA’s schedule beats fixed E, adaptive scheduling matters.
- If fixed E=13 matches FDA, the current gain comes primarily from allocating more epochs.
- If FDA reaches the same accuracy with fewer steps than fixed E=20, it still has a strong automatic-compute-selection result.

The accurate headline is not simply “FDA uses extra compute.” It is:

> FDA detects when a fixed one-epoch schedule undertrains and converts additional local computation into substantially better accuracy, broader LR robustness, and fewer communication rounds.

The remaining experiment is proving that FDA chooses those extra epochs better than a tuned fixed-epoch baseline.
