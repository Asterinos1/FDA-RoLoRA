# Codex CLI prompt: analyze the extracted thesis experiments

You are working inside the `FDA-LoRA` repository. Analyze every experiment CSV currently available in `results/round_metrics` and produce a rigorous report of the thesis experiments completed so far.

Before analyzing the CSVs, locate and read `round_metrics_agent_instructions.md`. Treat that file as the authoritative provenance record explaining which experiment generations are valid, superseded, capped, uncapped, calibration-only, or auxiliary.

## Objective

Create an evidence-based report explaining:

1. how the experimental design evolved;
2. which runs are usable for each analysis;
3. what the toy experiments showed;
4. what the learning-rate calibration showed;
5. what the original uncapped MAIN experiments showed;
6. what the newer 25,000-sample-capped MAIN experiments show so far;
7. whether FDA-RoLoRA reaches useful or higher accuracy earlier than FedAvg-RoLoRA;
8. whether FDA reduces communication rounds, wall-clock time, computation steps, or selected local epochs;
9. where FDA underperforms and whether that is caused by fewer local epochs, fewer completed rounds, instability, or an unfair protocol comparison;
10. which conclusions are supported now and which require additional completed runs.

Do not assume the filenames or CSV schema are perfectly consistent. Inspect the actual columns and representative files before writing the analysis code.

## Data provenance and validity rules

Apply these rules exactly.

### Toy experiments

- `TOY-BASELINE` and `TOY-BASELINES` are valid toy baselines.
- `TOY-FDA` is the original problematic FDA generation and must not enter final aggregates.
- `TOY-NEWFDA` contains duplicate or conflicting runs and must not enter final aggregates.
- `TOY-NEW2FDA` is a corrected intermediate generation. Report it separately if useful.
- `TOY-NEW3FDA` is the final corrected toy FDA generation and is the primary FDA set for comparison against matched toy baselines.
- Never average TOY-FDA, NEWFDA, NEW2FDA, and NEW3FDA together.
- Match toy comparisons by scenario, architecture, method, seed, and round budget.

### Learning-rate, pilot, and controller runs

- `THESIS-RBASE-LR...` runs are valid calibration evidence, not final MAIN results.
- `PILOT` runs are smoke tests and must not enter performance aggregates.
- `THESIS-RBASE-CTRL...` runs are controller/FDA ablations. Analyze them separately and compare only with appropriately matched MAIN FDA runs.

### Original MAIN generation

The original MAIN generation used uncapped datasets, fixed 20 local epochs for LoRA/RoLoRA, adaptive FDA E3TO30, the older stopping policy, seeds 42/43/44, and both rank 4 and some rank-8 configurations.

These runs are valid legacy results, but keep them in a separate `legacy_uncapped` cohort. Do not pool them with the later capped runs.

### New budget-focused MAIN generation

The newer generation used:

- FedAvg-LoRA E5 and E10;
- FedAvg-RoLoRA E5 and E10;
- FDA-RoLoRA-DualTrack E3TO20 with `gamma=0.3` and `theta=0`;
- rank 4 and LoRA alpha 8;
- seeds 42 and 43 in the final core design;
- a 25,000-example training cap for datasets larger than 25,000 examples;
- the newer dataset-specific early-stopping policy.

The cap applies to the new E5, E10, and FDA E3TO20 runs. RTE and MRPC are smaller than the cap and retain their natural dataset size.

For QNLI, MNLI, QQP, and SST-2, the final capped design uses C50-R30 and C20-R50. Earlier capped C20-R75 runs belong to an older round-budget version and must remain separate. Interrupted R75 runs are partial trajectories only.

For RTE and MRPC, C50 uses 12 rounds and C20 uses 30 rounds.

FFA-LoRA and seed 44 were removed from the final new MAIN design. Preserve completed examples as auxiliary evidence, but exclude them from the primary rank-4, seeds-42/43 comparison.

## File integrity and duplicate handling

- Do not modify or delete any source CSV.
- Compute a SHA-256 hash for every CSV.
- Use the CSV `extras` value as the canonical run ID when available; otherwise derive it from the filename.
- Strip only obvious download-copy suffixes such as `(1)` and exporter hash suffixes when identifying duplicate copies.
- If copies have identical hashes, analyze one and record the others as exact duplicates.
- If the same canonical run ID has different contents, do not concatenate or average the files and do not select the copy with the best accuracy.
- Prefer a verified completed run; otherwise prefer the copy with the greatest final round and most valid unique round rows. Document the decision.
- If the same run ID describes different protocols, architectures, or settings, flag it as an identity collision and exclude it until it can be resolved.
- If a CSV contains repeated round rows, retain the last row for each round in the derived analysis and flag the file.
- Empty, unreadable, or malformed files must be listed as excluded instead of silently ignored.

## Completion rules

Classify each run as one of:

- completed at planned horizon;
- completed through valid early stopping;
- incomplete/interrupted;
- malformed;
- unknown completion status.

Use a DONE/completion manifest if one exists. Otherwise infer early stopping only when the available trajectory and configuration provide sufficient evidence. A file ending before its planned maximum is not automatically an early-stopped success.

Incomplete runs may be discussed as interim trajectories but must not contribute to final endpoint or across-seed aggregates.

## Metric handling

- Detect the actual metric column aliases used in the CSVs.
- For MNLI, report matched accuracy, mismatched accuracy, and their mean.
- For MRPC and QQP, report both accuracy and F1.
- For SST-2, QNLI, and RTE, use accuracy as the primary task metric.
- For CoLA, if present, use Matthews correlation.
- Report final metric, best metric, best round, and metrics at common round horizons where possible.
- Treat best test-set accuracy as descriptive. Base primary conclusions on final values, common horizons, or predeclared target thresholds.
- Calculate total runtime from `time_round_total` when available.
- Calculate total computation steps from `round_steps` when available.
- Calculate total selected local epochs from `round_epochs` when available.
- For FDA, use the actual logged `round_epochs`; never estimate its work as `rounds × e_max`.
- If metrics appear to be percentages rather than fractions, normalize them and document the conversion.

## Required comparisons

### Corrected toy analysis

Compare matched `TOY-NEW3FDA` runs against the valid toy baselines. Show final accuracy, peak accuracy, convergence round, steps, local epochs, and runtime where available. Mention NEW2FDA only as a separate intermediate generation.

### Learning-rate calibration

Summarize the LR sweep and explain which learning rates were selected for the later MAIN runs. Do not mix calibration metrics into MAIN averages.

### Original uncapped MAIN analysis

Analyze the uncapped E20 LoRA/RoLoRA and E3TO30 FDA cohort separately. Compare only matched datasets, clients, ranks, seeds, and horizons. Explain how its larger local-epoch and data budgets affect interpretation.

### New capped MAIN analysis

For every dataset and client setting with sufficient completed runs, compare:

- FDA E3TO20 versus RoLoRA E5;
- FDA E3TO20 versus RoLoRA E10;
- FDA E3TO20 versus LoRA E5;
- FDA E3TO20 versus LoRA E10.

Use seeds 42 and 43 for the primary aggregate. Keep C20 and C50 separate. Keep C20-R50 and C20-R75 separate.

For FDA versus RoLoRA, explicitly answer:

- Which method achieved the higher final accuracy?
- Which method reached each shared accuracy threshold first by round?
- Which method reached it first by wall-clock time?
- Which method used fewer computation steps and selected local epochs?
- Did FDA achieve a better accuracy-versus-compute tradeoff even when its final accuracy was lower?
- Are the conclusions consistent across both seeds?

Choose shared thresholds from the observed trajectories rather than using unreachable thresholds. State every threshold used.

### Early-stopping assessment

Assess whether the current stopping gates are realistic based on completed trajectories. Recommend changes only when supported by multiple comparable runs. Do not retroactively claim that an already-running experiment used a threshold introduced after it started.

## Statistical presentation

- Show every individual seed result.
- For matched multi-seed groups, report mean, standard deviation, minimum, maximum, and seed count.
- Never describe a single-seed result as a multi-seed result.
- Do not overstate significance with only two seeds.
- Clearly label partial, auxiliary, legacy, calibration, and ablation evidence.
- Never combine capped and uncapped runs in one average.

## Deliverables

Create a `reports/experiment_analysis_so_far/` directory containing:

1. `thesis_experiment_report.md` — the complete human-readable report.
2. `run_inventory.csv` — one row per physical CSV with parsed configuration, family, validity, completion status, cap status, and exclusion reason.
3. `deduplicated_run_summary.csv` — one row per accepted canonical run with final/best metrics, runtime, steps, epochs, and provenance cohort.
4. `aggregate_results.csv` — matched aggregate statistics by dataset, method, client setting, epoch policy, sample regime, rank, and round budget.
5. `duplicate_and_exclusion_report.md` — all duplicates, conflicts, malformed files, incomplete runs, and excluded experiment generations.
6. `analysis_payload.json` — compact machine-readable evidence used for the written conclusions.
7. `figures/` — a small set of useful plots, including accuracy versus round and, where supported, accuracy versus time or computation steps.

The Markdown report should contain:

- executive summary;
- experiment-history and provenance section;
- dataset and method overview;
- toy-model findings;
- LR-calibration findings;
- original uncapped MAIN findings;
- new capped MAIN findings;
- FDA-versus-RoLoRA convergence and efficiency analysis;
- early-stopping assessment;
- limitations and missing experiments;
- concise thesis-ready conclusions.

Use tables for exact comparisons. Avoid dumping hundreds of rows into the report; put exhaustive details in the CSV/JSON deliverables.

## Verification before finishing

Run the analysis end-to-end and verify that:

- every physical CSV appears in `run_inventory.csv`;
- all output tables reconcile with the number of accepted and excluded files;
- TOY-FDA and TOY-NEWFDA do not enter final corrected aggregates;
- NEW3FDA remains identifiable;
- capped and uncapped MAIN runs are separate;
- E5, E10, E20, E3TO20, and E3TO30 remain separate;
- C20 and C50 remain separate;
- C20-R50 and C20-R75 remain separate;
- incomplete runs do not enter final aggregates;
- the report's numeric claims can be traced to the generated summary tables.

If the repository already contains an exporter or reporting script, inspect and improve it rather than creating a competing implementation unnecessarily. Keep all changes scoped to analysis/report generation. Do not alter training code, experiment CSVs, logs, checkpoints, or scheduler behavior.

At completion, print:

- the number of CSV files discovered;
- accepted canonical runs;
- exact duplicates;
- conflicting duplicates;
- incomplete runs;
- malformed or quarantined runs;
- the paths of every generated deliverable;
- a short summary of the strongest findings and remaining limitations.
