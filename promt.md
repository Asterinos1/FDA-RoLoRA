Perform a complete, reproducible analysis of every experiment result currently available in this repository and produce a polished thesis-style report as both LaTeX source and a successfully compiled PDF.

The report must cover:

1. Corrected toy-model experiments
2. Learning-rate calibration experiments
3. Main RoBERTa-base experiments completed so far
4. Cross-stage conclusions, limitations, and remaining work

Do not stop after inspecting the files or generating intermediate tables. Complete the analysis, write the LaTeX report, compile it, inspect the PDF, correct presentation problems, and deliver the final PDF.

# 1. General requirements

Work directly from the repository’s current files. Recursively inspect all relevant sources, including:

* `results/**/*.csv`
* `results/round_metrics/**/*.csv`
* Experiment manifests and inventories
* JSON and Markdown experiment exports
* Existing analysis reports
* `toy_experiment_runner.py`
* `thesis_experiments.py`
* `run_thesis_experiments.sh`
* `selected_lrs.json`
* Relevant modules under `fdaopt/`
* Relevant `.out` and `.err` logs
* `fda_opt.pdf`, if available
* `rolora_verr4.pdf`, if available
* Any other file needed to establish configuration, implementation history, or completion status

The raw CSVs and the source code associated with each run are authoritative for measured results. Existing reports may help identify context, but their numbers and conclusions must be independently verified.

Do not modify, rename, or delete raw experiment results.

Do not silently resolve ambiguous data. Document conflicts and exclude unresolved records from primary aggregates.

# 2. Reproducible analysis pipeline

Create a reproducible Python script that performs the complete analysis. It must:

* Recursively discover all experiment CSVs.
* Identify the experiment category from filenames, paths, metadata, and columns.
* Normalize inconsistent schemas.
* Normalize method names.
* Parse run names into structured configuration fields.
* Detect duplicate logical configurations.
* Apply the corrected toy-run selection rules below.
* Determine whether each run is completed, early-stopped, interrupted, partial, invalid, superseded, duplicated, or missing.
* Generate per-run metrics.
* Generate aggregate metrics across seeds.
* Export clean inventory and aggregate CSVs.
* Generate all figures used in the report.
* Generate LaTeX-compatible tables or table fragments where useful.
* Be safe to rerun without changing the raw results.

Check for:

* Missing columns
* Repeated CSV headers
* Malformed rows
* Duplicate rounds
* Missing rounds
* NaN or infinite values
* Inconsistent accuracy units
* Non-monotonic timestamps
* Contradictory configuration metadata
* Files containing only partial trajectories
* Files whose names and internal configurations disagree
* Duplicate files representing the same logical run

Detect whether accuracy is represented as 0–1 or 0–100 and normalize all reported results consistently. State the chosen unit in the report.

# 3. Mandatory run inventory

Before aggregating results, create a complete run inventory containing at least:

* Source CSV path
* Run name
* Experiment category
* Dataset or toy scenario
* Method
* FDA variant, where applicable
* Number of clients
* Clients participating per round
* Classes per client for toy experiments
* Partition alpha
* LoRA rank
* LoRA alpha
* Target modules
* Seed
* Client learning rate
* Server learning rate
* Client optimizer
* Server optimizer
* Batch size
* Selected layers
* Planned communication rounds
* Observed communication rounds
* Fixed or adaptive local-epoch policy
* Minimum and maximum allowed local epochs
* Best accuracy
* Final accuracy
* Best round
* Wall-clock duration, if recoverable
* Completion status
* Analysis inclusion status
* Exclusion or replacement reason
* Logical configuration identifier used for deduplication

Validate all run totals against this inventory.

# 4. Critical toy-model correction rules

The early toy FDA results contain known implementation mistakes. They must not be mixed with the corrected results.

Apply these rules:

1. Treat old `TOY-FDA_*` files as deprecated.
2. Treat `TOY-NEWFDA_*` files as deprecated and invalid for final analysis.
3. Do not use an old `NEWFDA` result merely because a corrected counterpart is missing.
4. Treat `TOY-NEW2FDA*` as the corrected FDA-RoLoRA-DualTrack result family.
5. Treat `TOY-NEW3FDA*` as the corrected follow-up family for the FDA SMA/window-3 variant.
6. If the same logical FDA-SMA configuration appears in both `NEW2FDA` and `NEW3FDA`, use the authoritative `NEW3FDA` result and exclude the overlapping `NEW2FDA` result from aggregation.
7. Retain valid non-overlapping `NEW2FDA` Standard, EMA, or other corrected variants where applicable.
8. Do not double-count a corrected result and the erroneous result it replaced.
9. Verify every mapping using run metadata, source code, inventory files, timestamps, and configuration fields.
10. If the evidence conflicts, place the affected files in an unresolved table rather than guessing.
11. Use seeds 42, 43, and 44 where those corrected runs are available.
12. Explain clearly that deprecated runs are excluded due to implementation mistakes and are not scientifically comparable to corrected runs.

Create a dedicated “Toy result lineage” table containing:

* Deprecated family
* Corrected replacement
* Method or variant
* Configuration
* Inclusion decision
* Reason
* Evidence used to establish the decision

# 5. Toy-model report section

Begin by explaining the toy experiment itself. Recover its exact design from the runner and training implementation.

Explain:

* The purpose of the toy experiment
* The synthetic classification task
* The model architecture
* The number of classes
* How data are divided among clients
* The full-participation setting
* The five-client/two-classes-per-client scenario
* The ten-client/one-class-per-client scenario
* Why the ten-client case is more heterogeneous or pathological
* Seeds
* Optimizer
* Learning rate
* Rank
* Planned rounds
* Baseline local-epoch policy
* FDA adaptive local-epoch range
* FDA Standard, SMA, and EMA variants that were actually executed
* How the FDA controller selects local computation
* The implementation errors that made the old result families invalid, but only when these errors can be established from the code or project records
* What was changed in the corrected runs

If a generic `local_epochs` field reports 1 but FDA has actual adaptive epoch counts stored elsewhere, explain the distinction. Use the real per-round FDA epoch values for computation analysis.

For every valid toy configuration, calculate:

* Number of valid seeds
* Individual seed results
* Mean and standard deviation of best accuracy
* Mean and standard deviation of final accuracy
* Best round
* Accuracy at fixed comparable rounds
* Rounds to relevant accuracy thresholds
* Time to relevant thresholds
* Total local epochs
* Mean local epochs per round
* Minimum and maximum local epochs per round
* Distribution of selected epochs
* Cumulative local epochs by communication round
* Optimizer steps, if recorded or reliably derivable
* Processed examples, if reliably derivable
* Wall-clock time
* Area under the accuracy-versus-round curve
* Stability across seeds

Compare:

* Five clients against ten clients
* Baselines against FDA
* FDA Standard against SMA and EMA
* Final performance
* Convergence speed
* Communication efficiency
* Local-computation requirements
* Wall-clock requirements
* Seed sensitivity

Determine whether FDA:

* Saves communication rounds
* Saves local computation
* Uses more computation to improve accuracy
* Redistributes computation across rounds
* Uses fewer epochs over more rounds
* Provides more stable or less stable training

Do not call FDA more efficient merely because it uses fewer communication rounds. Compare total epochs, optimizer steps, processed examples, and wall-clock time.

# 6. Learning-rate calibration section

Explain that the purpose of the LR experiments was to choose viable learning rates for the later main experiments, not to establish final method superiority.

The intended final reduced LR calibration used:

* `roberta-base`
* One local epoch per communication round
* Five calibration communication rounds
* RTE, SST-2, and QNLI
* FedAvg-LoRA and FedAvg-RoLoRA, plus any additional method genuinely present in the calibration files
* FDA reusing the selected RoLoRA learning rate if that is what the source code implements

Verify the actual configuration associated with each LR CSV. Older code may contain abandoned or obsolete LR plans. Do not combine runs from incompatible LR protocols without labelling them separately.

Inventory every tested learning rate and report:

* Dataset
* Method
* Learning rate
* Seed
* Planned and observed rounds
* Best accuracy
* Final accuracy
* Accuracy trajectory
* Instability or divergence
* Completion status
* Inclusion status

Identify:

* Which learning rates were actually tested
* Which runs completed
* Which runs failed or were interrupted
* Missing dataset/method/LR combinations
* The best-performing LR within the tested grid
* Whether the selected LR was a tested interior value or a boundary value
* Whether neighbouring values support the selection
* Whether the short five-round trajectory was sufficiently informative

The main experiment uses:

* RTE: 0.05 for LoRA, RoLoRA, and FDA Standard
* SST-2: 0.1
* QNLI: 0.1

Explain whether the calibration evidence supports those selections.

Explicitly explain that LR calibration used one local epoch per round, whereas the main baselines use 20 local epochs per round. Discuss this as an experimental limitation because the best LR under one local epoch may not necessarily remain optimal under 20 local epochs.

Do not retroactively present the LR calibration as a complete hyperparameter optimization study.

# 7. Main RoBERTa-base experiment section

The planned main experiment contains 81 configurations.

The design is:

* Model: `roberta-base`
* Methods:

  * FedAvg-LoRA
  * FedAvg-RoLoRA
  * FDA-RoLoRA-DualTrack
* Datasets:

  * RTE
  * SST-2
  * QNLI
* Seeds: 42, 43, 44
* Partition alpha: 1000
* Full participation
* Client settings:

  * C20/K20
  * C50/K50
* LoRA:

  * C20: rank 4, alpha 8
  * C50: rank 4/alpha 8 and rank 8/alpha 16
  * Target modules: `query` and `value`
* Client optimizer: SGD
* Server optimizer: SGD
* Server learning rate: 1
* Baselines: fixed 20 local epochs per round
* FDA: adaptive 3–30 local epochs per round
* FDA gamma: 0.3
* FDA theta: 0
* Early stopping for all methods:

  * Activated only after accuracy reaches 90%
  * Patience: 5
  * Minimum improvement: 0.01
  * Window: 3

Dataset configurations:

| Dataset | Batch size | Layers      | C20 rounds | C50 rounds | Learning rate |
| ------- | ---------: | ----------- | ---------: | ---------: | ------------: |
| RTE     |         32 | 8,9,10,11   |         30 |         12 |          0.05 |
| SST-2   |         64 | 9,10,11     |         75 |         30 |           0.1 |
| QNLI    |         32 | 7,8,9,10,11 |         75 |         30 |           0.1 |

Verify these settings against the runner and individual result metadata.

A previous known checkpoint contained 56 completed configurations out of 81:

* RTE: 27/27
* SST-2: 27/27
* QNLI: 2/27 completed
* Four QNLI configurations running
* Twenty-one QNLI configurations queued

This is only a historical sanity check. Recalculate the current status from the filesystem because more experiments may now be complete.

Classify every planned configuration as:

* Completed to the planned round count
* Completed through valid early stopping
* Interrupted
* Partial or incomplete
* Missing or not yet discovered
* Duplicate
* Superseded
* Invalid

Only completed or validly early-stopped runs may enter principal final-result aggregates.

Partial runs may be included in a separate provisional trajectory section, but:

* Do not call the last observed accuracy of a partial run its definitive final result.
* Do not rank partial runs against completed runs using final accuracy.
* Do not combine incomplete seed sets without reporting the number of available seeds.
* Do not fill missing seeds or rounds with zero.
* Do not extend early-stopped trajectories by forward-filling them for the principal analysis.

For every completed main run, report:

* Dataset
* Method
* Clients
* Rank
* Seed
* Best accuracy
* Final accuracy
* Best round
* Accuracy at fixed comparable rounds
* Rounds to 80%, 85%, and 90%
* Time to 80%, 85%, and 90%
* Thresholds never reached
* Area under the accuracy-versus-round curve
* Total local epochs
* Mean local epochs per round
* Minimum and maximum local epochs for FDA
* Total optimizer steps, if available
* Processed examples, if available
* Wall-clock time
* Early-stopping status
* Early-stopping round and reason

Aggregate across seeds while always reporting:

* Number of available seeds
* Individual seed values
* Mean
* Standard deviation
* Missing seeds

Do not report a standard deviation as meaningful when only one seed is available.

Make controlled comparisons only between runs with matching:

* Dataset
* Client count
* Rank
* Seed
* Relevant training configuration

Calculate for every controlled FDA-versus-baseline comparison:

* Best-accuracy difference
* Final-accuracy difference
* Communication-round difference
* Total-local-epoch ratio
* Optimizer-step ratio, if possible
* Wall-clock-time ratio
* Time-to-threshold difference
* Whether the comparison is complete or provisional

Pay particular attention to:

* RTE runs where 90% accuracy is never reached
* Whether early stopping affects each dataset
* Whether FDA performs fewer total local epochs
* Whether FDA uses fewer epochs over more communication rounds
* Whether FDA gains accuracy by spending more local computation
* C20 versus C50
* Rank 4 versus rank 8 at C50
* LoRA versus RoLoRA
* RoLoRA versus FDA-RoLoRA
* Seed variance
* Differences between best and final accuracy
* Cases where round efficiency and wall-clock efficiency disagree

# 8. Cross-stage analysis

After the Toy, LR, and Main sections, synthesize the evidence.

Address:

* What the corrected toy model demonstrated
* Which toy-model observations transferred to RoBERTa
* Which toy observations did not transfer
* What the LR calibration established
* What the LR calibration could not establish
* Where FDA-RoLoRA currently appears beneficial
* Where it appears neutral
* Where it appears worse
* Whether its main value is accuracy, communication efficiency, adaptive computation allocation, stability, or another property
* Whether any advantage remains after controlling for total local computation and wall-clock time
* What remains unknown because of incomplete QNLI experiments
* Which conclusions are strong enough for the thesis
* Which conclusions must remain provisional

Separate evidence from interpretation. Use explicit language such as:

* “The measured results show…”
* “This suggests…”
* “The current evidence does not establish…”
* “This comparison is provisional because…”

Do not turn a hypothesis into a confirmed conclusion.

# 9. Required tables

Include at least:

1. Complete experiment inventory summary
2. Toy-result lineage and replacement table
3. Excluded and unresolved toy files
4. Toy experimental design
5. Toy per-run results
6. Toy aggregate results with seed counts
7. LR inventory
8. LR results by dataset, method, and learning rate
9. Selected-LR justification
10. Main 81-configuration completion matrix
11. Main per-run results
12. Main aggregates with `n`, mean, and standard deviation
13. Threshold-reaching results
14. Local-computation comparison
15. Wall-clock comparison
16. Early-stopping summary
17. Missing, partial, or interrupted configurations
18. Confirmed findings
19. Provisional findings
20. Threats to validity

Large inventories should appear in appendices rather than interrupting the main discussion.

# 10. Required figures

Generate clear figures where supported by the data:

* Accuracy versus communication round
* FDA local epochs versus communication round
* Cumulative local epochs versus accuracy
* Accuracy versus cumulative local epochs
* Accuracy versus wall-clock time
* Learning rate versus best accuracy
* Learning rate versus final accuracy
* Mean accuracy trajectories with seed variability
* Best/final accuracy comparison by method
* Total-computation comparison by method

Separate or facet plots by dataset and controlled configuration. Do not place incompatible configurations on the same plot without clear labelling.

Use consistent colors and method names throughout the report.

Prefer vector PDF figures. Use high-resolution PNG only when vector output is impractical.

# 11. Statistical and numerical rules

* Normalize accuracy units consistently.
* State whether results are percentages or proportions.
* Report the seed count for every aggregate.
* Show individual seed values alongside aggregate results where practical.
* Preserve enough numerical precision for verification.
* Avoid declaring a winner based on tiny differences without discussing variance.
* Do not average incompatible datasets into a single headline number.
* Do not treat communication rounds as equivalent to computation.
* Do not infer optimizer steps unless the calculation is fully supported by batch size, data size, client participation, and epoch records.
* Mark derived metrics separately from directly recorded metrics.
* Validate every number appearing in the prose against a generated table.

# 12. LaTeX report

Create a complete thesis-style LaTeX report.

The primary deliverables are:

* `results_analysis_so_far.tex`
* `results_analysis_so_far.pdf`

The LaTeX report must include:

* Title page
* Generation date
* Abstract or executive summary
* Table of contents
* List of tables
* List of figures
* Introduction and experimental motivation
* Data-selection and validation methodology
* Toy-model methodology and results
* LR-calibration methodology and results
* Main-experiment methodology and interim results
* Cross-stage discussion
* Threats to validity
* Remaining work
* Conclusions
* Bibliography
* Appendices

Use `fda_opt.pdf` and `rolora_verr4.pdf`, if available, for theoretical background and cite them appropriately. Do not present measurements from those papers as results of the current experiments.

Use an academic layout and suitable packages, such as:

* `booktabs`
* `longtable`
* `tabularx`
* `adjustbox`
* `siunitx`
* `graphicx`
* `subcaption`
* `hyperref`
* `cleveref`
* `xcolor`
* `listings` or `fvextra`

Escape all LaTeX-sensitive characters in run names, paths, method identifiers, and configuration strings.

Every table and figure must:

* Have a descriptive caption
* Have a unique label
* Be referenced in the text
* Fit within the page margins
* Be legible at normal PDF zoom
* State or make clear the number of seeds represented
* Mark provisional or incomplete results
* Be interpreted in the surrounding prose

Use multi-page tables or appendices for large inventories. Do not shrink text until it becomes unreadable.

# 13. PDF compilation and visual validation

Compilation is part of the task.

Use `latexmk` if available. Otherwise use an installed LaTeX engine and run enough passes to resolve:

* Table of contents
* Cross-references
* Citations
* Bibliography
* Figure and table numbering

If no LaTeX engine is installed and package installation is permitted, install or use an appropriate LaTeX compiler. Do not claim that the PDF was created unless a valid PDF file exists.

Check the compilation log for:

* Fatal errors
* Undefined references
* Undefined citations
* Missing figures
* Missing fonts or packages
* Serious overfull or underfull boxes
* Broken tables
* Clipped text

Render or visually inspect the resulting PDF. Correct:

* Tables extending outside margins
* Unreadable text
* Overlapping labels
* Low-resolution figures
* Figures placed far from their discussion
* Empty pages
* Bad page breaks
* Malformed equations
* Incorrect references
* Missing bibliography entries
* Inconsistent formatting

Recompile after making corrections.

Before finishing, cross-check every important number in the PDF against the generated analysis CSVs.

# 14. Output structure

Create a dedicated output directory, for example:

`results/analysis_so_far/`

Place inside it:

* `results_analysis_so_far.tex`
* `results_analysis_so_far.pdf`
* `analyze_results_so_far.py`
* `analysis_run_inventory.csv`
* `analysis_aggregates.csv`
* `analysis_exclusions.csv`
* Generated LaTeX table fragments, if used
* A `figures/` directory
* A compilation log or short build record

Do not place generated files among the raw experiment CSVs.

# 15. Completion criteria

Do not finish until:

1. Every discovered CSV has an inventory decision.
2. Deprecated toy FDA results have been excluded.
3. Corrected `NEW2FDA` and `NEW3FDA` lineage has been handled without double-counting.
4. Toy, LR, and Main results have been analyzed.
5. Completed and partial runs have been separated.
6. All aggregates have been validated.
7. The LaTeX source has been generated.
8. The PDF has compiled successfully.
9. The PDF has been visually inspected.
10. Important layout and reference problems have been corrected.
11. The final files exist in the output directory.

In your final response, report:

* Path to the final PDF
* Path to the LaTeX source
* Path to the reproducible analysis script
* Current completed/partial/missing main-run counts
* Number of included and excluded CSVs
* LaTeX engine used
* Whether compilation and visual inspection succeeded
* Any unresolved data-quality issues
* A concise summary of the most important experimental findings

The final PDF is the primary deliverable. Do not stop after producing only summaries, CSVs, figures, Markdown, or uncompiled LaTeX.
