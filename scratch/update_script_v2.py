import re

with open("update_benchmark_suite.py", "r", encoding="utf-8") as f:
    code = f.read()

# Define pure_configs and pure_dataset_metadata after all_configs
pure_defs = '''
pure_configs = [
    # MNLI IID
    ("01_MNLI_C20_R50_LR0.1",  "MNLI ($C=20, R=50, \\\\text{lr}=0.1, \\\\alpha=1000$)",  "mnli", 20, 50, "0.1",  [0.65, 0.72, 0.77], ("R@65%", "R@72%", "R@77%")),
    ("03_MNLI_C50_R30_LR0.1",  "MNLI ($C=50, R=30, \\\\text{lr}=0.1, \\\\alpha=1000$)",  "mnli", 50, 30, "0.1",  [0.55, 0.65, 0.72], ("R@55%", "R@65%", "R@72%")),
    # QNLI IID
    ("05_QNLI_C20_R50_LR0.1",  "QNLI ($C=20, R=50, \\\\text{lr}=0.1, \\\\alpha=1000$)",  "qnli", 20, 50, "0.1",  [0.80, 0.83, 0.85], ("R@80%", "R@83%", "R@85%")),
    ("07_QNLI_C50_R30_LR0.1",  "QNLI ($C=50, R=30, \\\\text{lr}=0.1, \\\\alpha=1000$)",  "qnli", 50, 30, "0.1",  [0.75, 0.82, 0.86], ("R@75%", "R@82%", "R@86%")),
    # QQP IID
    ("09_QQP_C20_R50_LR0.1",   "QQP ($C=20, R=50, \\\\text{lr}=0.1, \\\\alpha=1000$)",   "qqp",  20, 50, "0.1",  [0.65, 0.74, 0.76], ("R@65%", "R@74%", "R@76%")),
    ("10_QQP_C50_R30_LR0.1",   "QQP ($C=50, R=30, \\\\text{lr}=0.1, \\\\alpha=1000$)",   "qqp",  50, 30, "0.1",  [0.75, 0.78, 0.80], ("R@75%", "R@78%", "R@80%")),
    # SST-2 IID
    ("11_SST2_C20_R50_LR0.1",  "SST-2 ($C=20, R=50, \\\\text{lr}=0.1, \\\\alpha=1000$)", "sst2", 20, 50, "0.1",  [0.85, 0.90, 0.924], ("R@85%", "R@90%", "R@92.4%")),
    ("12_SST2_C50_R30_LR0.1",  "SST-2 ($C=50, R=30, \\\\text{lr}=0.1, \\\\alpha=1000$)", "sst2", 50, 30, "0.1",  [0.85, 0.90, 0.91], ("R@85%", "R@90%", "R@91%")),
    # MNLI Non-IID
    ("17_MNLI_C20_R50_LR0.1_A0.5", "MNLI Non-IID ($C=20, R=50, \\\\text{lr}=0.1, \\\\alpha=0.5$)", "mnli", 20, 50, "0.1", [0.65, 0.72, 0.74], ("R@65%", "R@72%", "R@74%")),
    ("19_MNLI_C50_R30_LR0.1_A0.5", "MNLI Non-IID ($C=50, R=30, \\\\text{lr}=0.1, \\\\alpha=0.5$)", "mnli", 50, 30, "0.1", [0.50, 0.53, 0.55], ("R@50%", "R@53%", "R@55%")),
    # QNLI Non-IID
    ("18_QNLI_C20_R50_LR0.1_A0.5", "QNLI Non-IID ($C=20, R=50, \\\\text{lr}=0.1, \\\\alpha=0.5$)", "qnli", 20, 50, "0.1", [0.80, 0.83, 0.84], ("R@80%", "R@83%", "R@84%")),
    ("20_QNLI_C50_R30_LR0.1_A0.5", "QNLI Non-IID ($C=50, R=30, \\\\text{lr}=0.1, \\\\alpha=0.5$)", "qnli", 50, 30, "0.1", [0.55, 0.60, 0.65], ("R@55%", "R@60%", "R@65%")),
    # QQP Non-IID
    ("21_QQP_C20_R20_LR0.1_A0.5",  "QQP Non-IID ($C=20, R=20, \\\\text{lr}=0.1, \\\\alpha=0.5$)",  "qqp",  20, 20, "0.1", [0.65, 0.72, 0.76], ("R@65%", "R@72%", "R@76%")),
    ("22_QQP_C50_R24_LR0.1_A0.5",  "QQP Non-IID ($C=50, R=24, \\\\text{lr}=0.1, \\\\alpha=0.5$)",  "qqp",  50, 24, "0.1", [0.65, 0.70, 0.73], ("R@65%", "R@70%", "R@73%")),
    # SST-2 Non-IID
    ("23_SST2_C20_R25_LR0.1_A0.5", "SST-2 Non-IID ($C=20, R=25, \\\\text{lr}=0.1, \\\\alpha=0.5$)", "sst2", 20, 25, "0.1", [0.80, 0.85, 0.88], ("R@80%", "R@85%", "R@88%")),
    ("24_SST2_C50_R20_LR0.1_A0.5", "SST-2 Non-IID ($C=50, R=20, \\\\text{lr}=0.1, \\\\alpha=0.5$)", "sst2", 50, 20, "0.1", [0.55, 0.65, 0.80], ("R@55%", "R@65%", "R@80%")),
]

pure_dataset_metadata = [
    ("mnli", "MNLI (Multi-Genre Natural Language Inference)", 
     "Sentence-pair entailment classification across diverse genres (392,702 total pairs; capped at 25,000 samples = 6.37% used, 3 classes: entailment, neutral, contradiction)."),
    ("qnli", "QNLI (Question Natural Language Inference)", 
     "Binary sentence-pair classification determining whether a passage sentence contains the answer to a question (104,743 total pairs; capped at 25,000 samples = 23.87% used, 2 classes: entailment, not_entailment)."),
    ("qqp",  "QQP (Quora Question Pairs)", 
     "Binary semantic equivalence classification identifying whether two community questions from Quora are duplicates (363,846 total pairs; capped at 25,000 samples = 6.87% used, 2 classes: duplicate, not_duplicate)."),
    ("sst2", "SST-2 (Stanford Sentiment Treebank)", 
     "Single-sentence binary sentiment classification (positive vs. negative) of movie review excerpts (67,349 total sentences; capped at 25,000 samples = 37.12% used, 2 classes: positive, negative)."),
]
'''

idx_all_configs = code.find('all_configs = [')
idx_dataset_metadata = code.find('dataset_metadata = [')
# insert pure_defs before dataset_metadata
code = code[:idx_dataset_metadata] + pure_defs.strip() + "\n\n" + code[idx_dataset_metadata:]

# Update get_methods definition to accept pure_only=False
old_get_methods = 'def get_methods(ds, c, r, lr, include_lora=True, cfg_prefix=None):'
new_get_methods = 'def get_methods(ds, c, r, lr, include_lora=True, cfg_prefix=None, pure_only=False):'
code = code.replace(old_get_methods, new_get_methods, 1)

# In get_methods, handle pure_only=True for IID
old_methods_def = '''    methods = [
        ("FDA E3TO40 [Emax/1]", find_fda_k1_matches(ds, c, r, lr, 4, "E3TO40"), False),'''

new_methods_def = '''    if pure_only:
        methods = [
            ("FDA Pure E1TO20 [k=1.0]", find_fda_pure_matches(ds, c, r, lr, 4, "1"), False),
            ("FDA Pure E1TO20 [k=0.5]", find_fda_pure_matches(ds, c, r, lr, 4, "0.5"), False),
            ("RoLoRA E20", find_e20_matches(ds, c, lr, "rolora", 4), False),
            ("RoLoRA E15", f"results/round_metrics/*MAIN-DS-{ds}-C{c}-*-M-rolora-LR{lr}-RK4-S*-E15*.csv", False),
            ("RoLoRA E10", f"results/round_metrics/*MAIN-DS-{ds}-C{c}-R{r}-M-rolora-LR{lr}-RK4-S*-E10*.csv", False),
            ("RoLoRA E5", f"results/round_metrics/*MAIN-DS-{ds}-C{c}-R{r}-M-rolora-LR{lr}-RK4-S*-E5*.csv", False),
        ]
        if include_lora:
            methods.extend([
                ("LoRA E20", find_e20_matches(ds, c, lr, "lora", 4), True),
                ("LoRA E10", f"results/round_metrics/*MAIN-DS-{ds}-C{c}-R{r}-M-lora-LR{lr}-RK4-S*-E10*.csv", True),
                ("LoRA E5", f"results/round_metrics/*MAIN-DS-{ds}-C{c}-R{r}-M-lora-LR{lr}-RK4-S*-E5*.csv", True),
                ("FFA-LoRA E10", f"results/round_metrics/*MAIN-DS-{ds}-C{c}-R{r}-M-ffa-LR{lr}-RK4-S*-E10*.csv", True),
                ("FFA-LoRA E5", f"results/round_metrics/*MAIN-DS-{ds}-C{c}-R{r}-M-ffa-LR{lr}-RK4-S*-E5*.csv", True),
            ])
        return methods

    methods = [
        ("FDA E3TO40 [Emax/1]", find_fda_k1_matches(ds, c, r, lr, 4, "E3TO40"), False),'''

code = code.replace(old_methods_def, new_methods_def, 1)

# Update build_dataset_plots_section
old_plots_sec = '''def build_dataset_plots_section(include_lora=True):
    dir_name = "master_all_methods" if include_lora else "fda_vs_rolora"
    sub_prefix = "1" if include_lora else "2"
    sections = []
    
    for idx, (ds_key, ds_name, ds_desc) in enumerate(dataset_metadata):
        matching_cfgs = [c for c in all_configs if c[2] == ds_key]'''

new_plots_sec = '''def build_dataset_plots_section(include_lora=True, pure_only=False):
    dir_name = "master_all_methods" if include_lora else "fda_vs_rolora"
    sub_prefix = "1" if include_lora else "2"
    sections = []
    
    meta = pure_dataset_metadata if pure_only else dataset_metadata
    cfg_list = pure_configs if pure_only else all_configs
    
    for idx, (ds_key, ds_name, ds_desc) in enumerate(meta):
        matching_cfgs = [c for c in cfg_list if c[2] == ds_key]'''

code = code.replace(old_plots_sec, new_plots_sec, 1)

# Update build_consolidated_summary_table
old_tbl_sec = '''def build_consolidated_summary_table(include_lora=True):
    lines = [
        f"| Dataset / Config | Method | Seeds | Max Acc | Targeted Milestone 1 (Time, Complete MB) | Targeted Milestone 2 (Time, Complete MB) | Targeted Milestone 3 (Time, Complete MB) | Total Rounds | Complete MB | Best Round | Total Time (s / h) | Steps | Mean Ep/Rd | <span style=\\"color: {COLOR_MATRIX_A};\\">Mean Ep (A)/Rd</span> | <span style=\\"color: {COLOR_MATRIX_B};\\">Mean Ep (B)/Rd</span> |",
        "| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |"
    ]
    for cfg_tuple in all_configs:
        cfg_prefix, title, ds, c, r, lr, milestones, m_headers = cfg_tuple
        lines.append(f"| **Dataset / Config** | **Method** | **Seeds** | **Max Acc** | **{m_headers[0]} (Time, Complete MB)** | **{m_headers[1]} (Time, Complete MB)** | **{m_headers[2]} (Time, Complete MB)** | **Total Rounds** | **Complete MB** | **Best Round** | **Total Time (s / h)** | **Steps** | **Mean Ep/Rd** | <span style=\\"color: {COLOR_MATRIX_A};\\">**Mean Ep (A)/Rd**</span> | <span style=\\"color: {COLOR_MATRIX_B};\\">**Mean Ep (B)/Rd**</span> |")
        
        methods = get_methods(ds, c, r, lr, include_lora, cfg_prefix)'''

new_tbl_sec = '''def build_consolidated_summary_table(include_lora=True, pure_only=False):
    cfg_list = pure_configs if pure_only else all_configs
    lines = [
        f"| Dataset / Configuration | Method | Seeds | Max Acc | Targeted Milestone 1 (Time, Complete MB) | Targeted Milestone 2 (Time, Complete MB) | Targeted Milestone 3 (Time, Complete MB) | Total Rounds | Complete MB | Best Round | Total Time (s / h) | Steps | Mean Ep/Rd | <span style=\\"color: {COLOR_MATRIX_A};\\">Mean Ep (A)/Rd</span> | <span style=\\"color: {COLOR_MATRIX_B};\\">Mean Ep (B)/Rd</span> |",
        "| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |"
    ]
    for cfg_tuple in cfg_list:
        cfg_prefix, title, ds, c, r, lr, milestones, m_headers = cfg_tuple
        lines.append(f"| **Dataset / Configuration** | **Method** | **Seeds** | **Max Acc** | **{m_headers[0]} (Time, Complete MB)** | **{m_headers[1]} (Time, Complete MB)** | **{m_headers[2]} (Time, Complete MB)** | **Total Rounds** | **Complete MB** | **Best Round** | **Total Time (s / h)** | **Steps** | **Mean Ep/Rd** | <span style=\\"color: {COLOR_MATRIX_A};\\">**Mean Ep (A)/Rd**</span> | <span style=\\"color: {COLOR_MATRIX_B};\\">**Mean Ep (B)/Rd**</span> |")
        
        methods = get_methods(ds, c, r, lr, include_lora, cfg_prefix, pure_only=pure_only)'''

code = code.replace(old_tbl_sec, new_tbl_sec, 1)

# Update count_used_and_unique_runs
old_count = '''def count_used_and_unique_runs():
    used = set()
    unique_configs = set()
    for cfg in all_configs:
        cfg_prefix, title, ds, c, r, lr, milestones, m_headers = cfg
        methods = get_methods(ds, c, r, lr, include_lora=True, cfg_prefix=cfg_prefix)
        for mname, pat, is_lora in methods:
            matches = pat if isinstance(pat, list) else sorted(glob.glob(pat))
            if matches:
                unique_configs.add((cfg_prefix, mname))
            for m in matches:
                used.add(os.path.abspath(m))
    return len(used), len(unique_configs)'''

new_count = '''def count_used_and_unique_runs(pure_only=False):
    used = set()
    unique_configs = set()
    cfg_list = pure_configs if pure_only else all_configs
    for cfg in cfg_list:
        cfg_prefix, title, ds, c, r, lr, milestones, m_headers = cfg
        methods = get_methods(ds, c, r, lr, include_lora=True, cfg_prefix=cfg_prefix, pure_only=pure_only)
        for mname, pat, is_lora in methods:
            matches = pat if isinstance(pat, list) else sorted(glob.glob(pat))
            if matches:
                unique_configs.add((cfg_prefix, mname))
            for m in matches:
                used.add(os.path.abspath(m))
    return len(used), len(unique_configs)'''

code = code.replace(old_count, new_count, 1)

# Now update update_markdown()
old_update_md = code[code.find('def update_markdown():'):]

new_update_md = '''def update_markdown():
    print("Building consolidated tables and dataset plots...")

    total_runs, unique_runs = count_used_and_unique_runs(pure_only=False)
    pure_runs, pure_unique = count_used_and_unique_runs(pure_only=True)

    intro = f"""# Comprehensive Evaluation: FDA-RoLoRA vs. Baselines (RoLoRA, LoRA, FFA-LoRA)

- **Model**: `roberta-base` (125M parameters)
- **Data Partitioning**: High IID data scenarios (Dirichlet $\\\\alpha = 1000.0$) and Non-IID label skew scenarios (Dirichlet $\\\\alpha = 0.5$) across $C \\\\in \\\\{{20, 50\\\\}}$ federated clients
- **Sample Scaling & Preprocessing**: Datasets exceeding 25,000 training examples are capped at 25,000 samples ($\\\\text{{MAX\\\\_SAMPLES}} = 25{{,}}000$), while smaller datasets are utilized in full (100%).
- **Primary Thesis Benchmark Scope (FDA Pure vs. Baselines)**: **16 benchmark configurations** (8 IID configurations + 8 Non-IID Dirichlet $\\\\alpha=0.5$ configurations) across 4 primary GLUE tasks (MNLI, QNLI, QQP, SST-2).
- **Runs in Pure Benchmark Suite**: **{pure_runs} runs** ({pure_unique} unique configurations across seeds)
- **Total Project Runs (including exploratory tasks & legacy variants)**: **{total_runs} runs** ({unique_runs} unique configurations across 24 parameter groups)

| Tier | Dataset | Full Name | Task Type | Classes | Total Scale | Samples Used (Max 25k) | % Used | Description |
| :---: | :--- | :--- | :--- | :--- | :---: | :---: | :---: | :--- |
| **Tier 1** | **MNLI** | Multi-Genre Natural Language Inference | Sentence-pair Entailment | 3 | 392,702 pairs | 25,000 pairs | 6.37% | Predicts logical relationship between premise and hypothesis sentence pairs (`entailment`, `neutral`, `contradiction`) across diverse genres. |
|  | **QNLI** | Question Natural Language Inference | QA / Binary Entailment | 2 | 104,743 pairs | 25,000 pairs | 23.87% | Derived from SQuAD; evaluates whether a context sentence contains the answer to a question (`entailment` vs. `not_entailment`). |
| **Tier 2** | **QQP** | Quora Question Pairs | Semantic Paraphrase | 2 | 363,846 pairs | 25,000 pairs | 6.87% | Binary classification determining whether two community questions from Quora are semantically equivalent duplicates (`duplicate` vs. `not_duplicate`). |
|  | **SST-2** | Stanford Sentiment Treebank | Single-Sentence Sentiment | 2 | 67,349 sentences | 25,000 sentences | 37.12% | Binary sentiment classification (`positive` vs. `negative`) of single sentence snippets from movie reviews. |
| **Tier 3** | **MRPC** | Microsoft Research Paraphrase Corpus | Sentence-pair Paraphrase | 2 | 3,668 pairs | 3,668 pairs | 100.0% | Evaluates whether two sentences extracted from online news sources are semantically equivalent paraphrases (`equivalent` vs. `not_equivalent`). |
|  | **RTE** | Recognizing Textual Entailment | Binary Textual Entailment | 2 | 2,490 pairs | 2,490 pairs | 100.0% | Combines series of annual entailment challenges (news/Wikipedia) determining if a premise entails a hypothesis (`entailment` vs. `not_entailment`). |

---

## 1. Primary Benchmark Suite: FDA Pure vs. Baselines (RoLoRA, LoRA, FFA-LoRA)

Direct head-to-head comparison isolating the principled, unconstrained closed-loop dual-track controller (**FDA Pure**: $k=1.0$ and $k=0.5$ variants, $e \\\\in [1, E_{{\\\\max}}]$, NODISC $\\\\gamma=1.0$, Parity B-Start, NOGUARD) against all fixed-epoch baselines (**RoLoRA E20/E15/E10/E5**, **LoRA E20/E10/E5**, **FFA-LoRA E10/E5**) across all 16 benchmark scenarios where they were evaluated (8 IID scenarios + 8 Non-IID Dirichlet $\\\\alpha=0.5$ label skew scenarios), averaged across available random seeds (Seeds 42, 43, 44 as indicated in the Seeds column), including milestone convergence speeds and complete bi-directional communication cost (Complete MB, Float32 payload):

> [!NOTE]
> **Focused Primary Presentation**: This primary benchmark suite exclusively features **FDA Pure** alongside the fixed-epoch baselines. Earlier developmental FDA iterations (calibrated BDISC3 with empirical divisor $\\\\gamma=3.0$, lower floor clamp $e_{{\\\\min}}=3$, and A-start) have been removed from the primary presentation to reflect the definitive, mathematically pure formulation of the thesis. Complete archival records of all 24 configurations and 420 historical runs are preserved in Section 5.
"""

    table1 = build_consolidated_summary_table(include_lora=True, pure_only=True)

    s1_plots_header = """### Primary Telemetry & Dynamics Plots by Dataset & Parameter Group

For each benchmark dataset and parameter group in the primary suite, the complete 6-panel telemetry figure and companion Train vs. Validation Accuracy Gap analysis are shown directly below:
1. **Validation Accuracy vs. Round** (highlighting peak accuracy milestone markers `*`).
2. **Validation Accuracy vs. Cumulative Communication Cost (MB)** (payload-aware bi-directional Float32 tracking).
3. **Allocated Local Epochs ($E_t$)** (discrete alternating bins for adaptive FDA runs: Darker shade = Matrix A / odd rounds, Lighter shade = Matrix B / even rounds; dashed lines for fixed baselines).
4. **FDA Closed-Loop Dynamic Thresholds ($\\\\theta_A, \\\\theta_B$)** (tracking exploration threshold adaptation across active FDA Pure runs).
5. **Inter-Client Drift Variance ($\\\\sigma^2_t$) vs. Round** (decomposed into Matrix A and Matrix B for alternating methods; joint variance for standard LoRA and FFA-LoRA).
6. **Client Drift Variance vs. Cumulative Local Epochs ($\\\\sum E_t$)** (illustrating variance accumulation per unit of client compute budget).
7. **Train vs. Validation Accuracy Gap (Dual-Curve Telemetry)** (Direct comparison per method between on-line local client training accuracy [dashed] and global post-aggregation validation accuracy [solid], highlighting generalization dynamics and dropout / model-averaging effects).
8. **Inter-Client Drift Variance [Log-Scale] vs. Round** (Logarithmic progression matching the FDA-Opt literature scale $10^{-8}$ to $10^{-1}$ vs. communication rounds, revealing fine-grained drift orders of magnitude across training).
"""
    s1_plots = s1_plots_header.strip() + "\\n\\n" + build_dataset_plots_section(include_lora=True, pure_only=True)

    s2_intro = """## 2. Direct Architectural Telemetry: FDA Pure vs. RoLoRA Baselines (E20, E15, E10, E5)

Focused head-to-head architectural comparison isolating the random orthonormal projection constraint architecture between **FDA Pure Dual-Track** ($k=1.0$ and $k=0.5$) and standard fixed-epoch **RoLoRA (E20, E15, E10, E5)** across all 16 evaluated scenarios, averaged across available random seeds.
"""

    table2 = build_consolidated_summary_table(include_lora=False, pure_only=True)

    s2_plots_header = """### Architectural Telemetry & Dynamics Plots (FDA Pure vs. RoLoRA)

For each benchmark dataset and parameter group, the architectural head-to-head 6-panel telemetry figure and Log-Scale Client Drift Variance isolating FDA Pure against RoLoRA baselines are shown directly below:
"""
    s2_plots = s2_plots_header.strip() + "\\n\\n" + build_dataset_plots_section(include_lora=False, pure_only=True)

    s3_intro = """## 3. Targeted Strategic Convergence Telemetry (Consolidated Pure Scorecard)

Consolidated scorecard summarizing milestone speed, communication payload, and computation efficiency for FDA Pure against all baselines across all 16 GLUE benchmark configurations:
"""
    table3 = table1

    section4_text = ""
    if os.path.exists("FDA_vs_Baselines_v2.md"):
        with open("FDA_vs_Baselines_v2.md", "r", encoding="utf-8") as f:
            content = f.read()
            idx4 = content.find("## 4. Analytical Investigation:")
            idx5 = content.find("## 5. Complete Archival Benchmark Suite")
            if idx4 != -1:
                if idx5 != -1:
                    section4_text = "\\n---\\n\\n" + content[idx4:idx5].strip() + "\\n"
                else:
                    section4_text = "\\n---\\n\\n" + content[idx4:].strip() + "\\n"

    s5_intro = """\\n---\\n\\n## 5. Complete Archival Benchmark Suite (All 24 Historical Configurations & Exploratory Iterations)

For archival completeness, reproducibility, and research transparency, the full evaluation matrix across all 24 parameter configurations (including exploratory low-resource tasks MRPC and RTE, exploratory low learning rates $\\\\text{lr}=0.01$, and intermediate developmental FDA variants) is documented below:

"""
    archival_table = build_consolidated_summary_table(include_lora=True, pure_only=False)

    full_doc = (
        intro.strip() + "\\n\\n"
        + table1.strip() + "\\n\\n---\\n\\n"
        + s1_plots.strip() + "\\n\\n---\\n\\n"
        + s2_intro.strip() + "\\n\\n"
        + table2.strip() + "\\n\\n---\\n\\n"
        + s2_plots.strip() + "\\n\\n---\\n\\n"
        + s3_intro.strip() + "\\n\\n"
        + table3.strip() + "\\n"
        + section4_text
        + s5_intro
        + archival_table.strip() + "\\n"
    )

    with open("FDA_vs_Baselines_v2.md", "w", encoding="utf-8") as f:
        f.write(full_doc)
    print("FDA_vs_Baselines_v2.md successfully updated!")

if __name__ == "__main__":
    update_markdown()
'''

code = code[:code.find('def update_markdown():')] + new_update_md

with open("update_benchmark_suite.py", "w", encoding="utf-8") as f:
    f.write(code)

print("Updated update_benchmark_suite.py successfully!")
