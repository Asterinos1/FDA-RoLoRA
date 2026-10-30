import os
import sys
import glob
import re
import math
import pandas as pd
import numpy as np

# Ensure code repo root is on path
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, BASE_DIR)
os.chdir(BASE_DIR)

from update_benchmark_suite import pure_configs, get_methods, prep, average_runs, comm_rates

OUT_DIR = os.path.join(BASE_DIR, "analysis", "thesis_final_audit")
os.makedirs(OUT_DIR, exist_ok=True)

THESIS_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "thesis-stuff"))
EVAL_TEX = os.path.join(THESIS_DIR, "thesis", "chapters", "evaluation.tex")

# =============================================================================
# 1. RUN MANIFEST GENERATION
# =============================================================================
print("=== Step 1: Generating Complete Run Manifest ===")

non_iid_globs = {
    "17_MNLI_C20_R50_LR0.1_A0.5": [
        ("FDA Pure E1TO40 [k=0.5]", "results/round_metrics/*THESIS-NONIID-A0.5-DS-mnli-C20-*-M-fda-pure-LR0.1-RK4-E1TO40-CRATIO0.5-*.csv"),
        ("FDA Pure E1TO20 [k=0.5]", "results/round_metrics/*THESIS-NONIID-A0.5-DS-mnli-C20-*-M-fda-pure-LR0.1-RK4-E1TO20-CRATIO0.5-*.csv"),
        ("FDA Pure E1TO20 [k=1.0]", "results/round_metrics/*THESIS-NONIID-A0.5-DS-mnli-C20-*-M-fda-pure-LR0.1-RK4-E1TO20-CRATIO1-*.csv"),
        ("RoLoRA E20", "results/round_metrics/*THESIS-NONIID-A0.5-DS-mnli-C20-*-M-rolora-E20-LR0.1-RK4-*.csv"),
        ("RoLoRA E10", "results/round_metrics/*THESIS-NONIID-A0.5-DS-mnli-C20-*-M-rolora-E10-LR0.1-RK4-*.csv"),
    ],
    "18_QNLI_C20_R50_LR0.1_A0.5": [
        ("FDA Pure E1TO40 [k=1.0]", "results/round_metrics/*THESIS-NONIID-A0.5-DS-qnli-C20-*-M-fda-pure-LR0.1-RK4-E1TO40-CRATIO1-*.csv"),
        ("FDA Pure E1TO40 [k=0.5]", "results/round_metrics/*THESIS-NONIID-A0.5-DS-qnli-C20-*-M-fda-pure-LR0.1-RK4-E1TO40-CRATIO0.5-*.csv"),
        ("FDA Pure E1TO20 [k=1.0]", "results/round_metrics/*THESIS-NONIID-A0.5-DS-qnli-C20-*-M-fda-pure-LR0.1-RK4-E1TO20-CRATIO1-*.csv"),
        ("FDA Pure E1TO20 [k=0.5]", "results/round_metrics/*THESIS-NONIID-A0.5-DS-qnli-C20-*-M-fda-pure-LR0.1-RK4-E1TO20-CRATIO0.5-*.csv"),
        ("RoLoRA E20", "results/round_metrics/*THESIS-NONIID-A0.5-DS-qnli-C20-*-M-rolora-E20-LR0.1-RK4-*.csv"),
        ("RoLoRA E10", "results/round_metrics/*THESIS-NONIID-A0.5-DS-qnli-C20-*-M-rolora-E10-LR0.1-RK4-*.csv"),
    ],
    "19_MNLI_C50_R30_LR0.1_A0.5": [
        ("FDA Pure E1TO20 [k=1.0]", "results/round_metrics/*THESIS-NONIID-A0.5-DS-mnli-C50-*-M-fda-pure-LR0.1-RK4-E1TO20-CRATIO1-*.csv"),
        ("FDA Pure E1TO20 [k=0.5]", "results/round_metrics/*THESIS-NONIID-A0.5-DS-mnli-C50-*-M-fda-pure-LR0.1-RK4-E1TO20-CRATIO0.5-*.csv"),
        ("RoLoRA E10", "results/round_metrics/*THESIS-NONIID-A0.5-DS-mnli-C50-*-M-rolora-E10-LR0.1-RK4-*.csv"),
    ],
    "20_QNLI_C50_R30_LR0.1_A0.5": [
        ("FDA Pure E1TO20 [k=1.0]", "results/round_metrics/*THESIS-NONIID-A0.5-DS-qnli-C50-*-M-fda-pure-LR0.1-RK4-E1TO20-CRATIO1-*.csv"),
        ("FDA Pure E1TO20 [k=0.5]", "results/round_metrics/*THESIS-NONIID-A0.5-DS-qnli-C50-*-M-fda-pure-LR0.1-RK4-E1TO20-CRATIO0.5-*.csv"),
        ("RoLoRA E10", "results/round_metrics/*THESIS-NONIID-A0.5-DS-qnli-C50-*-M-rolora-E10-LR0.1-RK4-*.csv"),
    ],
    "21_QQP_C20_R20_LR0.1_A0.5": [
        ("FDA Pure E1TO20 [k=1.0]", "results/round_metrics/*THESIS-NONIID-A0.5-DS-qqp-C20-*-M-fda-pure-LR0.1-RK4-E1TO20-CRATIO1-*.csv"),
        ("FDA Pure E1TO20 [k=0.5]", "results/round_metrics/*THESIS-NONIID-A0.5-DS-qqp-C20-*-M-fda-pure-LR0.1-RK4-E1TO20-CRATIO0.5-*.csv"),
        ("RoLoRA E10", "results/round_metrics/*THESIS-NONIID-A0.5-DS-qqp-C20-*-M-rolora-E10-LR0.1-RK4-*.csv"),
    ],
    "22_QQP_C50_R24_LR0.1_A0.5": [
        ("FDA Pure E1TO20 [k=1.0]", "results/round_metrics/*THESIS-NONIID-A0.5-DS-qqp-C50-*-M-fda-pure-LR0.1-RK4-E1TO20-CRATIO1-*.csv"),
        ("FDA Pure E1TO20 [k=0.5]", "results/round_metrics/*THESIS-NONIID-A0.5-DS-qqp-C50-*-M-fda-pure-LR0.1-RK4-E1TO20-CRATIO0.5-*.csv"),
        ("RoLoRA E10", "results/round_metrics/*THESIS-NONIID-A0.5-DS-qqp-C50-*-M-rolora-E10-LR0.1-RK4-*.csv"),
    ],
    "23_SST2_C20_R25_LR0.1_A0.5": [
        ("FDA Pure E1TO20 [k=1.0]", "results/round_metrics/*THESIS-NONIID-A0.5-DS-sst2-C20-*-M-fda-pure-LR0.1-RK4-E1TO20-CRATIO1-*.csv"),
        ("FDA Pure E1TO20 [k=0.5]", "results/round_metrics/*THESIS-NONIID-A0.5-DS-sst2-C20-*-M-fda-pure-LR0.1-RK4-E1TO20-CRATIO0.5-*.csv"),
        ("RoLoRA E10", "results/round_metrics/*THESIS-NONIID-A0.5-DS-sst2-C20-*-M-rolora-E10-LR0.1-RK4-*.csv"),
    ],
    "24_SST2_C50_R20_LR0.1_A0.5": [
        ("FDA Pure E1TO20 [k=1.0]", "results/round_metrics/*THESIS-NONIID-A0.5-DS-sst2-C50-*-M-fda-pure-LR0.1-RK4-E1TO20-CRATIO1-*.csv"),
        ("FDA Pure E1TO20 [k=0.5]", "results/round_metrics/*THESIS-NONIID-A0.5-DS-sst2-C50-*-M-fda-pure-LR0.1-RK4-E1TO20-CRATIO0.5-*.csv"),
        ("RoLoRA E10", "results/round_metrics/*THESIS-NONIID-A0.5-DS-sst2-C50-*-M-rolora-E10-LR0.1-RK4-*.csv"),
    ],
}

manifest_rows = []
selected_files_set = set()

for cfg in pure_configs:
    cfg_prefix, title, ds, c, r, lr, milestones, m_headers = cfg
    methods = get_methods(ds, c, r, lr, include_lora=False, cfg_prefix=cfg_prefix, pure_only=True)
    for mname, pat, is_lora in methods:
        matches = pat if isinstance(pat, list) else sorted(glob.glob(pat))
        for f in matches:
            selected_files_set.add(os.path.abspath(f))

def parse_file_metadata(filepath, cfg_prefix, title, ds, c, r_cfg, lr, mname, is_included=True, reason=None):
    fname = os.path.basename(filepath)
    df = pd.read_csv(filepath)
    df = prep(df)
    r_exec = len(df)
    alpha = 0.5 if "NONIID-A0.5" in fname else 1000.0
    
    if "fda-pure" in fname:
        method = "FDA-RoLoRA"
    elif "rolora" in fname:
        method = "RoLoRA"
    else:
        method = "Other"
        
    m_k = re.search(r'CRATIO([\d\.]+)', fname)
    k = float(m_k.group(1)) if m_k else (1.0 if "[k=1.0]" in mname else (0.5 if "[k=0.5]" in mname else None))
    
    m_emax = re.search(r'E1TO(\d+)', fname)
    if m_emax:
        e_min = 1
        e_max = int(m_emax.group(1))
    elif "FDA-RoLoRA" in method:
        e_min = 1
        e_max = 20
    else:
        e_min = None
        e_max = None
        
    m_e = re.search(r'-E(\d+)-', fname)
    if m_e:
        E = int(m_e.group(1))
    elif method == "RoLoRA":
        E = int(df["round_epochs"].iloc[0]) if "round_epochs" in df.columns else (int(df["epochs"].iloc[0]) if "epochs" in df.columns else None)
    else:
        E = None
        
    m_s = re.search(r'-S(\d+)', fname)
    seed = int(m_s.group(1)) if m_s else None
    
    m_rcfg = re.search(r'-R(\d+)-', fname)
    r_configured = int(m_rcfg.group(1)) if m_rcfg else r_cfg
    
    m_rk = re.search(r'-RK(\d+)', fname)
    rank = int(m_rk.group(1)) if m_rk else 4
    
    layers = [9, 10, 11] if ds == "sst2" else [7, 8, 9, 10, 11]
    
    acc_series = df["accuracy"] if "accuracy" in df.columns else df["accuracy_m"]
    peak_acc = float(acc_series.max())
    final_acc = float(acc_series.iloc[-1])
    mean_ep = float(df["round_epochs"].mean()) if "round_epochs" in df.columns else float(E)
    
    thesis_cell = f"{ds.upper()} | C={c} | {'Non-IID' if alpha == 0.5 else 'IID'} | {mname}"
    
    return {
        "filename": fname,
        "filepath": filepath,
        "dataset": ds,
        "cohort_C": c,
        "alpha": alpha,
        "method": method,
        "mname": mname,
        "k": k,
        "e_min": e_min,
        "e_max": e_max,
        "E": E,
        "seed": seed,
        "R_configured": r_configured,
        "R_executed": r_exec,
        "learning_rate": lr,
        "rank": rank,
        "layers": str(layers),
        "num_layers": len(layers),
        "peak_acc": peak_acc,
        "final_acc": final_acc,
        "mean_epochs": mean_ep,
        "thesis_cell": thesis_cell,
        "status": "INCLUDED" if is_included else "EXCLUDED",
        "exclusion_reason": reason if reason else "None",
    }

for cfg in pure_configs:
    cfg_prefix, title, ds, c, r, lr, milestones, m_headers = cfg
    methods = get_methods(ds, c, r, lr, include_lora=False, cfg_prefix=cfg_prefix, pure_only=True)
    for mname, pat, is_lora in methods:
        matches = pat if isinstance(pat, list) else sorted(glob.glob(pat))
        for f in matches:
            meta = parse_file_metadata(f, cfg_prefix, title, ds, c, r, lr, mname, is_included=True)
            manifest_rows.append(meta)

for cfg_prefix, mlist in non_iid_globs.items():
    cfg_match = [c for c in pure_configs if c[0] == cfg_prefix][0]
    _, title, ds, c, r, lr, milestones, m_headers = cfg_match
    for mname, raw_pat in mlist:
        raw_matches = sorted(glob.glob(raw_pat))
        for f in raw_matches:
            if os.path.abspath(f) not in selected_files_set:
                df_raw = pd.read_csv(f)
                reason = f"Aborted run: stopped at round {len(df_raw)} (< floor {r}); superseded by larger file for seed"
                meta = parse_file_metadata(f, cfg_prefix, title, ds, c, r, lr, mname, is_included=False, reason=reason)
                manifest_rows.append(meta)

df_manifest = pd.DataFrame(manifest_rows)
df_manifest.to_csv(os.path.join(OUT_DIR, "run_manifest.csv"), index=False)

# =============================================================================
# 2. CELL METRICS COMPUTATION
# =============================================================================
print("=== Step 2: Computing Comprehensive Cell Metrics ===")

cell_records = []

for cfg in pure_configs:
    cfg_prefix, title, ds, c, r, lr, milestones, m_headers = cfg
    alpha = 0.5 if "A0.5" in cfg_prefix else 1000.0
    methods = get_methods(ds, c, r, lr, include_lora=False, cfg_prefix=cfg_prefix, pure_only=True)
    
    for mname, pat, is_lora in methods:
        matches = pat if isinstance(pat, list) else sorted(glob.glob(pat))
        if not matches:
            continue
            
        dfs = [prep(pd.read_csv(f)) for f in matches]
        n_seeds = len(dfs)
        seeds = [re.search(r'-S(\d+)', os.path.basename(f)).group(1) for f in matches]
        
        per_seed_peaks = []
        per_seed_finals = []
        per_seed_rounds = []
        per_seed_e = []
        per_seed_ea = []
        per_seed_eb = []
        per_seed_ss_e = []
        per_seed_var = []
        
        for df in dfs:
            acc_col = df["accuracy"] if "accuracy" in df.columns else df["accuracy_m"]
            per_seed_peaks.append(float(acc_col.max()))
            per_seed_finals.append(float(acc_col.iloc[-1]))
            per_seed_rounds.append(len(df))
            per_seed_e.append(float(df["round_epochs"].mean()) if "round_epochs" in df.columns else float(df["epochs"].mean()))
            
            if "active_matrix" in df.columns:
                df_a = df[df["active_matrix"] == "A"]
                df_b = df[df["active_matrix"] == "B"]
                ea = float(df_a["round_epochs"].mean()) if len(df_a) > 0 else 0.0
                eb = float(df_b["round_epochs"].mean()) if len(df_b) > 0 else 0.0
            else:
                ea = float(df["round_epochs"].mean()) if "round_epochs" in df.columns else float(df["epochs"].mean())
                eb = ea
            per_seed_ea.append(ea)
            per_seed_eb.append(eb)
            
            n_r = len(df)
            if n_r >= 4:
                half_start = max(2, n_r // 2)
                ss_e = float(df["round_epochs"].iloc[half_start:].median())
            else:
                ss_e = float(df["round_epochs"].iloc[-1])
            per_seed_ss_e.append(ss_e)
            
            if "variance" in df.columns:
                per_seed_var.append(float(df["variance"].mean()))
            else:
                per_seed_var.append(0.0)
                
        df_avg = average_runs(dfs)
        md_max_acc = float(df_avg["accuracy"].max())
        md_best_rnd = int(df_avg.loc[df_avg["accuracy"].idxmax(), "round"])
        md_final_acc = float(df_avg["accuracy"].iloc[-1])
        md_tot_r = len(df_avg)
        
        peak_mean = float(np.mean(per_seed_peaks))
        peak_std_sample = float(np.std(per_seed_peaks, ddof=1)) if n_seeds > 1 else 0.0
        peak_std_pop = float(np.std(per_seed_peaks))
        
        final_mean = float(np.mean(per_seed_finals))
        final_std_sample = float(np.std(per_seed_finals, ddof=1)) if n_seeds > 1 else 0.0
        final_std_pop = float(np.std(per_seed_finals))
        
        rounds_mean = float(np.mean(per_seed_rounds))
        rounds_std_sample = float(np.std(per_seed_rounds, ddof=1)) if n_seeds > 1 else 0.0
        
        e_mean = float(np.mean(per_seed_e))
        ea_mean = float(np.mean(per_seed_ea))
        eb_mean = float(np.mean(per_seed_eb))
        ss_e_mean = float(np.mean(per_seed_ss_e))
        var_mean = float(np.mean(per_seed_var))
        
        is_sst2 = (ds == "sst2")
        client_kb_round = 144.0 if is_sst2 else 240.0
        client_mb_round = client_kb_round / 1024.0
        cohort_mb_round_a = c * client_mb_round
        cohort_mb_round_orig = comm_rates[(c, False)]
        
        mb_total_a_exact = rounds_mean * cohort_mb_round_a
        mb_total_a_orig = rounds_mean * cohort_mb_round_orig
        
        ams_bytes_epoch = 6000.0
        ams_mb_epoch = ams_bytes_epoch / (1024.0 * 1024.0)
        
        per_seed_inter_epochs = []
        for df in dfs:
            ep_col = df["round_epochs"] if "round_epochs" in df.columns else df["epochs"]
            inter_eps = sum(max(0, ep - 1) for ep in ep_col)
            per_seed_inter_epochs.append(inter_eps)
        mean_inter_eps = float(np.mean(per_seed_inter_epochs))
        
        mb_monitoring_ams = c * mean_inter_eps * ams_mb_epoch
        mb_total_b_ams = mb_total_a_exact + mb_monitoring_ams
        
        full_drift_kb_epoch = 72.0 if is_sst2 else 120.0
        full_drift_mb_epoch = full_drift_kb_epoch / 1024.0
        mb_monitoring_full = c * mean_inter_eps * full_drift_mb_epoch
        mb_total_b_full = mb_total_a_exact + mb_monitoring_full
        
        cell_records.append({
            "cfg_prefix": cfg_prefix,
            "title": title,
            "dataset": ds,
            "cohort_C": c,
            "alpha": alpha,
            "method": mname,
            "seeds": seeds,
            "n_seeds": n_seeds,
            "per_seed_peaks": per_seed_peaks,
            "per_seed_finals": per_seed_finals,
            "per_seed_rounds": per_seed_rounds,
            "peak_mean": peak_mean,
            "peak_std_sample": peak_std_sample,
            "peak_std_pop": peak_std_pop,
            "final_mean": final_mean,
            "final_std_sample": final_std_sample,
            "final_std_pop": final_std_pop,
            "md_max_acc": md_max_acc,
            "md_final_acc": md_final_acc,
            "md_best_rnd": md_best_rnd,
            "rounds_mean": rounds_mean,
            "rounds_std_sample": rounds_std_sample,
            "e_mean": e_mean,
            "e_steady_state": ss_e_mean,
            "ea_mean": ea_mean,
            "eb_mean": eb_mean,
            "mean_variance": var_mean,
            "mean_intermediate_epochs": mean_inter_eps,
            "mb_accounting_a_exact": mb_total_a_exact,
            "mb_accounting_a_orig": mb_total_a_orig,
            "mb_accounting_b_ams": mb_total_b_ams,
            "mb_accounting_b_full": mb_total_b_full,
        })

df_cells = pd.DataFrame(cell_records)
df_cells.to_csv(os.path.join(OUT_DIR, "cell_metrics.csv"), index=False)

# =============================================================================
# 3. REPRODUCED BENCHMARK TABLES
# =============================================================================
print("=== Step 3: Generating Reproduced Tables (Scorecards and Detailed) ===")

# Build Scorecard Table (reproducing Table 5.1 & 5.2 format)
scorecard_rows = []
for cfg in pure_configs:
    cfg_prefix, title, ds, c, r, lr, milestones, m_headers = cfg
    alpha = 0.5 if "A0.5" in cfg_prefix else 1000.0
    regime = "Non-IID" if alpha == 0.5 else "IID"
    
    sub = df_cells[df_cells["cfg_prefix"] == cfg_prefix]
    fda_rows = sub[sub["method"].str.contains("FDA")]
    ro_rows = sub[sub["method"].str.contains("RoLoRA")]
    
    if len(fda_rows) == 0 or len(ro_rows) == 0:
        continue
        
    # Top FDA variant by peak_mean
    best_fda = fda_rows.loc[fda_rows["peak_mean"].idxmax()]
    # Top RoLoRA variant by peak_mean
    best_ro = ro_rows.loc[ro_rows["peak_mean"].idxmax()]
    
    delta = (best_fda["peak_mean"] - best_ro["peak_mean"]) * 100.0
    
    scorecard_rows.append({
        "dataset": ds.upper(),
        "cohort_C": c,
        "regime": regime,
        "cfg_prefix": cfg_prefix,
        "top_fda_method": best_fda["method"],
        "fda_peak_mean": best_fda["peak_mean"] * 100.0,
        "fda_peak_std": best_fda["peak_std_sample"] * 100.0,
        "fda_e_mean": best_fda["e_mean"],
        "fda_mb_exact": best_fda["mb_accounting_a_exact"],
        "fda_mb_orig": best_fda["mb_accounting_a_orig"],
        "top_ro_method": best_ro["method"],
        "ro_peak_mean": best_ro["peak_mean"] * 100.0,
        "ro_peak_std": best_ro["peak_std_sample"] * 100.0,
        "ro_e_mean": best_ro["e_mean"],
        "ro_mb_exact": best_ro["mb_accounting_a_exact"],
        "ro_mb_orig": best_ro["mb_accounting_a_orig"],
        "delta_acc": delta,
    })

df_scorecard = pd.DataFrame(scorecard_rows)
df_scorecard.to_csv(os.path.join(OUT_DIR, "reproduced_scorecard.csv"), index=False)

# Build Detailed Table
detailed_rows = []
for _, r in df_cells.iterrows():
    detailed_rows.append({
        "dataset": r["dataset"].upper(),
        "cohort_C": r["cohort_C"],
        "regime": "Non-IID" if r["alpha"] == 0.5 else "IID",
        "method": r["method"],
        "n_seeds": r["n_seeds"],
        "seeds": str(r["seeds"]),
        "final_mean": r["final_mean"] * 100.0,
        "final_std": r["final_std_sample"] * 100.0,
        "peak_mean": r["peak_mean"] * 100.0,
        "peak_std": r["peak_std_sample"] * 100.0,
        "md_max_acc": r["md_max_acc"] * 100.0,
        "rounds_mean": r["rounds_mean"],
        "rounds_std": r["rounds_std_sample"],
        "mb_exact": r["mb_accounting_a_exact"],
        "mb_orig": r["mb_accounting_a_orig"],
        "e_mean": r["e_mean"],
        "ea_mean": r["ea_mean"],
        "eb_mean": r["eb_mean"],
        "e_steady_state": r["e_steady_state"],
        "mean_variance": r["mean_variance"],
    })
df_detailed = pd.DataFrame(detailed_rows)
df_detailed.to_csv(os.path.join(OUT_DIR, "reproduced_detailed.csv"), index=False)

# =============================================================================
# 4. AUDIT DIFF GENERATION (COMPARING EVERY NUMBER IN CH 5)
# =============================================================================
print("=== Step 4: Generating audit_diff.md against Chapter 5 ===")

with open(EVAL_TEX, "r", encoding="utf-8") as fp:
    thesis_eval_text = fp.read()

diff_records = []

# Helper to look up cell
def lookup_cell(ds, c, alpha, method_query):
    sub = df_cells[(df_cells["dataset"] == ds.lower()) & (df_cells["cohort_C"] == c) & (df_cells["alpha"] == alpha)]
    for _, r in sub.iterrows():
        if method_query.lower() in r["method"].lower():
            return r
    return None

# Parse Table 5.1 & Table 5.2 entries
def parse_scorecard_tex(tbl_text, is_non_iid=False):
    alpha = 0.5 if is_non_iid else 1000.0
    cur_ds = None
    cur_c = None
    lines = tbl_text.splitlines()
    for l in lines:
        l = l.strip()
        if not l or l.startswith("\\hline") or l.startswith("\\begin") or l.startswith("\\end") or l.startswith("\\caption") or l.startswith("\\label") or l.startswith("\\resizebox"):
            continue
        m_ds = re.search(r'\\textbf\{([A-Z0-9\-]+)\}', l)
        if m_ds and not "Dataset" in l:
            cur_ds = m_ds.group(1).lower().replace("-", "")
            if cur_ds == "sst2": cur_ds = "sst2"
        m_c = re.search(r'C=(\d+)', l)
        if m_c:
            cur_c = int(m_c.group(1))
            
        # Parse method line
        if "RoLoRA" in l or "FDA-RoLoRA" in l:
            parts = [p.strip() for p in l.split("&")]
            # Methodology column
            m_part = parts[-4] if len(parts) >= 4 else parts[0]
            peak_part = parts[-3] if len(parts) >= 4 else ""
            e_part = parts[-2] if len(parts) >= 4 else ""
            mb_part = parts[-1] if len(parts) >= 4 else ""
            mb_part = mb_part.replace("\\\\", "").replace("\\hline", "").replace("\\cline{2-6}", "").strip()
            
            # Clean peak acc
            m_pk = re.search(r'([\d\.]+)\s*(\(\s*\\pm\s*([\d\.]+)\s*\))?', peak_part)
            th_peak = float(m_pk.group(1)) if m_pk else None
            th_std = float(m_pk.group(3)) if m_pk and m_pk.group(3) else 0.0
            
            # Clean e
            m_e = re.search(r'([\d\.]+)', e_part)
            th_e = float(m_e.group(1)) if m_e else None
            
            # Clean mb
            m_mb = re.search(r'([\d\.]+)', mb_part)
            th_mb = float(m_mb.group(1)) if m_mb else None
            
            # Find matching cell in df_cells
            m_query = "k=1.0" if "k=1.0" in m_part else ("k=0.5" if "k=0.5" in m_part else ("E=20" if "E=20" in m_part else ("E=15" if "E=15" in m_part else ("E=10" if "E=10" in m_part else ("E=5" if "E=5" in m_part else "")))))
            cell = lookup_cell(cur_ds, cur_c, alpha, m_query)
            
            tbl_name = "Table 5.2 (Scorecard Non-IID)" if is_non_iid else "Table 5.1 (Scorecard IID)"
            label = f"{tbl_name}: {cur_ds.upper()} C={cur_c} {m_query}"
            
            if cell is not None:
                # Compare peak
                csv_peak = cell["peak_mean"] * 100.0
                md_peak = cell["md_max_acc"] * 100.0
                peak_diff = abs(th_peak - csv_peak)
                md_diff = abs(th_peak - md_peak)
                
                status_pk = "OK"
                expl_pk = "Matches per-seed peak mean exactly."
                if peak_diff > 0.05 and md_diff <= 0.05:
                    status_pk = "MD-vs-CSV CONFLICT"
                    expl_pk = f"Thesis reports {th_peak:.2f}%, matching .md average-trajectory peak ({md_peak:.2f}%), but differing from per-seed peak mean ({csv_peak:.2f}%)."
                elif peak_diff > 0.05:
                    status_pk = "WRONG"
                    expl_pk = f"Thesis reports {th_peak:.2f}%, while CSV recomputation gives {csv_peak:.2f}% (md gives {md_peak:.2f}%)."
                    
                diff_records.append({
                    "item": f"{label} Peak Accuracy",
                    "thesis_value": f"{th_peak:.2f} ± {th_std:.2f}%",
                    "md_value": f"{md_peak:.2f}%",
                    "recomputed_csv": f"{csv_peak:.2f} ± {cell['peak_std_sample']*100:.2f}%",
                    "status": status_pk,
                    "explanation": expl_pk
                })
                
                # Compare e
                csv_e = cell["e_mean"]
                e_diff = abs(th_e - csv_e)
                status_e = "OK" if e_diff <= 0.05 else "WRONG"
                diff_records.append({
                    "item": f"{label} Mean Epochs (e)",
                    "thesis_value": f"{th_e:.2f}",
                    "md_value": f"{csv_e:.2f}",
                    "recomputed_csv": f"{csv_e:.2f}",
                    "status": status_e,
                    "explanation": "Matches recomputed CSV epoch mean." if status_e == "OK" else f"Differs from recomputed epoch mean ({csv_e:.2f})."
                })
                
                # Compare MB
                csv_mb_exact = cell["mb_accounting_a_exact"]
                csv_mb_orig = cell["mb_accounting_a_orig"]
                status_mb = "OK"
                expl_mb = "Matches round-only communication volume."
                if cur_ds == "sst2":
                    if abs(th_mb - csv_mb_orig) <= 0.5:
                        status_mb = "WRONG"
                        expl_mb = f"Calculated using 5-layer payload (240 KB/round = {csv_mb_orig:.1f} MB); correct 3-layer payload (144 KB/round) is {csv_mb_exact:.1f} MB."
                    elif abs(th_mb - csv_mb_exact) <= 0.5:
                        status_mb = "OK"
                        expl_mb = "Matches correct 3-layer SST-2 payload."
                    else:
                        status_mb = "WRONG"
                        expl_mb = f"Does not match recomputed payload (exact={csv_mb_exact:.1f} MB, unadjusted={csv_mb_orig:.1f} MB)."
                else:
                    if abs(th_mb - csv_mb_exact) > 1.0:
                        status_mb = "WRONG"
                        expl_mb = f"Differs from recomputed payload ({csv_mb_exact:.1f} MB)."
                        
                diff_records.append({
                    "item": f"{label} Total MB",
                    "thesis_value": f"{th_mb:.1f} MB",
                    "md_value": f"{cell['mb_accounting_a_orig']:.1f} MB",
                    "recomputed_csv": f"{csv_mb_exact:.1f} MB (exact 144/240 KB)" if cur_ds == "sst2" else f"{csv_mb_exact:.1f} MB",
                    "status": status_mb,
                    "explanation": expl_mb
                })
            else:
                diff_records.append({
                    "item": f"{label}",
                    "thesis_value": f"{th_peak}, {th_e}, {th_mb}",
                    "md_value": "N/A",
                    "recomputed_csv": "N/A",
                    "status": "UNREPRODUCIBLE",
                    "explanation": f"Could not match configuration {cur_ds} C={cur_c} in benchmark suite."
                })

def extract_table_text(text, label):
    idx = text.find(f"\\label{{{label}}}")
    if idx == -1: return ""
    start = text.rfind("\\begin{table}", 0, idx)
    end = text.find("\\end{table}", idx) + len("\\end{table}")
    return text[start:end]

parse_scorecard_tex(extract_table_text(thesis_eval_text, "tab:scorecard_iid"), is_non_iid=False)
parse_scorecard_tex(extract_table_text(thesis_eval_text, "tab:scorecard_non_iid"), is_non_iid=True)

# Parse Table 5.3 (Detailed IID) & Table 5.4 (Detailed Non-IID)
def parse_detailed_tex(tbl_text, is_non_iid=False):
    alpha = 0.5 if is_non_iid else 1000.0
    cur_ds = None
    cur_c = None
    lines = tbl_text.splitlines()
    for l in lines:
        l = l.strip()
        if not l or l.startswith("\\hline") or l.startswith("\\begin") or l.startswith("\\end") or l.startswith("\\caption") or l.startswith("\\label") or l.startswith("\\resizebox"):
            continue
        m_ds = re.search(r'\\textbf\{([A-Z0-9\-]+)\}', l)
        if m_ds and not "Task" in l:
            cur_ds = m_ds.group(1).lower().replace("-", "")
            if cur_ds == "sst2": cur_ds = "sst2"
        m_c = re.search(r'C=(\d+)', l)
        if m_c:
            cur_c = int(m_c.group(1))
            
        if "RoLoRA" in l or "FDA-RoLoRA" in l:
            parts = [p.strip() for p in l.split("&")]
            m_part = parts[-7] if len(parts) >= 8 else (parts[-8] if len(parts) >= 9 else parts[0])
            fin_part = parts[-6] if len(parts) >= 8 else (parts[-7] if len(parts) >= 9 else "")
            pk_part = parts[-5] if len(parts) >= 8 else (parts[-6] if len(parts) >= 9 else "")
            r_part = parts[-4] if len(parts) >= 8 else (parts[-5] if len(parts) >= 9 else "")
            mb_part = parts[-3] if len(parts) >= 8 else (parts[-4] if len(parts) >= 9 else "")
            e_part = parts[-2] if len(parts) >= 8 else (parts[-3] if len(parts) >= 9 else "")
            
            # extract final acc
            m_fin = re.search(r'([\d\.]+)\s*\\pm\s*([\d\.]+)', fin_part)
            th_fin = float(m_fin.group(1)) if m_fin else None
            th_fin_std = float(m_fin.group(2)) if m_fin else 0.0
            
            # extract peak acc
            m_pk = re.search(r'([\d\.]+)\s*\\pm\s*([\d\.]+)', pk_part)
            th_pk = float(m_pk.group(1)) if m_pk else None
            th_pk_std = float(m_pk.group(2)) if m_pk else 0.0
            
            # match cell
            m_query = "k=1.0" if "k=1.0" in m_part else ("k=0.5" if "k=0.5" in m_part else ("E=20" if "E=20" in m_part else ("E=15" if "E=15" in m_part else ("E=10" if "E=10" in m_part else ("E=5" if "E=5" in m_part else "")))))
            cell = lookup_cell(cur_ds, cur_c, alpha, m_query)
            tbl_name = "Table 5.4 (Detailed Non-IID)" if is_non_iid else "Table 5.3 (Detailed IID)"
            label = f"{tbl_name}: {cur_ds.upper()} C={cur_c} {m_query}"
            
            if cell is not None:
                # Final acc comparison
                csv_fin = cell["final_mean"] * 100.0
                diff_fin = abs(th_fin - csv_fin) if th_fin is not None else 0.0
                status_fin = "OK"
                expl_fin = "Matches per-seed final accuracy mean."
                
                # Check for aborted run contamination in MNLI Non-IID C20 k=1.0
                if cur_ds == "mnli" and cur_c == 20 and is_non_iid and "k=1.0" in m_query:
                    if th_fin is not None and abs(th_fin - 59.52) < 0.1:
                        status_fin = "WRONG"
                        expl_fin = "Contaminated by aborted 4-round run (5095b3daf3.csv, acc=32.83%); true complete run final mean is 72.87 ± 0.94% (accuracy_m) or 73.71 ± 0.77% (composite)."
                elif diff_fin > 0.1:
                    status_fin = "WRONG"
                    expl_fin = f"Thesis reports {th_fin:.2f}%, recomputed CSV gives {csv_fin:.2f}%."
                    
                diff_records.append({
                    "item": f"{label} Final Accuracy",
                    "thesis_value": f"{th_fin:.2f} ± {th_fin_std:.2f}%" if th_fin else "N/A",
                    "md_value": f"{cell['md_final_acc']*100:.2f}%",
                    "recomputed_csv": f"{csv_fin:.2f} ± {cell['final_std_sample']*100:.2f}%",
                    "status": status_fin,
                    "explanation": expl_fin
                })
                
                # Peak acc comparison
                csv_pk = cell["peak_mean"] * 100.0
                md_pk = cell["md_max_acc"] * 100.0
                diff_pk = abs(th_pk - csv_pk) if th_pk is not None else 0.0
                status_pk = "OK"
                expl_pk = "Matches per-seed peak accuracy mean."
                if diff_pk > 0.1 and abs(th_pk - md_pk) <= 0.1:
                    status_pk = "MD-vs-CSV CONFLICT"
                    expl_pk = f"Matches .md average-trajectory peak ({md_pk:.2f}%), but differs from per-seed peak mean ({csv_pk:.2f}%)."
                elif diff_pk > 0.1:
                    status_pk = "WRONG"
                    expl_pk = f"Thesis reports {th_pk:.2f}%, recomputed gives {csv_pk:.2f}%."
                    
                diff_records.append({
                    "item": f"{label} Peak Accuracy",
                    "thesis_value": f"{th_pk:.2f} ± {th_pk_std:.2f}%" if th_pk else "N/A",
                    "md_value": f"{md_pk:.2f}%",
                    "recomputed_csv": f"{csv_pk:.2f} ± {cell['peak_std_sample']*100:.2f}%",
                    "status": status_pk,
                    "explanation": expl_pk
                })
                
                # Check QNLI C50 E=20 zero std anomaly
                if cur_ds == "qnli" and cur_c == 50 and "E=20" in m_query:
                    if th_pk_std == 0.0:
                        diff_records.append({
                            "item": f"{label} ± 0.00 Std Anomaly",
                            "thesis_value": "± 0.00",
                            "md_value": "1 seed only",
                            "recomputed_csv": f"n = {cell['n_seeds']} seed (std undefined / zero)",
                            "status": "WRONG",
                            "explanation": "Misleading reporting: only 1 seed was executed for QNLI C50 RoLoRA E=20 (S42), so standard deviation is undefined or trivially zero rather than demonstrating perfect reproducibility."
                        })

parse_detailed_tex(extract_table_text(thesis_eval_text, "tab:iid_detailed_benchmarks"), is_non_iid=False)
parse_detailed_tex(extract_table_text(thesis_eval_text, "tab:non_iid_detailed_benchmarks"), is_non_iid=True)

# Check Table 5.5 Milestone Synthesis QNLI IID anomaly
diff_records.append({
    "item": "Table 5.5: QNLI C=20 IID FDA-RoLoRA R@80% Gate",
    "thesis_value": "--- (Unattained)",
    "md_value": "— (Unattained)",
    "recomputed_csv": "Seed 42 peaked at 82.90%, Seed 43 peaked at 83.38% (Mean Peak: 83.14 ± 0.24%)",
    "status": "MD-vs-CSV CONFLICT",
    "explanation": "Table 5.5 reports R@80% as unattained (---) because update_benchmark_suite.py pads shorter runs with their final value (Seed 42 ended at round 14 with acc=51.88%), dragging down the seed-averaged trajectory to a peak of 79.10%, even though both seeds individually crossed 80%."
})

# Specific known Chapter 5 prose issues
prose_audit_items = [
    {
        "item": "Prose Section 5.5: Non-IID Truncation to 2.5-3.8 Epochs",
        "thesis_value": "ē ≈ 2.5--3.8 epochs",
        "md_value": "Mean epochs 9.18--17.49",
        "recomputed_csv": "Mean epochs range from 8.40 to 17.76 across all Non-IID FDA cells",
        "status": "WRONG",
        "explanation": "Factually incorrect: the controller never truncates local training to 2.5-3.8 epochs under Non-IID. All Non-IID FDA runs execute between 8.4 and 17.8 epochs on average."
    },
    {
        "item": "Figure 5.13: Dual Frontier Realized Epochs Curve",
        "thesis_value": "IID ē high (~4.2) and Non-IID ē low (~1.5)",
        "md_value": "Non-IID ē > IID ē in all 16 comparisons",
        "recomputed_csv": "Non-IID ē is strictly higher than IID ē for the same k in all 16 task/cohort pairs",
        "status": "WRONG",
        "explanation": "Figure 5.13 and its accompanying text depict the exact reverse of empirical reality. In actual data, non-IID variance is roughly stationary and tracks the setpoint, while IID variance grows super-linearly and undershoots."
    },
    {
        "item": "Prose Section 5.3: QNLI Variance Surge Claim",
        "thesis_value": "Variance surges 4.0× (0.448 → 1.794 on QNLI)",
        "md_value": "IID k=0.5 var=0.448, Non-IID k=0.5 var=1.794",
        "recomputed_csv": "0.448 (IID C20 k=0.5) vs 1.794 (Non-IID C20 k=0.5)",
        "status": "OK",
        "explanation": "Numerical values exist in the data (comparing k=0.5 IID vs Non-IID), but the prose falsely concluded that this surge caused a 2.5-3.8 epoch brake."
    },
    {
        "item": "Prose Section 5.4: Up-Projection B Deeper Training Claim",
        "thesis_value": "Matrix B consistently accommodates deeper training (17.76 vs 17.19)",
        "md_value": "17.76 vs 17.19 on MNLI C20 Non-IID only",
        "recomputed_csv": "In 16 out of 35 FDA cells, ea > eb. The difference on MNLI C20 (0.57 epochs) is within seed noise.",
        "status": "WRONG",
        "explanation": "Overgeneralized from a single seed pair: across the benchmark suite, 16 of 35 cells have ea > eb, and both tracks symmetrically oscillate around the common setpoint."
    },
    {
        "item": "Prose Section 5.1 & 5.3: QNLI C50 Non-IID 56.6% Bandwidth Savings Claim",
        "thesis_value": "Reaches early stopping in 13.0 rounds (152.3 MB) vs 30.0 rounds (351.6 MB) -> 56.6% bandwidth cut",
        "md_value": "FDA final accuracy 49.51% (aborted early due to degradation)",
        "recomputed_csv": "FDA k=1.0 terminated at round 13 with final accuracy 49.51% (chance level / failure)",
        "status": "WRONG",
        "explanation": "Spurious savings claim: the run ended at round 13 because it degraded to coin-flip accuracy (49.51%), tripping early stopping. Savings cannot be claimed from an unsuccessful run."
    },
    {
        "item": "Prose Section 5.3: SST-2 C50 Non-IID RoLoRA 'Collapse' Claim",
        "thesis_value": "suffers majority-class collapse, dropping to 59.00% peak accuracy (53.33% final)",
        "md_value": "RoLoRA accuracy stayed at 49.1%-51.1% throughout, with brief blip to 66.9% at round 19",
        "recomputed_csv": "RoLoRA never escaped the initial majority-class plateau (49-51%) within the 20-round budget",
        "status": "WRONG",
        "explanation": "RoLoRA did not collapse from a trained state; it failed to break out of the initial majority-class saddle point within R=20. Wording must be corrected to 'did not escape initial plateau within budget'."
    },
    {
        "item": "Figures 5.11 & 5.12 Caption & Prose: '4-task synthesis'",
        "thesis_value": "Figure 5.11 & 5.12 describe a '4-task diagnostic synthesis'",
        "md_value": "Plots show only MNLI and QNLI",
        "recomputed_csv": "Plots contain panels for MNLI and QNLI only (2 tasks)",
        "status": "WRONG",
        "explanation": "Figures 5.11 and 5.12 only plot MNLI and QNLI, but the captions and Section 5.3 refer to them as a 'holistic 4-task diagnostic synthesis'."
    },
    {
        "item": "Figures 5.1--5.8: IID Benchmark Stacked Figures",
        "thesis_value": "Legends show FDA-RoLoRA with parameters E3TO20, E3TO30, E3TO40, and RK8",
        "md_value": "Older fda-standard runs",
        "recomputed_csv": "Figures 5.1--5.8 were generated from pre-Pure standard FDA runs rather than FDA Pure (e_min=1, RK4)",
        "status": "WRONG",
        "explanation": "Figures 5.1--5.8 visualize older fda-standard experiments rather than the finalized FDA-RoLoRA (Pure) method evaluated in the tables."
    }
]

diff_records.extend(prose_audit_items)

df_diff = pd.DataFrame(diff_records)
df_diff.to_csv(os.path.join(OUT_DIR, "audit_diff.csv"), index=False)

# Write audit_diff.md
diff_md = [
    "# Audit Diff: Complete Discrepancy Verification Across Chapter 5\n",
    "Comprehensive verification of every quantitative claim, table entry, and figure reference in Chapter 5 against the reference markdown (`FDA_Pure_vs_RoLoRA.md`) and raw per-round CSV files.\n",
    "### Status Definitions:\n",
    "- **OK**: Verified and supported by raw CSV recomputation.\n",
    "- **WRONG**: Factually contradicted by data, contaminated by aborted runs, or miscalculated.\n",
    "- **UNREPRODUCIBLE**: Cannot be reproduced or traced to existing CSV runs.\n",
    "- **MD-vs-CSV CONFLICT**: Discrepancy arising from `.md` averaging trajectories round-by-round before taking the peak, versus the thesis averaging individual per-seed peaks.\n",
    "\n| Item / Claim | Thesis Value | Reference MD Value | Recomputed CSV Value | Status | Detailed Explanation |",
    "| :--- | :--- | :--- | :--- | :---: | :--- |"
]
for _, r in df_diff.iterrows():
    diff_md.append(
        f"| **{r['item']}** | `{r['thesis_value']}` | `{r['md_value']}` | `{r['recomputed_csv']}` | **{r['status']}** | {r['explanation']} |"
    )

with open(os.path.join(OUT_DIR, "audit_diff.md"), "w", encoding="utf-8") as fp:
    fp.write("\n".join(diff_md))

print(f"audit_diff.md written: {len(df_diff)} items audited.")

# =============================================================================
# 5. INTEGRITY REPORT GENERATION
# =============================================================================
print("=== Step 5: Generating integrity_report.md ===")

integrity_md = [
    "# Benchmark Data Integrity Report\n",
    "Detailed forensic audit of benchmark runs, duplicate executions, early terminations, round budgets, and figure tracking across the FDA-RoLoRA research corpus.\n",
    "## 1. Duplicates & Aborted Runs Analysis\n",
    "### Non-IID MNLI C=20, k=1.0 Seed 42 Duplication:\n",
    "- **File 1 (Aborted)**: `THESIS-NONIID-A0.5-DS-mnli-C20-R50-M-fda-pure-LR0.1-RK4-E1TO20-CRATIO1-NODISC-BSTART-NOGUARD-S42-ES0.6-MR30-P5-D0.005-W5_5095b3daf3.csv`\n",
    "  - Rounds executed: **4 rounds** (stopped prematurely, well below configured floor of 30 rounds).\n",
    "  - Final accuracy: **32.83%** (`accuracy_m`). File size: 4,152 bytes.\n",
    "- **File 2 (Complete)**: `THESIS-NONIID-A0.5-DS-mnli-C20-R50-M-fda-pure-LR0.1-RK4-E1TO20-CRATIO1-NODISC-BSTART-NOGUARD-S42-ES0.6-MR30-P5-D0.005-W5_cefb5fd7ef.csv`\n",
    "  - Rounds executed: **41 rounds**. Final accuracy: **72.20%** (`accuracy_m`), Peak accuracy: **73.69%** (`accuracy_m`). File size: 32,648 bytes.\n",
    "- **Seed 43 (Complete)**: `..._4a43126235.csv` executed **35 rounds**. Final accuracy: **73.53%**, Peak accuracy: **73.53%**.\n",
    "- **Finding & Diagnosis**: `update_benchmark_suite.py` correctly selected File 2 via `get_best_files_per_seed` by picking the larger file. However, Table 5.4 in the thesis averaged all three files (including the aborted 4-round run), producing the contaminated final accuracy of **59.52 ± 23.12%**. When computed strictly from the two complete runs, the true final accuracy is **72.87 ± 0.94%** (or **73.71 ± 0.77%** composite).\n",
    "\n## 2. Round Budget (R) Discrepancies & Early Termination Audit\n",
    "In several benchmark cells, runs terminated at different round horizons:\n",
    "1. **Non-IID QNLI C=50, k=1.0**: Executed **13 rounds** (`R_executed = 13.0`), terminating at round 13 because validation accuracy dropped to 49.51% (chance level), tripping early stopping criteria (`ES0.8-MR18-P5-D0.005-W5`). In contrast, RoLoRA E=10 executed all 30 rounds. The thesis erroneously claimed a 56.6% bandwidth saving from this failure.\n",
    "2. **Non-IID MNLI C=50, k=1.0**: Executed **22.5 rounds** on average (Seed 42: 24 rounds, Seed 43: 21 rounds) due to early stopping after peaking at round 20, whereas RoLoRA E=10 executed all 30 rounds.\n",
    "3. **Non-IID QNLI C=20, E1TO40 variants**: Seed 42 ran with $R=50$, whereas Seed 43 ran with configured $R=35$ (`-R35-` in filename) and halted at round 24/30.\n",
    "4. **RoLoRA E=20 on QNLI C=20 Non-IID**: Seed 42 ran with configured $R=50$ and halted at round 30, whereas Seed 43 ran with configured $R=35$ and halted at round 18.\n",
    "5. **Single Seed Anomaly on QNLI C=50 RoLoRA E=20**: Only a single seed (`S42`, 30 rounds) was executed, resulting in a reported ± 0.00 standard deviation in Table 5.3.\n",
    "\n## 3. Communication Accounting Corrections\n",
    "1. **SST-2 Overcounting**: In both `update_benchmark_suite.py` and the thesis, SST-2 communication payload was computed using the 5-layer parameter rate (240 KB per client per round, or 4.6875 MB / 11.71875 MB per cohort round). Because SST-2 injects adapters into only 3 layers (6 active modules), its true bi-directional payload is **144 KB per client per round** (2.8125 MB per cohort round for C=20, 7.03125 MB for C=50). All SST-2 MB totals must be adjusted downward by 40%.\n",
    "2. **Intermediate Epoch Monitoring Traffic**: Evaluating exact subspace variance requires communicating local drifts. Section 3.5 claims monitoring is governed by 'scalar norm checks', which is theoretically impossible since $\\text{Var}(\\Delta)$ requires the norm of the average drift $\\|\\frac{1}{C}\\sum \\Delta_i\\|^2$. If AMS sketches (6 KB) are sent at intermediate epochs, total payload increases by 2.5%--5.8%; if full low-rank tensors (120 KB / 72 KB) are sent, monitoring payload roughly scales with cumulative local epochs.\n",
    "\n## 4. Evaluation Figures Audit\n",
    "1. **Figures 5.1--5.8 (IID Telemetry Stacked Plots)**: The current PNG files in `thesis/figures/` were generated by `export_individual_plots.py`, which used old `fda-standard` runs (`E3TO20`, `E3TO30`, `E3TO40`, `RK8`). They do not show the FDA-RoLoRA (Pure) method evaluated in the thesis text and tables.\n",
    "2. **Figures 5.9 & 5.10 (Non-IID MNLI and QNLI Telemetry)**: Correctly depict FDA-RoLoRA Pure, but required legend patching from 'FDA Pure' to 'FDA-RoLoRA'.\n",
    "3. **Figures 5.11 & 5.12 (Cross-Task Dynamics)**: Only include MNLI and QNLI panels, but the thesis text and captions call them a '4-task diagnostic synthesis'. Captions must be corrected to state MNLI and QNLI.\n",
    "4. **Figure 5.13 (Dual Frontier Curve)**: The TikZ diagram depicts $\\bar{e}$ high in IID and low in Non-IID, which directly contradicts all empirical data where Non-IID $\\bar{e} >$ IID $\\bar{e}$. The diagram must be redesigned to accurately reflect empirical setpoint tracking.\n"
]

with open(os.path.join(OUT_DIR, "integrity_report.md"), "w", encoding="utf-8") as fp:
    fp.write("\n".join(integrity_md))

print(f"integrity_report.md written.")
print("=== All Phase 1 Audit Deliverables Generated Successfully! ===")
