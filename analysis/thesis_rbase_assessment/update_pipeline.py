import os
import re
import glob
import json
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT_DIR = "analysis/thesis_rbase_assessment"
FIG_DIR = os.path.join(OUT_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 8.5,
    'figure.titlesize': 14,
    'lines.linewidth': 2.0,
    'lines.markersize': 4.5,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linestyle': '--'
})

csv_files = glob.glob("results/round_metrics/*THESIS-RBASE-MAIN*.csv")
print(f"Total matching CSVs found: {len(csv_files)}")

parsed_runs = []
for f in csv_files:
    bn = os.path.basename(f).replace(".csv", "")
    bn_clean = re.sub(r"_[0-9a-f]{10}$", "", bn)
    s = re.sub(r"^THESIS-RBASE-MAIN(-SMA-W\d+)?-DS-", "", bn_clean)
    m = re.match(r"^(?P<ds>[^-]+)-C(?P<clients>\d+)-R(?P<rounds>\d+)-M-(?P<rest>.*)$", s)
    if not m:
        print(f"Warning: unparsed file {bn}")
        continue
    ds = m.group("ds")
    c = int(m.group("clients"))
    rounds = int(m.group("rounds"))
    rest = m.group("rest")
    
    if rest.startswith("fda-standard"):
        method = "FDA-Standard"
    elif rest.startswith("fda-sma"):
        method = "FDA-SMA (W=3)"
    elif rest.startswith("ffa"):
        method = "FFA-LoRA"
    elif rest.startswith("rolora"):
        method = "RoLoRA"
    elif rest.startswith("lora"):
        method = "LoRA"
    else:
        method = "Other"
        
    lr_m = re.search(r"-LR([\d\.]+)", rest)
    lr = float(lr_m.group(1)) if lr_m else None
    
    rk_m = re.search(r"-RK(\d+)", rest)
    rk = int(rk_m.group(1)) if rk_m else 4
    
    ep_m = re.search(r"-E(\d+TO\d+|\d+)", rest)
    ep = ep_m.group(1) if ep_m else "20"
    
    seed_m = re.search(r"-S(\d+)", rest)
    seed = int(seed_m.group(1)) if seed_m else None
    
    es_m = re.search(r"-ES([\d\.]+)", rest)
    mr_m = re.search(r"-MR(\d+)", rest)
    p_m = re.search(r"-P(\d+)", rest)
    d_m = re.search(r"-D([\d\.]+)", rest)
    w_m = re.search(r"-W(\d+)", rest)
    
    es = float(es_m.group(1)) if es_m else None
    mr = float(mr_m.group(1)) if mr_m else None
    p = float(p_m.group(1)) if p_m else None
    d = float(d_m.group(1)) if d_m else None
    w = float(w_m.group(1)) if w_m else None
    
    base_id = re.sub(r"-S\d+", "", bn_clean)
    
    df_csv = pd.read_csv(f)
    if df_csv.empty:
        print(f"Empty CSV: {f}")
        continue
        
    if ds == "mnli" and "accuracy_m" in df_csv.columns:
        df_csv["accuracy"] = (df_csv["accuracy_m"] + df_csv["accuracy_mm"]) / 2.0
        
    final_acc = df_csv["accuracy"].iloc[-1]
    peak_acc = df_csv["accuracy"].max()
    best_rd = int(df_csv.loc[df_csv["accuracy"].idxmax(), "round"])
    total_rds = int(df_csv["round"].iloc[-1])
    best_idx = df_csv["accuracy"].idxmax()
    time_to_peak = df_csv.loc[:best_idx, "time_round_total"].sum()
    total_time = df_csv["time_round_total"].sum()
    
    parsed_runs.append({
        "run_id": bn_clean,
        "base_id": base_id,
        "dataset": ds,
        "clients": c,
        "configured_rounds": rounds,
        "method": method,
        "lr": lr,
        "rank": rk,
        "epochs": ep,
        "seed": seed,
        "es_threshold": es,
        "es_min_rounds": mr,
        "es_patience": p,
        "es_min_delta": d,
        "es_window": w,
        "final_accuracy": final_acc,
        "peak_accuracy": peak_acc,
        "best_round": best_rd,
        "total_rounds_completed": total_rds,
        "time_to_peak_seconds": time_to_peak,
        "total_time_seconds": total_time,
        "csv_path": f,
        "df": df_csv
    })

df_all = pd.DataFrame([{k: v for k, v in r.items() if k != 'df'} for r in parsed_runs])
df_all.sort_values(by=["dataset", "clients", "configured_rounds", "base_id", "seed"], inplace=True)
df_all.to_csv(os.path.join(OUT_DIR, "all_runs_extracted.csv"), index=False)
print(f"Saved all_runs_extracted.csv with {len(df_all)} runs.")

# Aggregation
def get_agg(g):
    seeds = sorted(g['seed'].tolist())
    n = len(seeds)
    return pd.Series({
        'n_seeds': n,
        'seeds': str(seeds),
        'final_acc_mean': g['final_accuracy'].mean() * 100.0,
        'final_acc_std': g['final_accuracy'].std(ddof=0) * 100.0 if n > 1 else 0.0,
        'peak_acc_mean': g['peak_accuracy'].mean() * 100.0,
        'peak_acc_std': g['peak_accuracy'].std(ddof=0) * 100.0 if n > 1 else 0.0,
        'best_round_mean': g['best_round'].mean(),
        'best_round_std': g['best_round'].std(ddof=0) if n > 1 else 0.0,
        'rounds_completed_mean': g['total_rounds_completed'].mean(),
        'rounds_completed_std': g['total_rounds_completed'].std(ddof=0) if n > 1 else 0.0,
        'time_to_peak_min_mean': (g['time_to_peak_seconds'] / 60.0).mean(),
        'time_to_peak_min_std': (g['time_to_peak_seconds'] / 60.0).std(ddof=0) if n > 1 else 0.0,
        'total_time_min_mean': (g['total_time_seconds'] / 60.0).mean(),
        'total_time_min_std': (g['total_time_seconds'] / 60.0).std(ddof=0) if n > 1 else 0.0,
    })

group_cols = ['dataset', 'clients', 'configured_rounds', 'base_id', 'method', 'rank', 'epochs', 'lr', 'es_threshold']
df_agg = df_all.groupby(group_cols, dropna=False).apply(get_agg, include_groups=False).reset_index()
df_agg.sort_values(by=["dataset", "clients", "configured_rounds", "method", "rank", "epochs"], inplace=True)
df_agg.to_csv(os.path.join(OUT_DIR, "aggregated_metrics.csv"), index=False)
print(f"Saved aggregated_metrics.csv with {len(df_agg)} aggregated configurations.")

lr4_e1_prefix = "THESIS-RBASE-LR4-E1-DS-"
lr4_e1_csv_files = glob.glob(f"results/round_metrics/{lr4_e1_prefix}*.csv")
print(f"Total LR4-E1 CSVs found: {len(lr4_e1_csv_files)}")

lr4_e1_runs = []
for f in lr4_e1_csv_files:
    bn = os.path.basename(f).replace(".csv", "")
    bn_clean = re.sub(r"_[0-9a-f]{10}$", "", bn)
    run_spec = bn_clean.removeprefix(lr4_e1_prefix)
    match = re.match(
        r"^(?P<ds>[^-]+)-C(?P<clients>\d+)-R(?P<rounds>\d+)-M-(?P<rest>.*)$",
        run_spec,
    )
    if not match:
        raise ValueError(f"Unable to parse LR4-E1 run filename: {bn}")

    rest = match.group("rest")
    if rest.startswith("fda-standard"):
        method = "FDA-Standard"
    elif rest.startswith("rolora"):
        method = "RoLoRA"
    elif rest.startswith("lora"):
        method = "LoRA"
    else:
        raise ValueError(f"Unsupported LR4-E1 method in filename: {bn}")

    lr_match = re.search(r"-LR([\d.]+)", rest)
    rank_match = re.search(r"-RK(\d+)", rest)
    seed_match = re.search(r"-S(\d+)", rest)
    epoch_match = re.search(r"-E(\d+TO\d+|\d+)", rest)
    if not lr_match or not rank_match or not seed_match:
        raise ValueError(f"Missing LR, rank, or seed metadata in filename: {bn}")

    run_metrics = pd.read_csv(f)
    required_columns = {"accuracy", "round", "time_round_total"}
    missing_columns = required_columns.difference(run_metrics.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Missing columns ({missing}) in LR4-E1 file: {f}")
    if run_metrics.empty:
        raise ValueError(f"Empty LR4-E1 CSV: {f}")

    if epoch_match:
        epochs = epoch_match.group(1)
    elif "local_epochs" in run_metrics.columns:
        epochs = str(int(run_metrics["local_epochs"].iloc[0]))
    else:
        epochs = "1"

    best_idx = run_metrics["accuracy"].idxmax()
    lr4_e1_runs.append({
        "run_id": bn_clean,
        "dataset": match.group("ds"),
        "clients": int(match.group("clients")),
        "configured_rounds": int(match.group("rounds")),
        "method": method,
        "lr": float(lr_match.group(1)),
        "rank": int(rank_match.group(1)),
        "epochs": epochs,
        "seed": int(seed_match.group(1)),
        "final_accuracy": run_metrics["accuracy"].iloc[-1],
        "peak_accuracy": run_metrics["accuracy"].max(),
        "best_round": int(run_metrics.loc[best_idx, "round"]),
        "total_rounds_completed": int(run_metrics["round"].iloc[-1]),
        "time_to_peak_seconds": run_metrics.loc[:best_idx, "time_round_total"].sum(),
        "total_time_seconds": run_metrics["time_round_total"].sum(),
        "csv_path": f,
    })

df_lr4_e1 = pd.DataFrame(lr4_e1_runs)
df_lr4_e1.sort_values(by=["dataset", "method", "lr"], inplace=True)
df_lr4_e1.to_csv(os.path.join(OUT_DIR, "lr4_e1_metrics.csv"), index=False)
print(f"Saved lr4_e1_metrics.csv with {len(df_lr4_e1)} runs.")

# THESIS-RBASE-CALIB-DS parsing
calib_prefix = "THESIS-RBASE-CALIB-DS-"
calib_csv_files = glob.glob(f"results/round_metrics/{calib_prefix}*.csv")
print(f"Total CALIB CSVs found: {len(calib_csv_files)}")

calib_runs = []
for f in sorted(calib_csv_files):
    bn = os.path.basename(f)
    m = re.match(r"THESIS-RBASE-CALIB-DS-(?P<ds>[^-]+)-C(?P<clients>\d+)-R(?P<rounds>\d+)-M-fda-calib-LR(?P<lr>[\d\.]+)-RK(?P<rk>\d+)-E(?P<emin>\d+)TO(?P<emax>\d+)-CRATIO(?P<cratio>[\d\.]+)-(?P<disc>[^-]+)-S(?P<seed>\d+)-ES(?P<es>[\d\.]+)-MR(?P<mr>\d+)-P(?P<p>\d+)-D(?P<d>[\d\.]+)-W(?P<w>\d+)", bn)
    if not m:
        continue
    d = m.groupdict()
    df_c = pd.read_csv(f)
    if d["ds"] == "mnli" and "accuracy_m" in df_c.columns:
        df_c["accuracy"] = (df_c["accuracy_m"] + df_c["accuracy_mm"]) / 2.0
    cratio_f = float(d["cratio"])
    cratio_label = "1.0" if d["cratio"] == "1" else d["cratio"]
    disc_label = d["disc"]
    best_idx = df_c["accuracy"].idxmax()
    
    calib_runs.append({
        "run_id": bn.replace(".csv", ""),
        "dataset": d["ds"],
        "clients": int(d["clients"]),
        "configured_rounds": int(d["rounds"]),
        "lr": float(d["lr"]),
        "rank": int(d["rk"]),
        "e_min": int(d["emin"]),
        "e_max": int(d["emax"]),
        "controller_ratio": cratio_f,
        "cratio_label": cratio_label,
        "variance_discount": disc_label,
        "b_discount_active": disc_label == "BDISC3",
        "variant": f"CRATIO{cratio_label}-{disc_label}",
        "seed": int(d["seed"]),
        "es_threshold": float(d["es"]),
        "final_accuracy": df_c["accuracy"].iloc[-1] * 100.0,
        "peak_accuracy": df_c["accuracy"].max() * 100.0,
        "best_round": int(df_c.loc[best_idx, "round"]),
        "total_rounds_completed": int(df_c["round"].iloc[-1]),
        "mean_epochs": df_c["round_epochs"].mean() if "round_epochs" in df_c.columns else None,
        "time_to_peak_min": df_c.loc[:best_idx, "time_round_total"].sum() / 60.0,
        "total_time_min": df_c["time_round_total"].sum() / 60.0,
        "csv_path": f,
    })

df_calib = pd.DataFrame(calib_runs)
df_calib.sort_values(["dataset", "clients", "configured_rounds", "controller_ratio", "variance_discount"], inplace=True)
df_calib.to_csv(os.path.join(OUT_DIR, "calib_metrics.csv"), index=False)
print(f"Saved calib_metrics.csv with {len(df_calib)} runs.")


# Plotting Styles
def get_style(method, rk, epochs, es):
    if method == "FDA-Calibrated":
        label = f"FDA-Calibrated (r={rk}, {epochs})"
        return label, "#800080", "-", "D"

    label = f"{method} (r={rk}, E={epochs}"
    if es is not None and not (isinstance(es, float) and np.isnan(es)) and str(es) != "None":
        label += f", ES={es})"
    else:
        label += ")"
        
    if method == "FDA-Standard":
        if "3TO30" in str(epochs) and rk == 8:
            return label, "#0047AB", "--", "^"
        elif "3TO30" in str(epochs):
            return label, "#1f77b4", "-", "o"
        else:
            return label, "#0e879c", "-.", "s"
    elif method == "FDA-SMA (W=3)":
        return label, "#008080", "--", "D"
    elif method == "LoRA":
        if str(epochs) == "20" and rk == 8:
            return label, "#800026", "--", "^"
        elif str(epochs) == "20":
            return label, "#b2182b", "-", "d"
        elif str(epochs) == "10":
            return label, "#d62728", "--", "s"
        elif str(epochs) == "5":
            return label, "#ff7f0e", "-.", "o"
        else:
            return label, "#f46d43", ":", "x"
    elif method == "RoLoRA":
        if str(epochs) == "20" and rk == 8:
            return label, "#00441b", "--", "^"
        elif str(epochs) == "20":
            return label, "#005a32", "-", "d"
        elif str(epochs) == "10":
            return label, "#1b7837", "--", "s"
        elif str(epochs) == "5":
            return label, "#2ca02c", "-.", "o"
        else:
            return label, "#66c2a5", ":", "x"
    elif method == "FFA-LoRA":
        if str(epochs) == "10":
            return label, "#7570b3", "-", "s"
        else:
            return label, "#9467bd", "--", "o"
            
    return label, "#333333", "-", "x"

from collections import defaultdict
combos = defaultdict(lambda: defaultdict(list))
for r in parsed_runs:
    combos[(r["dataset"], r["clients"], r["configured_rounds"])][r["base_id"]].append({
        "rid": r["run_id"],
        "seed": r["seed"],
        "method": r["method"],
        "rank": r["rank"],
        "epochs": r["epochs"],
        "es": r["es_threshold"],
        "rounds": r["df"]["round"].values,
        "accuracy": r["df"]["accuracy"].values * 100.0,
        "cum_time_min": (r["df"]["time_round_total"].cumsum().values) / 60.0,
        "max_acc": r["peak_accuracy"] * 100.0,
        "time_to_best_min": r["time_to_peak_seconds"] / 60.0,
        "best_round": r["best_round"],
        "total_rounds": r["total_rounds_completed"]
    })

# Add chosen Calibrated FDA runs into combos (replacing FDA-SMA in benchmark sessions)
CALIB_BEST_FILES = {
    ("mnli", 20, 50): ("results/round_metrics/THESIS-RBASE-CALIB-DS-mnli-C20-R50-M-fda-calib-LR0.1-RK4-E3TO20-CRATIO0.75-NODISC-S42-ES0.6-MR6-P3-D0.005-W3_2b31090468.csv", "$\\kappa=0.75$", 0.60),
    ("mnli", 50, 30): ("results/round_metrics/THESIS-RBASE-CALIB-DS-mnli-C50-R30-M-fda-calib-LR0.1-RK4-E3TO20-CRATIO1-BDISC3-S42-ES0.55-MR6-P3-D0.005-W3_6ac953b6cd.csv", "$\\kappa=1.0$", 0.55),
    ("mrpc", 20, 30): ("results/round_metrics/THESIS-RBASE-CALIB-DS-mrpc-C20-R30-M-fda-calib-LR0.05-RK4-E3TO20-CRATIO1-NODISC-S42-ES0.7-MR6-P3-D0.005-W3_b5ff21fea5.csv", "$\\kappa=1.0$", 0.70),
    ("mrpc", 50, 12): ("results/round_metrics/THESIS-RBASE-CALIB-DS-mrpc-C50-R12-M-fda-calib-LR0.05-RK4-E3TO20-CRATIO1-BDISC3-S42-ES0.7-MR6-P3-D0.005-W3_a85262ad15.csv", "$\\kappa=1.0$", 0.70),
    ("qnli", 20, 50): ("results/round_metrics/THESIS-RBASE-CALIB-DS-qnli-C20-R50-M-fda-calib-LR0.1-RK4-E3TO20-CRATIO1-BDISC3-S42-ES0.8-MR6-P3-D0.005-W3_bdfaeeb38f.csv", "$\\kappa=1.0$", 0.80),
    ("qnli", 50, 30): ("results/round_metrics/THESIS-RBASE-CALIB-DS-qnli-C50-R30-M-fda-calib-LR0.1-RK4-E3TO20-CRATIO1-NODISC-S42-ES0.8-MR6-P3-D0.005-W3_91f5c8d337.csv", "$\\kappa=1.0$", 0.80),
    ("qqp",  50, 30): ("results/round_metrics/THESIS-RBASE-CALIB-DS-qqp-C50-R30-M-fda-calib-LR0.1-RK4-E3TO20-CRATIO1-BDISC3-S42-ES0.75-MR6-P3-D0.005-W3_67d492e88e.csv", "$\\kappa=1.0$", 0.75),
    ("rte",  20, 30): ("results/round_metrics/THESIS-RBASE-CALIB-DS-rte-C20-R30-M-fda-calib-LR0.05-RK4-E3TO20-CRATIO1-BDISC3-S42-ES0.6-MR6-P3-D0.005-W3_a4cb3854fe.csv", "$\\kappa=1.0$", 0.60),
    ("rte",  50, 12): ("results/round_metrics/THESIS-RBASE-CALIB-DS-rte-C50-R12-M-fda-calib-LR0.05-RK4-E3TO20-CRATIO1-BDISC3-S42-ES0.6-MR6-P3-D0.005-W3_7029b9b2aa.csv", "$\\kappa=1.0$", 0.60),
    ("sst2", 50, 30): ("results/round_metrics/THESIS-RBASE-CALIB-DS-sst2-C50-R30-M-fda-calib-LR0.1-RK4-E3TO20-CRATIO1-BDISC3-S42-ES0.85-MR6-P3-D0.005-W3_b66f0b3dcf.csv", "$\\kappa=1.0$", 0.85),
}
for (c_ds, c_c, c_r), (c_path, c_label, c_es) in CALIB_BEST_FILES.items():
    if not os.path.exists(c_path):
        continue
    df_c = pd.read_csv(c_path)
    if c_ds == "mnli" and "accuracy_m" in df_c.columns:
        df_c["accuracy"] = (df_c["accuracy_m"] + df_c["accuracy_mm"]) / 2.0
    best_idx = df_c["accuracy"].idxmax()
    base_id = f"THESIS-RBASE-CALIB-DS-{c_ds}-C{c_c}-R{c_r}-M-fda-calib"
    combos[(c_ds, c_c, c_r)][base_id].append({
        "rid": os.path.basename(c_path).replace(".csv", ""),
        "seed": 42,
        "method": "FDA-Calibrated",
        "rank": 4,
        "epochs": c_label,
        "es": c_es,
        "rounds": df_c["round"].values,
        "accuracy": df_c["accuracy"].values * 100.0,
        "cum_time_min": (df_c["time_round_total"].cumsum().values) / 60.0,
        "max_acc": df_c["accuracy"].max() * 100.0,
        "time_to_best_min": df_c.loc[:best_idx, "time_round_total"].sum() / 60.0,
        "best_round": int(df_c.loc[best_idx, "round"]),
        "total_rounds": int(df_c["round"].iloc[-1])
    })

def plot_benchmark_panel(cfgs, ds, c, rounds, fig_basename, title_prefix=""):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.2))
    
    for base_id, r_list in sorted(cfgs.items()):
        r0 = r_list[0]
        label, color, ls, marker = get_style(r0["method"], r0["rank"], r0["epochs"], r0["es"])
        n_seeds = len(r_list)
        seed_label = f"{label} (N={n_seeds})"
        
        max_r_completed = max(len(x["rounds"]) for x in r_list)
        common_rounds = np.arange(1, max_r_completed + 1)
        
        aligned_accs = []
        aligned_times = []
        
        for x in r_list:
            r_arr = x["rounds"]
            a_arr = x["accuracy"]
            t_arr = x["cum_time_min"]
            
            full_a = np.zeros(max_r_completed)
            full_t = np.zeros(max_r_completed)
            
            n_actual = len(r_arr)
            full_a[:n_actual] = a_arr
            full_t[:n_actual] = t_arr
            
            if n_actual < max_r_completed:
                full_a[n_actual:] = a_arr[-1]
                full_t[n_actual:] = t_arr[-1]
                
            aligned_accs.append(full_a)
            aligned_times.append(full_t)
            
        mean_acc = np.mean(aligned_accs, axis=0)
        std_acc = np.std(aligned_accs, axis=0) if n_seeds > 1 else np.zeros_like(mean_acc)
        mean_time = np.mean(aligned_times, axis=0)
        
        markevery = max(1, len(common_rounds) // 10)
        
        ax1.plot(common_rounds, mean_acc, label=seed_label, color=color, linestyle=ls,
                 marker=marker, markevery=markevery)
                             
        ax2.plot(mean_time, mean_acc, label=seed_label, color=color, linestyle=ls,
                 marker=marker, markevery=markevery)

    display_title = f"{ds.upper()} (C={c}, R={rounds})"
    if title_prefix:
        display_title += f" -- {title_prefix}"
        
    ax1.set_title(f"{display_title} : Accuracy vs. Round", fontweight='bold')
    ax1.set_xlabel("Communication Round")
    ax1.set_ylabel("Test Accuracy (%)")
    ax1.set_xlim(left=1, right=rounds)
    
    ax2.set_title(f"{display_title} : Accuracy vs. Wall-Clock Time", fontweight='bold')
    ax2.set_xlabel("Elapsed Wall-Clock Time (Minutes)")
    ax2.set_ylabel("Test Accuracy (%)")
    
    ax1.legend(loc="best", framealpha=0.85, edgecolor='#cccccc', fontsize=8)
    ax2.legend(loc="best", framealpha=0.85, edgecolor='#cccccc', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, f"{fig_basename}.pdf"), dpi=300, bbox_inches="tight")
    plt.savefig(os.path.join(FIG_DIR, f"{fig_basename}.png"), dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Generated {fig_basename}")

# 1. Full combination figures
combo_order = [
    ("mnli", 20, 50, "fig1_mnli_c20_r50"),
    ("mnli", 50, 30, "fig1_mnli_c50_r30"),
    ("mrpc", 20, 30, "fig2_mrpc_c20_r30"),
    ("mrpc", 50, 12, "fig3_mrpc_c50_r12"),
    ("qnli", 20, 50, "fig4_qnli_c20_r50"),
    ("qnli", 50, 30, "fig4_qnli_c50_r30"),
    ("qqp", 20, 50, "fig5_qqp_c20_r50"),
    ("qqp", 50, 30, "fig5_qqp_c50_r30"),
    ("rte", 20, 30, "fig6_rte_c20_r30"),
    ("rte", 50, 12, "fig7_rte_c50_r12"),
    ("sst2", 20, 50, "fig8_sst2_c20_r50"),
    ("sst2", 20, 75, "fig8_sst2_c20_r75"),
    ("sst2", 50, 30, "fig9_sst2_c50_r30")
]

for ds, c, rounds, fig_basename in combo_order:
    cfgs = {k: v for k, v in combos[(ds, c, rounds)].items() if v[0]["method"] != "FDA-SMA (W=3)"}
    plot_benchmark_panel(cfgs, ds, c, rounds, fig_basename)

# 2. Sub-regime figures for Report Sections
# Section 1: MNLI
mnli20_all = {k: v for k, v in combos[("mnli", 20, 50)].items() if v[0]["method"] != "FDA-SMA (W=3)"}
plot_benchmark_panel(mnli20_all, "mnli", 20, 50, "fig1_mnli_c20_r50_e5", "Epochs E=5 Regime")

mnli_all = combos[("mnli", 50, 30)]
plot_benchmark_panel({k: v for k, v in mnli_all.items() if v[0]["method"] in ["FDA-Standard", "FDA-Calibrated"] or v[0]["epochs"] == "5"},
                     "mnli", 50, 30, "fig1a_mnli_c50_e5", "Epochs E=5 Regime")
plot_benchmark_panel({k: v for k, v in mnli_all.items() if v[0]["method"] in ["FDA-Standard", "FDA-Calibrated"] or v[0]["epochs"] == "10"},
                     "mnli", 50, 30, "fig1b_mnli_c50_e10", "Epochs E=10 Regime")

# Section 2: MRPC C20 R30
mrpc20_all = combos[("mrpc", 20, 30)]
plot_benchmark_panel({k: v for k, v in mrpc20_all.items() if v[0]["method"] in ["FDA-Standard", "FDA-Calibrated"] or v[0]["epochs"] == "5"},
                     "mrpc", 20, 30, "fig2a_mrpc_c20_e5", "Epochs E=5 Regime")
plot_benchmark_panel({k: v for k, v in mrpc20_all.items() if v[0]["method"] in ["FDA-Standard", "FDA-Calibrated"] or v[0]["epochs"] == "10"},
                     "mrpc", 20, 30, "fig2b_mrpc_c20_e10", "Epochs E=10 Regime")

# Section 3: MRPC C50 R12
mrpc50_all = combos[("mrpc", 50, 12)]
plot_benchmark_panel({k: v for k, v in mrpc50_all.items() if v[0]["method"] in ["FDA-Standard", "FDA-Calibrated"] or v[0]["epochs"] == "5"},
                     "mrpc", 50, 12, "fig3a_mrpc_c50_e5", "Epochs E=5 Regime")
plot_benchmark_panel({k: v for k, v in mrpc50_all.items() if v[0]["method"] in ["FDA-Standard", "FDA-Calibrated"] or v[0]["epochs"] == "10"},
                     "mrpc", 50, 12, "fig3b_mrpc_c50_e10", "Epochs E=10 Regime")

# Section 4: QNLI C20 R50 (NEW)
qnli20_all = {k: v for k, v in combos[("qnli", 20, 50)].items() if v[0]["method"] != "FDA-SMA (W=3)"}
plot_benchmark_panel(qnli20_all, "qnli", 20, 50, "fig4_qnli_c20_r50_e5", "Epochs E=5 Regime")

# Section 5: QNLI C50 R30
qnli50_all = combos[("qnli", 50, 30)]
plot_benchmark_panel({k: v for k, v in qnli50_all.items() if v[0]["method"] in ["FDA-Standard", "FDA-Calibrated"] or v[0]["epochs"] == "5"},
                     "qnli", 50, 30, "fig4a_qnli_c50_e5", "Epochs E=5 Regime")
plot_benchmark_panel({k: v for k, v in qnli50_all.items() if v[0]["method"] in ["FDA-Standard", "FDA-Calibrated"] or v[0]["epochs"] in ["10", "20"]},
                     "qnli", 50, 30, "fig4b_qnli_c50_e10", "Epochs E=10 & Baseline Regime")

# Section 6: QQP
qqp20_all = {k: v for k, v in combos[("qqp", 20, 50)].items() if v[0]["method"] != "FDA-SMA (W=3)"}
plot_benchmark_panel(qqp20_all, "qqp", 20, 50, "fig5_qqp_c20_r50_e5", "Epochs E=5 Regime")

qqp_all = combos[("qqp", 50, 30)]
plot_benchmark_panel({k: v for k, v in qqp_all.items() if v[0]["method"] in ["FDA-Standard", "FDA-Calibrated"] or v[0]["epochs"] == "5"},
                     "qqp", 50, 30, "fig5_qqp_c50_e5", "Epochs E=5 Regime")
plot_benchmark_panel({k: v for k, v in qqp_all.items() if v[0]["method"] in ["FDA-Standard", "FDA-Calibrated"] or v[0]["epochs"] == "10"},
                     "qqp", 50, 30, "fig5b_qqp_c50_e10", "Epochs E=10 Regime")

# Section 7: RTE C20 R30
rte20_all = combos[("rte", 20, 30)]
plot_benchmark_panel({k: v for k, v in rte20_all.items() if v[0]["method"] in ["FDA-Standard", "FDA-Calibrated"] or v[0]["epochs"] == "5"},
                     "rte", 20, 30, "fig6a_rte_c20_e5", "Epochs E=5 Regime")
plot_benchmark_panel({k: v for k, v in rte20_all.items() if v[0]["method"] in ["FDA-Standard", "FDA-Calibrated"] or v[0]["epochs"] in ["10", "20"]},
                     "rte", 20, 30, "fig6b_rte_c20_e10", "Epochs E=10 & Baseline Regime")

# Section 8: RTE C50 R12
rte50_all = combos[("rte", 50, 12)]
plot_benchmark_panel({k: v for k, v in rte50_all.items() if v[0]["method"] in ["FDA-Standard", "FDA-Calibrated"] or v[0]["epochs"] == "5"},
                     "rte", 50, 12, "fig7a_rte_c50_e5", "Epochs E=5 Regime")
plot_benchmark_panel({k: v for k, v in rte50_all.items() if v[0]["method"] in ["FDA-Standard", "FDA-Calibrated"] or v[0]["epochs"] == "10"},
                     "rte", 50, 12, "fig7b_rte_c50_e10", "Epochs E=10 Regime")
plot_benchmark_panel({k: v for k, v in rte50_all.items() if v[0]["method"] in ["FDA-Standard", "FDA-Calibrated"] or v[0]["epochs"] == "20" or v[0]["rank"] == 8},
                     "rte", 50, 12, "fig7c_rte_c50_e20", "Epochs E=20 & Rank r=8 Regime")

# Section 9: SST2
sst20_r50_all = {k: v for k, v in combos[("sst2", 20, 50)].items() if v[0]["method"] != "FDA-SMA (W=3)"}
plot_benchmark_panel(sst20_r50_all, "sst2", 20, 50, "fig8_sst2_c20_r50_e5", "Epochs E=5 Regime")

sst20_all = {k: v for k, v in combos[("sst2", 20, 75)].items() if v[0]["method"] != "FDA-SMA (W=3)"}
plot_benchmark_panel(sst20_all, "sst2", 20, 75, "fig8_sst2_c20_e20", "Local Epochs E=20 Regime")

# Section 10: SST2 C50 R30 (E5, E10, E20)
sst50_all = combos[("sst2", 50, 30)]
plot_benchmark_panel({k: v for k, v in sst50_all.items() if v[0]["method"] in ["FDA-Standard", "FDA-Calibrated"] or v[0]["epochs"] == "5"},
                     "sst2", 50, 30, "fig9a_sst2_c50_e5", "Epochs E=5 Regime")
plot_benchmark_panel({k: v for k, v in sst50_all.items() if v[0]["method"] in ["FDA-Standard", "FDA-Calibrated"] or v[0]["epochs"] == "10"},
                     "sst2", 50, 30, "fig9b_sst2_c50_e10", "Epochs E=10 Regime")
plot_benchmark_panel({k: v for k, v in sst50_all.items() if v[0]["method"] in ["FDA-Standard", "FDA-Calibrated"] or v[0]["epochs"] == "20"},
                     "sst2", 50, 30, "fig9c_sst2_c50_e20", "Epochs E=20 & Rank r=4, 8 Regime")

# Direct comparisons FDA vs RoLoRA
for (ds, c, rounds, _) in combo_order:
    cfgs = {k: v for k, v in combos[(ds, c, rounds)].items() if v[0]["method"] != "FDA-SMA (W=3)"}
    direct_cfgs = {k: v for k, v in cfgs.items() if v[0]["method"] in ["FDA-Standard", "FDA-Calibrated", "RoLoRA"]}
    plot_benchmark_panel(direct_cfgs, ds, c, rounds, f"fig_direct_{ds}_c{c}_r{rounds}", "Direct Comparison: FDA vs. RoLoRA")

# 3. Global Summary Across All Parameter Combinations
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5.5))
b_names = [f"{ds.upper()}\n(C{clients},R{rounds})" for ds, clients, rounds, _ in combo_order]
methods_tracked = ["FDA-Standard", "FDA-Calibrated", "LoRA", "RoLoRA", "FFA-LoRA"]
colors_tracked = {"FDA-Standard": "#1f77b4", "FDA-Calibrated": "#800080", "LoRA": "#ff7f0e", "RoLoRA": "#2ca02c", "FFA-LoRA": "#9467bd"}

best_peaks = {m: [] for m in methods_tracked}
best_times = {m: [] for m in methods_tracked}

for ds, c, rounds, _ in combo_order:
    cfgs = combos[(ds, c, rounds)]
    for m in methods_tracked:
        m_runs = [r_list for base_id, r_list in cfgs.items() if r_list[0]["method"] == m]
        if m_runs:
            best_cfg = max(m_runs, key=lambda rl: np.mean([x["max_acc"] for x in rl]))
            peak_m = np.mean([x["max_acc"] for x in best_cfg])
            time_m = np.mean([x["time_to_best_min"] for x in best_cfg])
            best_peaks[m].append(peak_m)
            best_times[m].append(time_m)
        else:
            best_peaks[m].append(np.nan)
            best_times[m].append(np.nan)

x = np.arange(len(b_names))
width = 0.16

for i, m in enumerate(methods_tracked):
    offset = (i - 2.0) * width
    vals = best_peaks[m]
    ax1.bar(x + offset, vals, width, label=m, color=colors_tracked[m], alpha=0.9, edgecolor='black', linewidth=0.5)

ax1.set_title("Peak Test Accuracy Across Parameter Combinations", fontweight='bold', fontsize=12, pad=10)
ax1.set_ylabel("Peak Accuracy (%)")
ax1.set_xticks(x)
ax1.set_xticklabels(b_names, fontsize=8.5)
ax1.set_ylim(30, 105)
ax1.legend(loc="upper left", framealpha=0.92, edgecolor='#cccccc')

for i, m in enumerate(methods_tracked):
    offset = (i - 2.0) * width
    vals = best_times[m]
    ax2.bar(x + offset, vals, width, label=m, color=colors_tracked[m], alpha=0.9, edgecolor='black', linewidth=0.5)

ax2.set_title("Time to Peak Accuracy Across Parameter Combinations", fontweight='bold', fontsize=12, pad=10)
ax2.set_ylabel("Elapsed Time to Peak (Minutes, Log Scale)")
ax2.set_yscale('log')
ax2.set_xticks(x)
ax2.set_xticklabels(b_names, fontsize=8.5)
ax2.set_ylim(1, 40000)
ax2.legend(loc="upper right", framealpha=0.92, edgecolor='#cccccc')

plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "fig0_cross_benchmark_summary.pdf"), dpi=300, bbox_inches="tight")
plt.savefig(os.path.join(FIG_DIR, "fig0_cross_benchmark_summary.png"), dpi=200, bbox_inches="tight")
plt.close()
print("Generated fig0_cross_benchmark_summary")

# Export communication costs
comm_rows = []
for (ds, c, rounds), cfgs in combos.items():
    for base_id, r_list in cfgs.items():
        r0 = r_list[0]
        m = r0["method"]
        rk = r0["rank"]
        ep = r0["epochs"]
        
        if rk == 4:
            if m == "LoRA":
                uplink_per_round = c * 0.234375
            else:
                uplink_per_round = c * 0.1171875
        elif rk == 8:
            if m == "LoRA":
                uplink_per_round = c * 0.46875
            else:
                uplink_per_round = c * 0.234375
        else:
            uplink_per_round = c * 0.1171875
            
        bidir_per_round = 2.0 * uplink_per_round
        
        completed_rds = np.mean([x["total_rounds"] for x in r_list])
        peak_rd = np.mean([x["best_round"] for x in r_list])
        peak_acc = np.mean([x["max_acc"] for x in r_list])
        
        comm_rows.append({
            "dataset": ds.upper(),
            "clients": c,
            "rounds": rounds,
            "method": m,
            "rank": rk,
            "epochs": ep,
            "mb_per_round_uplink": uplink_per_round,
            "mb_per_round_bidir": bidir_per_round,
            "completed_rounds": completed_rds,
            "total_mb_uplink": completed_rds * uplink_per_round,
            "total_mb_bidir": completed_rds * bidir_per_round,
            "peak_round": peak_rd,
            "peak_mb_uplink": peak_rd * uplink_per_round,
            "peak_mb_bidir": peak_rd * bidir_per_round,
            "peak_acc": peak_acc
        })

df_comm = pd.DataFrame(comm_rows)
df_comm.sort_values(by=["dataset", "clients", "rounds", "method", "rank", "epochs"], inplace=True)
df_comm.to_csv(os.path.join(OUT_DIR, "communication_costs.csv"), index=False)
print("Saved communication_costs.csv")

lr4_e1_datasets = ["qnli", "rte", "sst2"]
lr4_e1_colors = {
    "FDA-Standard": "#1f77b4",
    "LoRA": "#ff7f0e",
    "RoLoRA": "#2ca02c",
}
fig, axes = plt.subplots(2, len(lr4_e1_datasets), figsize=(16, 8))

for column, dataset in enumerate(lr4_e1_datasets):
    dataset_runs = df_lr4_e1[df_lr4_e1["dataset"] == dataset]
    accuracy_ax = axes[0, column]
    time_ax = axes[1, column]

    for method in ["FDA-Standard", "LoRA", "RoLoRA"]:
        method_runs = dataset_runs[dataset_runs["method"] == method].sort_values("lr")
        if method_runs.empty:
            continue

        learning_rates = method_runs["lr"].to_numpy()
        peak_accuracies = method_runs["peak_accuracy"].to_numpy() * 100.0
        peak_times = method_runs["time_to_peak_seconds"].to_numpy() / 60.0
        color = lr4_e1_colors[method]

        accuracy_ax.plot(
            learning_rates,
            peak_accuracies,
            color=color,
            marker="o",
            label=method,
        )
        time_ax.plot(
            learning_rates,
            peak_times,
            color=color,
            marker="o",
            label=method,
        )

    accuracy_ax.set_xscale("log")
    accuracy_ax.set_title(f"{dataset.upper()}: Peak Accuracy", fontweight="bold")
    accuracy_ax.set_xlabel("Client Learning Rate")
    accuracy_ax.set_ylabel("Peak Test Accuracy (%)")
    accuracy_ax.legend(loc="best")

    time_ax.set_xscale("log")
    time_ax.set_yscale("log")
    time_ax.set_title(f"{dataset.upper()}: Time to Peak", fontweight="bold")
    time_ax.set_xlabel("Client Learning Rate")
    time_ax.set_ylabel("Elapsed Time to Peak (Minutes)")
    time_ax.legend(loc="best")

fig.suptitle(
    "THESIS-RBASE-LR4-E1-DS Learning-Rate Sweep (C=20, R=45, S=42)",
    fontweight="bold",
)
plt.tight_layout()
plt.savefig(
    os.path.join(FIG_DIR, "fig_lr4_e1_learning_rate_summary.pdf"),
    dpi=300,
    bbox_inches="tight",
)
plt.savefig(
    os.path.join(FIG_DIR, "fig_lr4_e1_learning_rate_summary.png"),
    dpi=200,
    bbox_inches="tight",
)
plt.close()
print("Generated fig_lr4_e1_learning_rate_summary")

# Calibration figures:
print("Generating calibration figures...")

# Baseline map for S=42 E3TO20 FDA-Standard
baselines_map = {}
baseline_trajectories = {}
for r in parsed_runs:
    if r["method"] == "FDA-Standard" and r["epochs"] == "3TO20" and r["seed"] == 42:
        k = (r["dataset"], r["clients"], r["configured_rounds"])
        baselines_map[k] = {
            "peak_acc": r["peak_accuracy"] * 100.0,
            "final_acc": r["final_accuracy"] * 100.0,
            "mean_epochs": r["df"]["round_epochs"].mean() if "round_epochs" in r["df"].columns else 7.5,
            "total_time": r["total_time_seconds"] / 60.0,
            "time_to_peak": r["time_to_peak_seconds"] / 60.0,
        }
        baseline_trajectories[k] = r["df"]

# Figure 1: Summary Bar Charts (Peak Accuracy & Mean Epochs)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5.5))
settings_ordered = [
    ("mnli", 50, 30, "MNLI\n(C50, R30)"),
    ("mrpc", 20, 30, "MRPC\n(C20, R30)"),
    ("mrpc", 50, 12, "MRPC\n(C50, R12)"),
    ("qnli", 20, 50, "QNLI\n(C20, R50)"),
    ("qnli", 50, 30, "QNLI\n(C50, R30)"),
    ("qqp",  50, 30, "QQP\n(C50, R30)"),
    ("rte",  20, 30, "RTE\n(C20, R30)"),
    ("rte",  50, 12, "RTE\n(C50, R12)"),
    ("sst2", 50, 30, "SST-2\n(C50, R30)"),
]

variants_plot = [
    ("CRATIO0.5 (Base)", "#7f7f7f", "Baseline: $\\kappa=0.5$, No Disc"),
    ("CRATIO0.75-NODISC", "#1f77b4", "$\\kappa=0.75$, NODISC"),
    ("CRATIO0.75-BDISC3", "#17becf", "$\\kappa=0.75$, BDISC3"),
    ("CRATIO1.0-NODISC",  "#ff7f0e", "$\\kappa=1.0$, NODISC"),
    ("CRATIO1.0-BDISC3",  "#d62728", "$\\kappa=1.0$, BDISC3"),
]

x_pos = np.arange(len(settings_ordered))
width = 0.16

for idx, (v_key, color, label) in enumerate(variants_plot):
    pos = x_pos + (idx - 2) * width
    peak_vals = []
    epoch_vals = []
    
    for ds, c, r, _ in settings_ordered:
        if v_key == "CRATIO0.5 (Base)":
            b = baselines_map.get((ds, c, r))
            peak_vals.append(b["peak_acc"] if b else np.nan)
            epoch_vals.append(b["mean_epochs"] if b else np.nan)
        else:
            match = df_calib[(df_calib["dataset"] == ds) & 
                             (df_calib["clients"] == c) & 
                             (df_calib["configured_rounds"] == r) & 
                             (df_calib["variant"] == v_key)]
            if not match.empty:
                peak_vals.append(match["peak_accuracy"].iloc[0])
                epoch_vals.append(match["mean_epochs"].iloc[0])
            else:
                peak_vals.append(np.nan)
                epoch_vals.append(np.nan)
                
    ax1.bar(pos, peak_vals, width, label=label, color=color, alpha=0.88, edgecolor='black', linewidth=0.5)
    ax2.bar(pos, epoch_vals, width, label=label, color=color, alpha=0.88, edgecolor='black', linewidth=0.5)

ax1.set_title("Peak Test Accuracy across Controller Setpoints and Variance Discount", fontweight='bold', fontsize=11.5)
ax1.set_ylabel("Peak Accuracy (%)")
ax1.set_xticks(x_pos)
ax1.set_xticklabels([s[3] for s in settings_ordered], fontsize=9)
ax1.set_ylim(40, 100)
ax1.legend(loc="lower right", framealpha=0.9, edgecolor='#cccccc', fontsize=8.5)

ax2.set_title("Mean Adaptive Local Epochs ($E$) across Settings", fontweight='bold', fontsize=11.5)
ax2.set_ylabel("Mean Local Epochs per Round")
ax2.set_xticks(x_pos)
ax2.set_xticklabels([s[3] for s in settings_ordered], fontsize=9)
ax2.set_ylim(0, 22)
ax2.axhline(20, color='red', linestyle=':', alpha=0.7, label="$E_{\\max}=20$ Cap")
ax2.axhline(10, color='gray', linestyle=':', alpha=0.7, label="$E_{\\max}/2=10$ Midpoint")
ax2.legend(loc="upper left", framealpha=0.9, edgecolor='#cccccc', fontsize=8.5)

plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "fig_calib_accuracy_epochs_summary.pdf"), dpi=300, bbox_inches="tight")
plt.savefig(os.path.join(FIG_DIR, "fig_calib_accuracy_epochs_summary.png"), dpi=200, bbox_inches="tight")
plt.close()
print("Generated fig_calib_accuracy_epochs_summary")

# Figure 2: MNLI & MRPC Trajectories
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

mnli_base = baseline_trajectories.get(("mnli", 50, 30))
if mnli_base is not None:
    ax1.plot(mnli_base["round"], mnli_base["accuracy"] * 100.0, label="Baseline ($\\kappa=0.5$, NODISC)", color="#7f7f7f", linestyle=":", marker="x", markevery=3)
    ax2.plot(mnli_base["time_round_total"].cumsum() / 60.0, mnli_base["accuracy"] * 100.0, label="Baseline ($\\kappa=0.5$, NODISC)", color="#7f7f7f", linestyle=":", marker="x", markevery=3)

colors_calib = {
    "CRATIO0.75-NODISC": ("#1f77b4", "-", "o"),
    "CRATIO0.75-BDISC3": ("#17becf", "--", "s"),
    "CRATIO1-NODISC":    ("#ff7f0e", "-.", "^"),
    "CRATIO1-BDISC3":    ("#d62728", "-", "D"),
}

mnli_calib_files = sorted(glob.glob("results/round_metrics/*THESIS-RBASE-CALIB-DS-mnli-C50-R30*.csv"))
for f in mnli_calib_files:
    df_c = pd.read_csv(f)
    if "accuracy_m" in df_c.columns:
        df_c["accuracy"] = (df_c["accuracy_m"] + df_c["accuracy_mm"]) / 2.0
    for v_str, (col, ls, mk) in colors_calib.items():
        if v_str in f:
            v_label = v_str.replace("CRATIO1-", "CRATIO1.0-")
            ax1.plot(df_c["round"], df_c["accuracy"] * 100.0, label=f"Calib: {v_label}", color=col, linestyle=ls, marker=mk, markevery=3)
            ax2.plot(df_c["time_round_total"].cumsum() / 60.0, df_c["accuracy"] * 100.0, label=f"Calib: {v_label}", color=col, linestyle=ls, marker=mk, markevery=3)

ax1.set_title("MNLI (C=50, R=30): Accuracy vs. Round", fontweight='bold')
ax1.set_xlabel("Communication Round")
ax1.set_ylabel("Test Accuracy (%)")
ax1.set_xlim(1, 30)
ax1.legend(loc="lower right", framealpha=0.9, fontsize=8.5)

ax2.set_title("MNLI (C=50, R=30): Accuracy vs. Wall-Clock Time", fontweight='bold')
ax2.set_xlabel("Elapsed Wall-Clock Time (Minutes)")
ax2.set_ylabel("Test Accuracy (%)")
ax2.legend(loc="lower right", framealpha=0.9, fontsize=8.5)

mrpc_base = baseline_trajectories.get(("mrpc", 20, 30))
if mrpc_base is not None:
    ax3.plot(mrpc_base["round"], mrpc_base["accuracy"] * 100.0, label="Baseline ($\\kappa=0.5$, NODISC)", color="#7f7f7f", linestyle=":", marker="x", markevery=3)
    ax4.plot(mrpc_base["time_round_total"].cumsum() / 60.0, mrpc_base["accuracy"] * 100.0, label="Baseline ($\\kappa=0.5$, NODISC)", color="#7f7f7f", linestyle=":", marker="x", markevery=3)

mrpc_calib_files = sorted(glob.glob("results/round_metrics/*THESIS-RBASE-CALIB-DS-mrpc-C20-R30*.csv"))
for f in mrpc_calib_files:
    df_c = pd.read_csv(f)
    for v_str, (col, ls, mk) in colors_calib.items():
        if v_str in f:
            v_label = v_str.replace("CRATIO1-", "CRATIO1.0-")
            ax3.plot(df_c["round"], df_c["accuracy"] * 100.0, label=f"Calib: {v_label}", color=col, linestyle=ls, marker=mk, markevery=3)
            ax4.plot(df_c["time_round_total"].cumsum() / 60.0, df_c["accuracy"] * 100.0, label=f"Calib: {v_label}", color=col, linestyle=ls, marker=mk, markevery=3)

ax3.set_title("MRPC (C=20, R=30): Accuracy vs. Round", fontweight='bold')
ax3.set_xlabel("Communication Round")
ax3.set_ylabel("Test Accuracy (%)")
ax3.set_xlim(1, 30)
ax3.legend(loc="lower right", framealpha=0.9, fontsize=8.5)

ax4.set_title("MRPC (C=20, R=30): Accuracy vs. Wall-Clock Time", fontweight='bold')
ax4.set_xlabel("Elapsed Wall-Clock Time (Minutes)")
ax4.set_ylabel("Test Accuracy (%)")
ax4.legend(loc="lower right", framealpha=0.9, fontsize=8.5)

plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "fig_calib_trajectories_mnli_mrpc.pdf"), dpi=300, bbox_inches="tight")
plt.savefig(os.path.join(FIG_DIR, "fig_calib_trajectories_mnli_mrpc.png"), dpi=200, bbox_inches="tight")
plt.close()
print("Generated fig_calib_trajectories_mnli_mrpc")

# Figure 3: QNLI & SST-2 Trajectories
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

qnli_base = baseline_trajectories.get(("qnli", 50, 30))
if qnli_base is not None:
    ax1.plot(qnli_base["round"], qnli_base["accuracy"] * 100.0, label="Baseline ($\\kappa=0.5$, NODISC)", color="#7f7f7f", linestyle=":", marker="x", markevery=3)
    ax2.plot(qnli_base["time_round_total"].cumsum() / 60.0, qnli_base["accuracy"] * 100.0, label="Baseline ($\\kappa=0.5$, NODISC)", color="#7f7f7f", linestyle=":", marker="x", markevery=3)

qnli_calib_files = sorted(glob.glob("results/round_metrics/*THESIS-RBASE-CALIB-DS-qnli-C50-R30*.csv"))
for f in qnli_calib_files:
    df_c = pd.read_csv(f)
    for v_str, (col, ls, mk) in colors_calib.items():
        if v_str in f:
            v_label = v_str.replace("CRATIO1-", "CRATIO1.0-")
            ax1.plot(df_c["round"], df_c["accuracy"] * 100.0, label=f"Calib: {v_label}", color=col, linestyle=ls, marker=mk, markevery=3)
            ax2.plot(df_c["time_round_total"].cumsum() / 60.0, df_c["accuracy"] * 100.0, label=f"Calib: {v_label}", color=col, linestyle=ls, marker=mk, markevery=3)

ax1.set_title("QNLI (C=50, R=30): Accuracy vs. Round", fontweight='bold')
ax1.set_xlabel("Communication Round")
ax1.set_ylabel("Test Accuracy (%)")
ax1.set_xlim(1, 30)
ax1.legend(loc="lower right", framealpha=0.9, fontsize=8.5)

ax2.set_title("QNLI (C=50, R=30): Accuracy vs. Wall-Clock Time", fontweight='bold')
ax2.set_xlabel("Elapsed Wall-Clock Time (Minutes)")
ax2.set_ylabel("Test Accuracy (%)")
ax2.legend(loc="lower right", framealpha=0.9, fontsize=8.5)

sst2_base = baseline_trajectories.get(("sst2", 50, 30))
if sst2_base is not None:
    ax3.plot(sst2_base["round"], sst2_base["accuracy"] * 100.0, label="Baseline ($\\kappa=0.5$, NODISC)", color="#7f7f7f", linestyle=":", marker="x", markevery=3)
    ax4.plot(sst2_base["time_round_total"].cumsum() / 60.0, sst2_base["accuracy"] * 100.0, label="Baseline ($\\kappa=0.5$, NODISC)", color="#7f7f7f", linestyle=":", marker="x", markevery=3)

sst2_calib_files = sorted(glob.glob("results/round_metrics/*THESIS-RBASE-CALIB-DS-sst2-C50-R30*.csv"))
for f in sst2_calib_files:
    df_c = pd.read_csv(f)
    for v_str, (col, ls, mk) in colors_calib.items():
        if v_str in f:
            v_label = v_str.replace("CRATIO1-", "CRATIO1.0-")
            ax3.plot(df_c["round"], df_c["accuracy"] * 100.0, label=f"Calib: {v_label}", color=col, linestyle=ls, marker=mk, markevery=3)
            ax4.plot(df_c["time_round_total"].cumsum() / 60.0, df_c["accuracy"] * 100.0, label=f"Calib: {v_label}", color=col, linestyle=ls, marker=mk, markevery=3)

ax3.set_title("SST-2 (C=50, R=30): Accuracy vs. Round", fontweight='bold')
ax3.set_xlabel("Communication Round")
ax3.set_ylabel("Test Accuracy (%)")
ax3.set_xlim(1, 30)
ax3.legend(loc="lower right", framealpha=0.9, fontsize=8.5)

ax4.set_title("SST-2 (C=50, R=30): Accuracy vs. Wall-Clock Time", fontweight='bold')
ax4.set_xlabel("Elapsed Wall-Clock Time (Minutes)")
ax4.set_ylabel("Test Accuracy (%)")
ax4.legend(loc="lower right", framealpha=0.9, fontsize=8.5)

plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "fig_calib_trajectories_qnli_sst2.pdf"), dpi=300, bbox_inches="tight")
plt.savefig(os.path.join(FIG_DIR, "fig_calib_trajectories_qnli_sst2.png"), dpi=200, bbox_inches="tight")
plt.close()
print("Generated fig_calib_trajectories_qnli_sst2")

print("ALL FIGURES GENERATED SUCCESSFULLY!")

