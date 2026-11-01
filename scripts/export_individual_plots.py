import os
import glob
import re
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# All 16 configurations across GLUE suite
all_configs = [
    # MNLI
    ("01_MNLI_C20_R50_LR0.1",  "MNLI", "mnli", 20, 50, "0.1"),
    ("02_MNLI_C20_R50_LR0.01", "MNLI (lr=0.01)", "mnli", 20, 50, "0.01"),
    ("03_MNLI_C50_R30_LR0.1",  "MNLI", "mnli", 50, 30, "0.1"),
    ("04_MNLI_C50_R30_LR0.01", "MNLI (lr=0.01)", "mnli", 50, 30, "0.01"),
    # QNLI
    ("05_QNLI_C20_R50_LR0.1",  "QNLI", "qnli", 20, 50, "0.1"),
    ("06_QNLI_C20_R50_LR0.01", "QNLI (lr=0.01)", "qnli", 20, 50, "0.01"),
    ("07_QNLI_C50_R30_LR0.1",  "QNLI", "qnli", 50, 30, "0.1"),
    ("08_QNLI_C50_R30_LR0.01", "QNLI (lr=0.01)", "qnli", 50, 30, "0.01"),
    # QQP
    ("09_QQP_C20_R50_LR0.1",   "QQP",  "qqp",  20, 50, "0.1"),
    ("10_QQP_C50_R30_LR0.1",   "QQP",  "qqp",  50, 30, "0.1"),
    # SST-2
    ("11_SST2_C20_R50_LR0.1",  "SST-2", "sst2", 20, 50, "0.1"),
    ("12_SST2_C50_R30_LR0.1",  "SST-2", "sst2", 50, 30, "0.1"),
    # MRPC
    ("13_MRPC_C20_R30_LR0.05", "MRPC", "mrpc", 20, 30, "0.05"),
    ("14_MRPC_C50_R12_LR0.05", "MRPC", "mrpc", 50, 12, "0.05"),
    # RTE
    ("15_RTE_C20_R30_LR0.05",  "RTE",  "rte",  20, 30, "0.05"),
    ("16_RTE_C50_R12_LR0.05",  "RTE",  "rte",  50, 12, "0.05"),
]

# Color schemes exactly matching telemetry
c_fda20_acc = "#1f77b4"
c_fda20_var = "#1d4ed8"
c_fda30_acc = "#7b1fa2"
c_fda30_var = "#6b21a8"
c_fda40_acc = "#be185d"
c_fda40_var = "#9d174d"

c_fda20_odd  = "#1e40af"
c_fda20_even = "#60a5fa"
c_fda30_odd  = "#581c87"
c_fda30_even = "#c084fc"
c_fda40_odd  = "#9d174d"
c_fda40_even = "#f472b6"

c_ro20_acc   = "#9a3412"
c_ro20_rk8_acc = "#c2410c"
c_ro20_ep    = "#9a3412"
c_ro15_acc   = "#ea580c"
c_ro15_ep    = "#ea580c"
c_ro15_var   = "#c2410c"
c_ro10_acc  = "#ff7f0e"
c_ro10_ep   = "#ea580c"
c_ro10_var  = "#c2410c"
c_ro5_acc   = "#d62728"
c_ro5_ep    = "#b91c1c"
c_ro5_var   = "#991b1b"

c_lora10_acc = "#2ca02c"
c_lora10_ep  = "#16a34a"
c_lora10_var = "#15803d"
c_lora5_acc  = "#059669"
c_lora5_ep   = "#047857"
c_lora5_var  = "#065f46"

c_lora20_acc = "#15803d"
c_lora20_rk8_acc = "#166534"
c_lora20_ep  = "#15803d"
c_lora20_var = "#14532d"

# FFA-LoRA palette
c_ffa10_acc = "#0891b2"
c_ffa10_ep  = "#0e7490"
c_ffa10_var = "#155e75"
c_ffa5_acc  = "#06b6d4"
c_ffa5_ep   = "#0891b2"
c_ffa5_var  = "#164e63"

# Communication cost per round (bi-directional complete payload, Float32)
# C=20: FDA/RoLoRA = 4.6875 MB, LoRA/FFA = 9.375 MB
# C=50: FDA/RoLoRA = 11.71875 MB, LoRA/FFA = 23.4375 MB
comm_rates = {
    (20, False): 4.6875,
    (20, True): 9.375,
    (50, False): 11.71875,
    (50, True): 23.4375,
}

def prep(df):
    if "round" in df.columns:
        df = df.sort_values("round").reset_index(drop=True)
    if "accuracy" not in df.columns and {"accuracy_m", "accuracy_mm"}.issubset(df.columns):
        df["accuracy"] = (df["accuracy_m"] + df["accuracy_mm"]) / 2.0
    return df

def average_runs(dfs):
    if len(dfs) == 1:
        return dfs[0]
    max_r = max(len(df) for df in dfs)
    records = []
    for r_idx in range(max_r):
        rnd = r_idx + 1
        active_dfs = [df for df in dfs if len(df) >= rnd]
        act_mat = "A" if rnd % 2 == 1 else "B"
        
        accs = []
        for df in dfs:
            if len(df) >= rnd:
                accs.append(df["accuracy"].iloc[r_idx])
            else:
                accs.append(df["accuracy"].iloc[-1])
        acc = float(np.mean(accs))

        train_accs = []
        for df in dfs:
            if "training_accuracy" in df.columns:
                if len(df) >= rnd:
                    train_accs.append(df["training_accuracy"].iloc[r_idx])
                else:
                    train_accs.append(df["training_accuracy"].iloc[-1])
        train_acc = float(np.mean(train_accs)) if train_accs else None
        
        ep = float(np.mean([df["round_epochs"].iloc[r_idx] for df in active_dfs]))
        var = float(np.mean([df["variance"].iloc[r_idx] for df in active_dfs]))
        
        row = {
            "round": rnd,
            "active_matrix": act_mat,
            "accuracy": acc,
            "round_epochs": ep,
            "variance": var,
        }
        if train_acc is not None:
            row["training_accuracy"] = train_acc
        thetas = [df["theta_phase"].iloc[r_idx] for df in active_dfs if "theta_phase" in df.columns]
        if thetas:
            row["theta_phase"] = float(np.mean(thetas))
        records.append(row)
    return pd.DataFrame(records)

def load_fda_csv(ds, c, r, e_str, lr):
    pats = [
        f"results/round_metrics/*THESIS-RBASE-MAIN-DS-{ds}-C{c}-*-M-fda-k1-LR{lr}-RK4-{e_str}-S*-*.csv",
        f"results/round_metrics/*CALIB-DS-{ds}-C{c}-*-M-fda-calib-LR{lr}*{e_str}*CRATIO1*BDISC3*.csv",
    ]
    matches = []
    for p in pats:
        matches.extend(glob.glob(p))
    matches = sorted(list(set(matches)))
    if matches:
        dfs = [prep(pd.read_csv(f)) for f in matches]
        return average_runs(dfs)
    return None

def load_baseline_csv(ds, c, r, method, e_str, lr):
    pat = f"results/round_metrics/*MAIN-DS-{ds}-C{c}-*-M-{method}-LR{lr}-RK4-S*-{e_str}*.csv"
    matches = sorted(glob.glob(pat))
    if matches:
        dfs = [prep(pd.read_csv(f)) for f in matches]
        return average_runs(dfs)
    return None

def load_baseline_e20_csv(ds, c, r, method, lr, rk=4):
    pat1 = f"results/round_metrics/*MAIN-DS-{ds}-C{c}-*-M-{method}-LR{lr}-RK{rk}-S*-E20*.csv"
    pat2 = f"results/round_metrics/*MAIN-DS-{ds}-C{c}-*-M-{method}-LR{lr}-RK{rk}-S*_[0-9a-f]*.csv"
    matches = sorted(list(set(glob.glob(pat1) + glob.glob(pat2))))
    valid = []
    for m in matches:
        if "-E5-" in m or "-E10-" in m or "-E15-" in m:
            continue
        try:
            df = pd.read_csv(m, nrows=2)
            ep = df.get("epochs", df.get("round_epochs", [None]))[0]
            if ep == 20:
                valid.append(m)
        except Exception:
            pass
    if valid:
        dfs = [prep(pd.read_csv(f)) for f in valid]
        return average_runs(dfs)
    return None

def load_old_fda_csv(ds, c, r, e_str, lr, rk=4):
    pats = [
        f"results/round_metrics/*THESIS-RBASE-MAIN-DS-{ds}-C{c}-*-M-fda-k05-LR{lr}-RK{rk}-{e_str}-S*-*.csv",
        f"results/round_metrics/*THESIS-RBASE-MAIN-DS-{ds}-C{c}-*-M-fda-standard-LR{lr}-RK{rk}-{e_str}-S*-*.csv",
    ]
    matches = []
    for p in pats:
        matches.extend(glob.glob(p))
    matches = sorted(list(set(matches)))
    if matches:
        dfs = [prep(pd.read_csv(f)) for f in matches]
        return average_runs(dfs)
    return None

def get_old_fda_runs(ds, c, r, lr):
    runs_found = []
    # E3TO10 - Teal / Seafoam
    df_10 = load_old_fda_csv(ds, c, r, "E3TO10", lr, 4)
    if df_10 is not None:
        runs_found.append({
            "label": "FDA k=0.5 E3TO10",
            "df": df_10,
            "c_acc": "#0d9488",
            "c_odd": "#0f766e",
            "c_even": "#5eead4",
            "marker": "o",
            "ls": "-."
        })
    # E3TO20 - Amber / Gold
    df_20 = load_old_fda_csv(ds, c, r, "E3TO20", lr, 4)
    if df_20 is not None:
        runs_found.append({
            "label": "FDA k=0.5 E3TO20",
            "df": df_20,
            "c_acc": "#d97706",
            "c_odd": "#b45309",
            "c_even": "#fcd34d",
            "marker": "p",
            "ls": "-."
        })
    # E3TO30 - Slate / Charcoal
    df_30 = load_old_fda_csv(ds, c, r, "E3TO30", lr, 4)
    if df_30 is not None:
        runs_found.append({
            "label": "FDA k=0.5 E3TO30",
            "df": df_30,
            "c_acc": "#475569",
            "c_odd": "#334155",
            "c_even": "#94a3b8",
            "marker": "h",
            "ls": "-."
        })
    # E3TO40 - Olive / Moss
    df_40 = load_old_fda_csv(ds, c, r, "E3TO40", lr, 4)
    if df_40 is not None:
        runs_found.append({
            "label": "FDA k=0.5 E3TO40",
            "df": df_40,
            "c_acc": "#65a30d",
            "c_odd": "#3f6212",
            "c_even": "#bef264",
            "marker": "s",
            "ls": "-."
        })
    # RK8 E3TO30 - Crimson / Coral
    df_rk8 = load_old_fda_csv(ds, c, r, "E3TO30", lr, 8)
    if df_rk8 is not None:
        runs_found.append({
            "label": "FDA k=0.5 RK8",
            "df": df_rk8,
            "c_acc": "#dc2626",
            "c_odd": "#991b1b",
            "c_even": "#fca5a5",
            "marker": "P",
            "ls": "-."
        })
    return runs_found

def render_config_plots(cfg_prefix, display_name, ds, c, r, lr, include_lora, out_dir):
    xlim = (1, 12) if r == 12 else (0, r)
    
    comparison_tag = "All Methods" if include_lora else "FDA vs RoLoRA"

    df_csv_e20 = load_fda_csv(ds, c, r, "E3TO20", lr)
    df_csv_e30 = load_fda_csv(ds, c, r, "E3TO30", lr)
    df_csv_e40 = load_fda_csv(ds, c, r, "E3TO40", lr)
    old_runs = get_old_fda_runs(ds, c, r, lr)

    df_csv_ro20 = load_baseline_e20_csv(ds, c, r, "rolora", lr, 4)
    df_csv_ro20_rk8 = load_baseline_e20_csv(ds, c, r, "rolora", lr, 8)
    df_csv_ro15 = load_baseline_csv(ds, c, r, "rolora", "E15", lr)
    df_csv_ro10 = load_baseline_csv(ds, c, r, "rolora", "E10", lr)
    df_csv_ro5 = load_baseline_csv(ds, c, r, "rolora", "E5", lr)

    df_csv_lo20 = load_baseline_e20_csv(ds, c, r, "lora", lr, 4) if include_lora else None
    df_csv_lo20_rk8 = load_baseline_e20_csv(ds, c, r, "lora", lr, 8) if include_lora else None
    df_csv_lo10 = load_baseline_csv(ds, c, r, "lora", "E10", lr) if include_lora else None
    df_csv_lo5 = load_baseline_csv(ds, c, r, "lora", "E5", lr) if include_lora else None

    df_csv_ffa10 = load_baseline_csv(ds, c, r, "ffa", "E10", lr) if include_lora else None
    df_csv_ffa5 = load_baseline_csv(ds, c, r, "ffa", "E5", lr) if include_lora else None

    rate_fda_ro = comm_rates[(c, False)]
    rate_lora = comm_rates[(c, True)]

    # -------------------------------------------------------------
    # 1. Helper: draw accuracy vs round
    # -------------------------------------------------------------
    def draw_acc(ax, is_standalone=False):
        ax.set_title(f"{display_name} (Clients={c}, Rounds={r}, lr={lr}, α=1000) — Validation Accuracy ({comparison_tag})", 
                     fontsize=11.5 if is_standalone else 10.0, fontweight="bold")
        ax.set_ylabel("Accuracy", fontsize=9.5)
        ax.set_xlabel("Communication Round", fontsize=9.5)
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.set_xlim(xlim)

        # FDA E3TO20
        if df_csv_e20 is not None:
            acc = df_csv_e20["accuracy"]
            best_rnd = int(df_csv_e20.loc[acc.idxmax(), "round"])
            peak_acc = acc.max()
            lbl_acc = f"FDA E3TO20 ({peak_acc*100:.1f}%, R{best_rnd})"
            ax.plot(df_csv_e20["round"], acc, label=lbl_acc, color=c_fda20_acc, linestyle="-", marker="o", markersize=4.5, linewidth=2.0, zorder=4)
            ax.scatter(best_rnd, peak_acc, marker="*", s=160, color="#0047AB", edgecolors="black", linewidth=0.5, zorder=7)

        # FDA E3TO30
        if df_csv_e30 is not None:
            acc = df_csv_e30["accuracy"]
            best_rnd = int(df_csv_e30.loc[acc.idxmax(), "round"])
            peak_acc = acc.max()
            lbl_acc = f"FDA E3TO30 ({peak_acc*100:.1f}%, R{best_rnd})"
            ax.plot(df_csv_e30["round"], acc, label=lbl_acc, color=c_fda30_acc, linestyle="-", marker="^", markersize=5.0, linewidth=2.2, zorder=5)
            ax.scatter(best_rnd, peak_acc, marker="*", s=180, color="#4a148c", edgecolors="gold", linewidth=0.8, zorder=8)

        # FDA E3TO40
        if df_csv_e40 is not None:
            acc = df_csv_e40["accuracy"]
            best_rnd = int(df_csv_e40.loc[acc.idxmax(), "round"])
            peak_acc = acc.max()
            lbl_acc = f"FDA E3TO40 ({peak_acc*100:.1f}%, R{best_rnd})"
            ax.plot(df_csv_e40["round"], acc, label=lbl_acc, color=c_fda40_acc, linestyle="-", marker="D", markersize=4.8, linewidth=2.0, zorder=5)
            ax.scatter(best_rnd, peak_acc, marker="*", s=170, color=c_fda40_odd, edgecolors="black", linewidth=0.5, zorder=8)

        # Old FDA runs
        for orun in old_runs:
            df_old = orun["df"]
            acc = df_old["accuracy"]
            best_rnd = int(df_old.loc[acc.idxmax(), "round"])
            peak_acc = acc.max()
            lbl_acc = f"{orun['label']} ({peak_acc*100:.1f}%, R{best_rnd})"
            ax.plot(df_old["round"], acc, label=lbl_acc, color=orun["c_acc"], linestyle=orun["ls"], marker=orun["marker"], markersize=4.2, linewidth=1.8, zorder=4)
            ax.scatter(best_rnd, peak_acc, marker="*", s=150, color=orun["c_odd"], edgecolors="black", linewidth=0.5, zorder=7)

        # RoLoRA
        if df_csv_ro20 is not None:
            acc = df_csv_ro20["accuracy"]
            best_rnd = int(df_csv_ro20.loc[acc.idxmax(), "round"])
            peak_acc = acc.max()
            lbl_acc = f"RoLoRA E20 ({peak_acc*100:.1f}%, R{best_rnd})"
            ax.plot(df_csv_ro20["round"], acc, label=lbl_acc, color=c_ro20_acc, linestyle="-", marker="v", markersize=4.2, linewidth=1.8, zorder=3)
            ax.scatter(best_rnd, peak_acc, marker="*", s=150, color="#78350f", edgecolors="black", linewidth=0.5, zorder=6)

        if df_csv_ro20_rk8 is not None:
            acc = df_csv_ro20_rk8["accuracy"]
            best_rnd = int(df_csv_ro20_rk8.loc[acc.idxmax(), "round"])
            peak_acc = acc.max()
            lbl_acc = f"RoLoRA E20 RK8 ({peak_acc*100:.1f}%, R{best_rnd})"
            ax.plot(df_csv_ro20_rk8["round"], acc, label=lbl_acc, color=c_ro20_rk8_acc, linestyle="--", marker="v", markersize=3.8, linewidth=1.5, zorder=3)

        if df_csv_ro15 is not None:
            acc = df_csv_ro15["accuracy"]
            best_rnd = int(df_csv_ro15.loc[acc.idxmax(), "round"])
            peak_acc = acc.max()
            lbl_acc = f"RoLoRA E15 ({peak_acc*100:.1f}%, R{best_rnd})"
            ax.plot(df_csv_ro15["round"], acc, label=lbl_acc, color=c_ro15_acc, linestyle="-", marker=">", markersize=4.2, linewidth=1.8, zorder=3)
            ax.scatter(best_rnd, peak_acc, marker="*", s=150, color="#9a3412", edgecolors="black", linewidth=0.5, zorder=6)

        if df_csv_ro10 is not None:
            acc = df_csv_ro10["accuracy"]
            best_rnd = int(df_csv_ro10.loc[acc.idxmax(), "round"])
            peak_acc = acc.max()
            lbl_acc = f"RoLoRA E10 ({peak_acc*100:.1f}%, R{best_rnd})"
            ax.plot(df_csv_ro10["round"], acc, label=lbl_acc, color=c_ro10_acc, linestyle="-", marker="s", markersize=4.2, linewidth=1.8, zorder=3)
            ax.scatter(best_rnd, peak_acc, marker="*", s=150, color="#b23b00", edgecolors="black", linewidth=0.5, zorder=6)

        if df_csv_ro5 is not None:
            acc = df_csv_ro5["accuracy"]
            best_rnd = int(df_csv_ro5.loc[acc.idxmax(), "round"])
            peak_acc = acc.max()
            lbl_acc = f"RoLoRA E5 ({peak_acc*100:.1f}%, R{best_rnd})"
            ax.plot(df_csv_ro5["round"], acc, label=lbl_acc, color=c_ro5_acc, linestyle="--", marker="x", markersize=4.5, linewidth=1.6, zorder=2)

        # LoRA & FFA-LoRA
        if include_lora:
            if df_csv_lo20 is not None:
                acc = df_csv_lo20["accuracy"]
                best_rnd = int(df_csv_lo20.loc[acc.idxmax(), "round"])
                peak_acc = acc.max()
                lbl_acc = f"LoRA E20 ({peak_acc*100:.1f}%, R{best_rnd})"
                ax.plot(df_csv_lo20["round"], acc, label=lbl_acc, color=c_lora20_acc, linestyle="-", marker="<", markersize=4.0, linewidth=1.8, zorder=3)

            if df_csv_lo20_rk8 is not None:
                acc = df_csv_lo20_rk8["accuracy"]
                best_rnd = int(df_csv_lo20_rk8.loc[acc.idxmax(), "round"])
                peak_acc = acc.max()
                lbl_acc = f"LoRA E20 RK8 ({peak_acc*100:.1f}%, R{best_rnd})"
                ax.plot(df_csv_lo20_rk8["round"], acc, label=lbl_acc, color=c_lora20_rk8_acc, linestyle="--", marker="<", markersize=3.6, linewidth=1.4, zorder=2)

            if df_csv_lo10 is not None:
                acc = df_csv_lo10["accuracy"]
                best_rnd = int(df_csv_lo10.loc[acc.idxmax(), "round"])
                peak_acc = acc.max()
                lbl_acc = f"LoRA E10 ({peak_acc*100:.1f}%, R{best_rnd})"
                ax.plot(df_csv_lo10["round"], acc, label=lbl_acc, color=c_lora10_acc, linestyle="-", marker="D", markersize=4.0, linewidth=1.8, zorder=3)

            if df_csv_lo5 is not None:
                acc = df_csv_lo5["accuracy"]
                best_rnd = int(df_csv_lo5.loc[acc.idxmax(), "round"])
                peak_acc = acc.max()
                lbl_acc = f"LoRA E5 ({peak_acc*100:.1f}%, R{best_rnd})"
                ax.plot(df_csv_lo5["round"], acc, label=lbl_acc, color=c_lora5_acc, linestyle="--", marker="v", markersize=4.2, linewidth=1.6, zorder=2)

            if df_csv_ffa10 is not None:
                acc = df_csv_ffa10["accuracy"]
                best_rnd = int(df_csv_ffa10.loc[acc.idxmax(), "round"])
                peak_acc = acc.max()
                lbl_acc = f"FFA-LoRA E10 ({peak_acc*100:.1f}%, R{best_rnd})"
                ax.plot(df_csv_ffa10["round"], acc, label=lbl_acc, color=c_ffa10_acc, linestyle="-", marker="h", markersize=4.2, linewidth=1.8, zorder=3)
                ax.scatter(best_rnd, peak_acc, marker="*", s=140, color="#155e75", edgecolors="black", linewidth=0.5, zorder=6)

            if df_csv_ffa5 is not None:
                acc = df_csv_ffa5["accuracy"]
                best_rnd = int(df_csv_ffa5.loc[acc.idxmax(), "round"])
                peak_acc = acc.max()
                lbl_acc = f"FFA-LoRA E5 ({peak_acc*100:.1f}%, R{best_rnd})"
                ax.plot(df_csv_ffa5["round"], acc, label=lbl_acc, color=c_ffa5_acc, linestyle="--", marker="p", markersize=4.0, linewidth=1.5, zorder=2)

        ax.legend(fontsize=6.8 if is_standalone else 6.2, loc="lower right", framealpha=0.88, facecolor="white", edgecolor="#cccccc")

    # -------------------------------------------------------------
    # 2. Helper: draw allocated epochs vs round
    # -------------------------------------------------------------
    def draw_ep(ax, is_standalone=False):
        ax.set_title(f"{display_name} (Clients={c}, Rounds={r}, lr={lr}, α=1000) — Allocated Local Epochs ({comparison_tag})", 
                     fontsize=11.5 if is_standalone else 10.0, fontweight="bold", color="#333333")
        ax.set_ylabel("Local Epochs", fontsize=9.5)
        ax.set_xlabel("Communication Round", fontsize=9.5)
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.set_xlim(xlim)

        fda_runs_bars = []
        if df_csv_e20 is not None:
            fda_runs_bars.append(("FDA E3TO20", df_csv_e20, c_fda20_odd, c_fda20_even))
        if df_csv_e30 is not None:
            fda_runs_bars.append(("FDA E3TO30", df_csv_e30, c_fda30_odd, c_fda30_even))
        if df_csv_e40 is not None:
            fda_runs_bars.append(("FDA E3TO40", df_csv_e40, c_fda40_odd, c_fda40_even))
        for orun in old_runs:
            fda_runs_bars.append((orun["label"], orun["df"], orun["c_odd"], orun["c_even"]))

        num_fda = len(fda_runs_bars)
        if num_fda > 0:
            w = 0.8 / num_fda
            offsets = np.linspace(-0.4 + w/2, 0.4 - w/2, num_fda)
            for idx, (label, df_run, c_odd, c_even) in enumerate(fda_runs_bars):
                rounds = df_run["round"].values
                eps = df_run["round_epochs"].values
                colors = [c_odd if rn % 2 == 1 else c_even for rn in rounds]
                ax.bar(rounds + offsets[idx], eps, width=w*0.92, color=colors, edgecolor="black", linewidth=0.35, label=f"{label} (A=Dark, B=Light)", zorder=3)

        if df_csv_ro20 is not None:
            ax.plot(df_csv_ro20["round"], [20] * len(df_csv_ro20), label="RoLoRA E20", color=c_ro20_ep, linestyle="-", linewidth=1.6, zorder=4)
        if df_csv_ro15 is not None:
            ax.plot(df_csv_ro15["round"], [15] * len(df_csv_ro15), label="RoLoRA E15", color=c_ro15_ep, linestyle="-", linewidth=1.6, zorder=4)
        if df_csv_ro10 is not None:
            ax.plot(df_csv_ro10["round"], [10] * len(df_csv_ro10), label="RoLoRA E10", color=c_ro10_ep, linestyle="--", linewidth=1.6, zorder=4)
        if df_csv_ro5 is not None:
            ax.plot(df_csv_ro5["round"], [5] * len(df_csv_ro5), label="RoLoRA E5", color=c_ro5_ep, linestyle=":", linewidth=1.6, zorder=4)

        if include_lora:
            if df_csv_lo20 is not None:
                ax.plot(df_csv_lo20["round"], [20.3] * len(df_csv_lo20), label="LoRA E20", color=c_lora20_ep, linestyle="-.", linewidth=1.6, zorder=4)
            if df_csv_lo10 is not None:
                ax.plot(df_csv_lo10["round"], [10.2] * len(df_csv_lo10), label="LoRA E10", color=c_lora10_ep, linestyle="-.", linewidth=1.6, zorder=4)
            if df_csv_lo5 is not None:
                ax.plot(df_csv_lo5["round"], [5.2] * len(df_csv_lo5), label="LoRA E5", color=c_lora5_ep, linestyle="-.", linewidth=1.6, zorder=4)
            if df_csv_ffa10 is not None:
                ax.plot(df_csv_ffa10["round"], [10.4] * len(df_csv_ffa10), label="FFA-LoRA E10", color=c_ffa10_ep, linestyle=":", linewidth=1.6, zorder=4)
            if df_csv_ffa5 is not None:
                ax.plot(df_csv_ffa5["round"], [5.4] * len(df_csv_ffa5), label="FFA-LoRA E5", color=c_ffa5_ep, linestyle=":", linewidth=1.6, zorder=4)

        h_ep, l_ep = ax.get_legend_handles_labels()
        fda_idx = [i for i, l in enumerate(l_ep) if "FDA" in l]
        other_idx = [i for i, l in enumerate(l_ep) if "FDA" not in l]
        ord_idx = fda_idx + other_idx
        ax.legend([h_ep[i] for i in ord_idx], [l_ep[i] for i in ord_idx],
                  fontsize=6.8 if is_standalone else 6.0, loc="upper right", framealpha=0.88, facecolor="white", edgecolor="#cccccc")

    # -------------------------------------------------------------
    # 3. Helper: draw client drift variance vs round
    # -------------------------------------------------------------
    def draw_var(ax, is_standalone=False):
        ax.set_title(f"{display_name} (Clients={c}, Rounds={r}, lr={lr}, α=1000) — Client Drift Variance ({comparison_tag})", 
                     fontsize=11.5 if is_standalone else 10.0, fontweight="bold", color="#333333")
        ax.set_ylabel(r"Variance ($\sigma^2_t$)", fontsize=9.5)
        ax.set_xlabel("Communication Round", fontsize=9.5)
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.set_xlim(xlim)

        # FDA E3TO30
        if df_csv_e30 is not None:
            df_A = df_csv_e30[df_csv_e30['active_matrix'] == 'A']
            df_B = df_csv_e30[df_csv_e30['active_matrix'] == 'B']
            ax.plot(df_A['round'], df_A['variance'], label="FDA E3TO30 (A)", color=c_fda30_odd, linestyle="-", marker="^", markersize=4.2, linewidth=1.8, alpha=0.9, zorder=5)
            ax.plot(df_B['round'], df_B['variance'], label="FDA E3TO30 (B)", color=c_fda30_even, linestyle="--", marker="^", markersize=4.0, linewidth=1.6, alpha=0.9, zorder=5)

        # FDA E3TO20
        if df_csv_e20 is not None:
            df_A = df_csv_e20[df_csv_e20['active_matrix'] == 'A']
            df_B = df_csv_e20[df_csv_e20['active_matrix'] == 'B']
            ax.plot(df_A['round'], df_A['variance'], label="FDA E3TO20 (A)", color=c_fda20_odd, linestyle="-", marker="o", markersize=4.0, linewidth=1.8, alpha=0.9, zorder=4)
            ax.plot(df_B['round'], df_B['variance'], label="FDA E3TO20 (B)", color=c_fda20_even, linestyle="--", marker="o", markersize=3.8, linewidth=1.6, alpha=0.9, zorder=4)

        # FDA E3TO40
        if df_csv_e40 is not None:
            df_A = df_csv_e40[df_csv_e40['active_matrix'] == 'A']
            df_B = df_csv_e40[df_csv_e40['active_matrix'] == 'B']
            ax.plot(df_A['round'], df_A['variance'], label="FDA E3TO40 (A)", color=c_fda40_odd, linestyle="-", marker="D", markersize=4.0, linewidth=1.8, alpha=0.9, zorder=5)
            ax.plot(df_B['round'], df_B['variance'], label="FDA E3TO40 (B)", color=c_fda40_even, linestyle="--", marker="D", markersize=3.8, linewidth=1.6, alpha=0.9, zorder=5)

        # Old FDA runs
        for orun in old_runs:
            df_old = orun["df"]
            df_A = df_old[df_old['active_matrix'] == 'A']
            df_B = df_old[df_old['active_matrix'] == 'B']
            lbl_tag = orun["label"].replace("FDA ", "")
            ax.plot(df_A['round'], df_A['variance'], label=f"{lbl_tag} (A)", color=orun["c_odd"], linestyle="-.", marker=orun["marker"], markersize=4.0, linewidth=1.6, alpha=0.9, zorder=4)
            ax.plot(df_B['round'], df_B['variance'], label=f"{lbl_tag} (B)", color=orun["c_even"], linestyle=":", marker=orun["marker"], markersize=3.8, linewidth=1.4, alpha=0.9, zorder=4)

        # RoLoRA
        if df_csv_ro20 is not None:
            df_A = df_csv_ro20[df_csv_ro20['active_matrix'] == 'A']
            df_B = df_csv_ro20[df_csv_ro20['active_matrix'] == 'B']
            ax.plot(df_A['round'], df_A['variance'], label="RoLoRA E20 (A)", color="#78350f", linestyle="-", marker="v", markersize=3.8, linewidth=1.6, alpha=0.85, zorder=3)
            ax.plot(df_B['round'], df_B['variance'], label="RoLoRA E20 (B)", color="#d97706", linestyle="--", marker="v", markersize=3.6, linewidth=1.5, alpha=0.85, zorder=3)

        if df_csv_ro15 is not None:
            df_A = df_csv_ro15[df_csv_ro15['active_matrix'] == 'A']
            df_B = df_csv_ro15[df_csv_ro15['active_matrix'] == 'B']
            ax.plot(df_A['round'], df_A['variance'], label="RoLoRA E15 (A)", color="#c2410c", linestyle="-", marker=">", markersize=3.8, linewidth=1.6, alpha=0.85, zorder=3)
            ax.plot(df_B['round'], df_B['variance'], label="RoLoRA E15 (B)", color="#fdba74", linestyle="--", marker=">", markersize=3.6, linewidth=1.5, alpha=0.85, zorder=3)

        if df_csv_ro10 is not None:
            df_A = df_csv_ro10[df_csv_ro10['active_matrix'] == 'A']
            df_B = df_csv_ro10[df_csv_ro10['active_matrix'] == 'B']
            ax.plot(df_A['round'], df_A['variance'], label="RoLoRA E10 (A)", color="#c2410c", linestyle="-", marker="s", markersize=3.8, linewidth=1.6, alpha=0.85, zorder=3)
            ax.plot(df_B['round'], df_B['variance'], label="RoLoRA E10 (B)", color="#fb923c", linestyle="--", marker="s", markersize=3.6, linewidth=1.5, alpha=0.85, zorder=3)

        if df_csv_ro5 is not None:
            df_A = df_csv_ro5[df_csv_ro5['active_matrix'] == 'A']
            df_B = df_csv_ro5[df_csv_ro5['active_matrix'] == 'B']
            ax.plot(df_A['round'], df_A['variance'], label="RoLoRA E5 (A)", color="#991b1b", linestyle="-", marker="x", markersize=4.0, linewidth=1.5, alpha=0.85, zorder=2)
            ax.plot(df_B['round'], df_B['variance'], label="RoLoRA E5 (B)", color="#f87171", linestyle="--", marker="x", markersize=3.8, linewidth=1.4, alpha=0.85, zorder=2)

        # LoRA & FFA-LoRA
        if include_lora:
            if df_csv_lo20 is not None:
                ax.plot(df_csv_lo20['round'], df_csv_lo20['variance'], label="LoRA E20", color=c_lora20_var, linestyle="-", marker="<", markersize=3.4, linewidth=1.5, alpha=0.85, zorder=3)
            if df_csv_lo10 is not None:
                ax.plot(df_csv_lo10['round'], df_csv_lo10['variance'], label="LoRA E10", color=c_lora10_var, linestyle="-", marker="D", markersize=3.4, linewidth=1.5, alpha=0.85, zorder=3)
            if df_csv_lo5 is not None:
                ax.plot(df_csv_lo5['round'], df_csv_lo5['variance'], label="LoRA E5", color=c_lora5_var, linestyle=":", marker="v", markersize=3.4, linewidth=1.5, alpha=0.85, zorder=2)
            if df_csv_ffa10 is not None:
                ax.plot(df_csv_ffa10['round'], df_csv_ffa10['variance'], label="FFA-LoRA E10", color=c_ffa10_var, linestyle="-", marker="h", markersize=3.4, linewidth=1.5, alpha=0.85, zorder=3)
            if df_csv_ffa5 is not None:
                ax.plot(df_csv_ffa5['round'], df_csv_ffa5['variance'], label="FFA-LoRA E5", color=c_ffa5_var, linestyle=":", marker="p", markersize=3.4, linewidth=1.5, alpha=0.85, zorder=2)

        ax.legend(ncol=2, fontsize=6.5 if is_standalone else 5.8, loc="upper left", framealpha=0.88, facecolor="white", edgecolor="#cccccc")

    # -------------------------------------------------------------
    # 3b. Helper: draw client drift variance vs round on LOG SCALE (matching fda_opt.pdf Figure 3)
    # -------------------------------------------------------------
    def draw_var_log(ax, is_standalone=False):
        ax.set_title(f"{display_name} (C={c}, R={r}, lr={lr}, α=1000) — Client Drift Variance [Log Scale] ({comparison_tag})", 
                     fontsize=10.0 if is_standalone else 8.8, fontweight="bold", color="#333333")
        ax.set_ylabel(r"Model Variance ($\sigma^2_t$, Log Scale)", fontsize=9.5)
        ax.set_xlabel("Communication Round", fontsize=9.5)
        ax.set_yscale("log")
        ax.grid(True, which="both", linestyle="--", alpha=0.35)
        ax.set_xlim(xlim)

        def _safe_var(d):
            if d is None or len(d) == 0 or 'variance' not in d.columns:
                return d
            return d[d['variance'] > 0]

        # FDA E3TO30
        if df_csv_e30 is not None:
            df_A = _safe_var(df_csv_e30[df_csv_e30['active_matrix'] == 'A'])
            df_B = _safe_var(df_csv_e30[df_csv_e30['active_matrix'] == 'B'])
            if df_A is not None and len(df_A): ax.plot(df_A['round'], df_A['variance'], label="FDA E3TO30 (A)", color=c_fda30_odd, linestyle="-", marker="^", markersize=4.2, linewidth=1.8, alpha=0.9, zorder=5)
            if df_B is not None and len(df_B): ax.plot(df_B['round'], df_B['variance'], label="FDA E3TO30 (B)", color=c_fda30_even, linestyle="--", marker="^", markersize=4.0, linewidth=1.6, alpha=0.9, zorder=5)

        # FDA E3TO20
        if df_csv_e20 is not None:
            df_A = _safe_var(df_csv_e20[df_csv_e20['active_matrix'] == 'A'])
            df_B = _safe_var(df_csv_e20[df_csv_e20['active_matrix'] == 'B'])
            if df_A is not None and len(df_A): ax.plot(df_A['round'], df_A['variance'], label="FDA E3TO20 (A)", color=c_fda20_odd, linestyle="-", marker="o", markersize=4.0, linewidth=1.8, alpha=0.9, zorder=4)
            if df_B is not None and len(df_B): ax.plot(df_B['round'], df_B['variance'], label="FDA E3TO20 (B)", color=c_fda20_even, linestyle="--", marker="o", markersize=3.8, linewidth=1.6, alpha=0.9, zorder=4)

        # FDA E3TO40
        if df_csv_e40 is not None:
            df_A = _safe_var(df_csv_e40[df_csv_e40['active_matrix'] == 'A'])
            df_B = _safe_var(df_csv_e40[df_csv_e40['active_matrix'] == 'B'])
            if df_A is not None and len(df_A): ax.plot(df_A['round'], df_A['variance'], label="FDA E3TO40 (A)", color=c_fda40_odd, linestyle="-", marker="D", markersize=4.0, linewidth=1.8, alpha=0.9, zorder=5)
            if df_B is not None and len(df_B): ax.plot(df_B['round'], df_B['variance'], label="FDA E3TO40 (B)", color=c_fda40_even, linestyle="--", marker="D", markersize=3.8, linewidth=1.6, alpha=0.9, zorder=5)

        # Old FDA runs
        for orun in old_runs:
            df_old = orun["df"]
            df_A = _safe_var(df_old[df_old['active_matrix'] == 'A'])
            df_B = _safe_var(df_old[df_old['active_matrix'] == 'B'])
            lbl_tag = orun["label"].replace("FDA ", "")
            if df_A is not None and len(df_A): ax.plot(df_A['round'], df_A['variance'], label=f"{lbl_tag} (A)", color=orun["c_odd"], linestyle="-.", marker=orun["marker"], markersize=4.0, linewidth=1.6, alpha=0.9, zorder=4)
            if df_B is not None and len(df_B): ax.plot(df_B['round'], df_B['variance'], label=f"{lbl_tag} (B)", color=orun["c_even"], linestyle=":", marker=orun["marker"], markersize=3.8, linewidth=1.4, alpha=0.9, zorder=4)

        # RoLoRA
        if df_csv_ro20 is not None:
            df_A = _safe_var(df_csv_ro20[df_csv_ro20['active_matrix'] == 'A'])
            df_B = _safe_var(df_csv_ro20[df_csv_ro20['active_matrix'] == 'B'])
            if df_A is not None and len(df_A): ax.plot(df_A['round'], df_A['variance'], label="RoLoRA E20 (A)", color="#78350f", linestyle="-", marker="v", markersize=3.8, linewidth=1.6, alpha=0.85, zorder=3)
            if df_B is not None and len(df_B): ax.plot(df_B['round'], df_B['variance'], label="RoLoRA E20 (B)", color="#d97706", linestyle="--", marker="v", markersize=3.6, linewidth=1.5, alpha=0.85, zorder=3)

        if df_csv_ro15 is not None:
            df_A = _safe_var(df_csv_ro15[df_csv_ro15['active_matrix'] == 'A'])
            df_B = _safe_var(df_csv_ro15[df_csv_ro15['active_matrix'] == 'B'])
            if df_A is not None and len(df_A): ax.plot(df_A['round'], df_A['variance'], label="RoLoRA E15 (A)", color="#c2410c", linestyle="-", marker=">", markersize=3.8, linewidth=1.6, alpha=0.85, zorder=3)
            if df_B is not None and len(df_B): ax.plot(df_B['round'], df_B['variance'], label="RoLoRA E15 (B)", color="#fdba74", linestyle="--", marker=">", markersize=3.6, linewidth=1.5, alpha=0.85, zorder=3)

        if df_csv_ro10 is not None:
            df_A = _safe_var(df_csv_ro10[df_csv_ro10['active_matrix'] == 'A'])
            df_B = _safe_var(df_csv_ro10[df_csv_ro10['active_matrix'] == 'B'])
            if df_A is not None and len(df_A): ax.plot(df_A['round'], df_A['variance'], label="RoLoRA E10 (A)", color="#c2410c", linestyle="-", marker="s", markersize=3.8, linewidth=1.6, alpha=0.85, zorder=3)
            if df_B is not None and len(df_B): ax.plot(df_B['round'], df_B['variance'], label="RoLoRA E10 (B)", color="#fb923c", linestyle="--", marker="s", markersize=3.6, linewidth=1.5, alpha=0.85, zorder=3)

        if df_csv_ro5 is not None:
            df_A = _safe_var(df_csv_ro5[df_csv_ro5['active_matrix'] == 'A'])
            df_B = _safe_var(df_csv_ro5[df_csv_ro5['active_matrix'] == 'B'])
            if df_A is not None and len(df_A): ax.plot(df_A['round'], df_A['variance'], label="RoLoRA E5 (A)", color="#991b1b", linestyle="-", marker="x", markersize=4.0, linewidth=1.5, alpha=0.85, zorder=2)
            if df_B is not None and len(df_B): ax.plot(df_B['round'], df_B['variance'], label="RoLoRA E5 (B)", color="#f87171", linestyle="--", marker="x", markersize=3.8, linewidth=1.4, alpha=0.85, zorder=2)

        # LoRA & FFA-LoRA
        if include_lora:
            if df_csv_lo20 is not None:
                d = _safe_var(df_csv_lo20)
                if d is not None and len(d): ax.plot(d['round'], d['variance'], label="LoRA E20", color=c_lora20_var, linestyle="-", marker="<", markersize=3.4, linewidth=1.5, alpha=0.85, zorder=3)
            if df_csv_lo10 is not None:
                d = _safe_var(df_csv_lo10)
                if d is not None and len(d): ax.plot(d['round'], d['variance'], label="LoRA E10", color=c_lora10_var, linestyle="-", marker="D", markersize=3.4, linewidth=1.5, alpha=0.85, zorder=3)
            if df_csv_lo5 is not None:
                d = _safe_var(df_csv_lo5)
                if d is not None and len(d): ax.plot(d['round'], d['variance'], label="LoRA E5", color=c_lora5_var, linestyle=":", marker="v", markersize=3.4, linewidth=1.5, alpha=0.85, zorder=2)
            if df_csv_ffa10 is not None:
                d = _safe_var(df_csv_ffa10)
                if d is not None and len(d): ax.plot(d['round'], d['variance'], label="FFA-LoRA E10", color=c_ffa10_var, linestyle="-", marker="h", markersize=3.4, linewidth=1.5, alpha=0.85, zorder=3)
            if df_csv_ffa5 is not None:
                d = _safe_var(df_csv_ffa5)
                if d is not None and len(d): ax.plot(d['round'], d['variance'], label="FFA-LoRA E5", color=c_ffa5_var, linestyle=":", marker="p", markersize=3.4, linewidth=1.5, alpha=0.85, zorder=2)

        ax.legend(ncol=2, fontsize=6.5 if is_standalone else 5.8, loc="best", framealpha=0.88, facecolor="white", edgecolor="#cccccc")

    # -------------------------------------------------------------
    # 4. Helper: draw accuracy vs communication cost (MB)
    # -------------------------------------------------------------
    def draw_acc_vs_cost(ax, is_standalone=False):
        ax.set_title(f"{display_name} (Clients={c}, Rounds={r}, lr={lr}, α=1000) — Accuracy vs. Cost ({comparison_tag})", 
                     fontsize=11.5 if is_standalone else 10.0, fontweight="bold")
        ax.set_ylabel("Validation Accuracy", fontsize=9.5)
        ax.set_xlabel("Bi-directional Cost (MB)", fontsize=9.5)
        ax.grid(True, linestyle="--", alpha=0.4)

        if df_csv_e20 is not None:
            acc = df_csv_e20["accuracy"]
            cost_x = df_csv_e20["round"] * rate_fda_ro
            best_rnd = int(df_csv_e20.loc[acc.idxmax(), "round"])
            peak_acc = acc.max()
            best_cost = best_rnd * rate_fda_ro
            lbl = f"FDA E3TO20 ({peak_acc*100:.1f}%)"
            ax.plot(cost_x, acc, label=lbl, color=c_fda20_acc, linestyle="-", marker="o", markersize=4.5, linewidth=2.0, zorder=4)
            ax.scatter(best_cost, peak_acc, marker="*", s=160, color="#0047AB", edgecolors="black", linewidth=0.5, zorder=7)

        if df_csv_e30 is not None:
            acc = df_csv_e30["accuracy"]
            cost_x = df_csv_e30["round"] * rate_fda_ro
            best_rnd = int(df_csv_e30.loc[acc.idxmax(), "round"])
            peak_acc = acc.max()
            best_cost = best_rnd * rate_fda_ro
            lbl = f"FDA E3TO30 ({peak_acc*100:.1f}%)"
            ax.plot(cost_x, acc, label=lbl, color=c_fda30_acc, linestyle="-", marker="^", markersize=5.0, linewidth=2.2, zorder=5)
            ax.scatter(best_cost, peak_acc, marker="*", s=180, color="#4a148c", edgecolors="gold", linewidth=0.8, zorder=8)

        if df_csv_e40 is not None:
            acc = df_csv_e40["accuracy"]
            cost_x = df_csv_e40["round"] * rate_fda_ro
            best_rnd = int(df_csv_e40.loc[acc.idxmax(), "round"])
            peak_acc = acc.max()
            best_cost = best_rnd * rate_fda_ro
            lbl = f"FDA E3TO40 ({peak_acc*100:.1f}%)"
            ax.plot(cost_x, acc, label=lbl, color=c_fda40_acc, linestyle="-", marker="D", markersize=4.8, linewidth=2.0, zorder=5)
            ax.scatter(best_cost, peak_acc, marker="*", s=170, color=c_fda40_odd, edgecolors="black", linewidth=0.5, zorder=8)

        for orun in old_runs:
            df_old = orun["df"]
            acc = df_old["accuracy"]
            cost_x = df_old["round"] * rate_fda_ro
            best_rnd = int(df_old.loc[acc.idxmax(), "round"])
            peak_acc = acc.max()
            best_cost = best_rnd * rate_fda_ro
            lbl = f"{orun['label']} ({peak_acc*100:.1f}%)"
            ax.plot(cost_x, acc, label=lbl, color=orun["c_acc"], linestyle=orun["ls"], marker=orun["marker"], markersize=4.2, linewidth=1.8, zorder=4)
            ax.scatter(best_cost, peak_acc, marker="*", s=150, color=orun["c_odd"], edgecolors="black", linewidth=0.5, zorder=7)

        if df_csv_ro20 is not None:
            acc = df_csv_ro20["accuracy"]
            cost_x = df_csv_ro20["round"] * rate_fda_ro
            best_rnd = int(df_csv_ro20.loc[acc.idxmax(), "round"])
            peak_acc = acc.max()
            best_cost = best_rnd * rate_fda_ro
            lbl = f"RoLoRA E20 ({peak_acc*100:.1f}%)"
            ax.plot(cost_x, acc, label=lbl, color=c_ro20_acc, linestyle="-", marker="v", markersize=4.0, linewidth=1.6, zorder=3)
            ax.scatter(best_cost, peak_acc, marker="*", s=140, color="#78350f", edgecolors="black", linewidth=0.5, zorder=6)

        if df_csv_ro20_rk8 is not None:
            acc = df_csv_ro20_rk8["accuracy"]
            cost_x = df_csv_ro20_rk8["round"] * rate_fda_ro
            peak_acc = acc.max()
            lbl = f"RoLoRA E20 RK8 ({peak_acc*100:.1f}%)"
            ax.plot(cost_x, acc, label=lbl, color=c_ro20_rk8_acc, linestyle="--", marker="v", markersize=3.6, linewidth=1.4, zorder=3)

        if df_csv_ro15 is not None:
            acc = df_csv_ro15["accuracy"]
            cost_x = df_csv_ro15["round"] * rate_fda_ro
            best_rnd = int(df_csv_ro15.loc[acc.idxmax(), "round"])
            peak_acc = acc.max()
            best_cost = best_rnd * rate_fda_ro
            lbl = f"RoLoRA E15 ({peak_acc*100:.1f}%)"
            ax.plot(cost_x, acc, label=lbl, color=c_ro15_acc, linestyle="-", marker=">", markersize=4.0, linewidth=1.6, zorder=3)
            ax.scatter(best_cost, peak_acc, marker="*", s=140, color="#9a3412", edgecolors="black", linewidth=0.5, zorder=6)

        if df_csv_ro10 is not None:
            acc = df_csv_ro10["accuracy"]
            cost_x = df_csv_ro10["round"] * rate_fda_ro
            best_rnd = int(df_csv_ro10.loc[acc.idxmax(), "round"])
            peak_acc = acc.max()
            best_cost = best_rnd * rate_fda_ro
            lbl = f"RoLoRA E10 ({peak_acc*100:.1f}%)"
            ax.plot(cost_x, acc, label=lbl, color=c_ro10_acc, linestyle="-", marker="s", markersize=4.2, linewidth=1.8, zorder=3)
            ax.scatter(best_cost, peak_acc, marker="*", s=150, color="#b23b00", edgecolors="black", linewidth=0.5, zorder=6)

        if df_csv_ro5 is not None:
            acc = df_csv_ro5["accuracy"]
            cost_x = df_csv_ro5["round"] * rate_fda_ro
            peak_acc = acc.max()
            lbl = f"RoLoRA E5 ({peak_acc*100:.1f}%)"
            ax.plot(cost_x, acc, label=lbl, color=c_ro5_acc, linestyle="--", marker="x", markersize=4.5, linewidth=1.6, zorder=2)

        if include_lora:
            if df_csv_lo20 is not None:
                acc = df_csv_lo20["accuracy"]
                cost_x = df_csv_lo20["round"] * rate_lora
                peak_acc = acc.max()
                lbl = f"LoRA E20 ({peak_acc*100:.1f}%)"
                ax.plot(cost_x, acc, label=lbl, color=c_lora20_acc, linestyle="-", marker="<", markersize=3.8, linewidth=1.6, zorder=3)

            if df_csv_lo20_rk8 is not None:
                acc = df_csv_lo20_rk8["accuracy"]
                cost_x = df_csv_lo20_rk8["round"] * rate_lora
                peak_acc = acc.max()
                lbl = f"LoRA E20 RK8 ({peak_acc*100:.1f}%)"
                ax.plot(cost_x, acc, label=lbl, color=c_lora20_rk8_acc, linestyle="--", marker="<", markersize=3.4, linewidth=1.3, zorder=2)

            if df_csv_lo10 is not None:
                acc = df_csv_lo10["accuracy"]
                cost_x = df_csv_lo10["round"] * rate_lora
                peak_acc = acc.max()
                lbl = f"LoRA E10 ({peak_acc*100:.1f}%)"
                ax.plot(cost_x, acc, label=lbl, color=c_lora10_acc, linestyle="-", marker="D", markersize=4.0, linewidth=1.8, zorder=3)

            if df_csv_lo5 is not None:
                acc = df_csv_lo5["accuracy"]
                cost_x = df_csv_lo5["round"] * rate_lora
                peak_acc = acc.max()
                lbl = f"LoRA E5 ({peak_acc*100:.1f}%)"
                ax.plot(cost_x, acc, label=lbl, color=c_lora5_acc, linestyle="--", marker="v", markersize=4.2, linewidth=1.6, zorder=2)

            if df_csv_ffa10 is not None:
                acc = df_csv_ffa10["accuracy"]
                cost_x = df_csv_ffa10["round"] * rate_lora
                best_rnd = int(df_csv_ffa10.loc[acc.idxmax(), "round"])
                peak_acc = acc.max()
                best_cost = best_rnd * rate_lora
                lbl = f"FFA-LoRA E10 ({peak_acc*100:.1f}%)"
                ax.plot(cost_x, acc, label=lbl, color=c_ffa10_acc, linestyle="-", marker="h", markersize=4.0, linewidth=1.8, zorder=3)
                ax.scatter(best_cost, peak_acc, marker="*", s=140, color="#155e75", edgecolors="black", linewidth=0.5, zorder=6)

            if df_csv_ffa5 is not None:
                acc = df_csv_ffa5["accuracy"]
                cost_x = df_csv_ffa5["round"] * rate_lora
                peak_acc = acc.max()
                lbl = f"FFA-LoRA E5 ({peak_acc*100:.1f}%)"
                ax.plot(cost_x, acc, label=lbl, color=c_ffa5_acc, linestyle="--", marker="p", markersize=3.8, linewidth=1.5, zorder=2)

        ax.legend(fontsize=6.8 if is_standalone else 6.2, loc="lower right", framealpha=0.88, facecolor="white", edgecolor="#cccccc")

    # -------------------------------------------------------------
    # 5. Helper: draw FDA threshold per matrix per round
    # -------------------------------------------------------------
    def draw_fda_threshold(ax, is_standalone=False):
        ax.set_title(f"{display_name} (Clients={c}, Rounds={r}, lr={lr}, α=1000) — FDA Threshold per Matrix", 
                     fontsize=11.5 if is_standalone else 10.0, fontweight="bold", color="#333333")
        ax.set_ylabel(r"Controller Threshold ($\theta$)", fontsize=9.5)
        ax.set_xlabel("Communication Round", fontsize=9.5)
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.set_xlim(xlim)

        plotted = False
        if df_csv_e20 is not None:
            df_a20 = df_csv_e20[df_csv_e20["active_matrix"] == "A"]
            df_b20 = df_csv_e20[df_csv_e20["active_matrix"] == "B"]
            if not df_a20.empty and "theta_phase" in df_a20.columns:
                ax.plot(df_a20["round"], df_a20["theta_phase"], label=r"FDA E3TO20: $\theta_A$ (Matrix A)", 
                        color=c_fda20_odd, linestyle="-", marker="o", markersize=4.5, linewidth=1.8, zorder=4)
                plotted = True
            if not df_b20.empty and "theta_phase" in df_b20.columns:
                ax.plot(df_b20["round"], df_b20["theta_phase"], label=r"FDA E3TO20: $\theta_B$ (Matrix B)", 
                        color=c_fda20_even, linestyle="-", marker="s", markersize=4.5, linewidth=1.8, zorder=4)
                plotted = True

        if df_csv_e30 is not None:
            df_a30 = df_csv_e30[df_csv_e30["active_matrix"] == "A"]
            df_b30 = df_csv_e30[df_csv_e30["active_matrix"] == "B"]
            if not df_a30.empty and "theta_phase" in df_a30.columns:
                ax.plot(df_a30["round"], df_a30["theta_phase"], label=r"FDA E3TO30: $\theta_A$ (Matrix A)", 
                        color=c_fda30_odd, linestyle="-", marker="^", markersize=5.0, linewidth=1.8, zorder=5)
                plotted = True
            if not df_b30.empty and "theta_phase" in df_b30.columns:
                ax.plot(df_b30["round"], df_b30["theta_phase"], label=r"FDA E3TO30: $\theta_B$ (Matrix B)", 
                        color=c_fda30_even, linestyle="-", marker="d", markersize=5.0, linewidth=1.8, zorder=5)
                plotted = True

        if df_csv_e40 is not None:
            df_a40 = df_csv_e40[df_csv_e40["active_matrix"] == "A"]
            df_b40 = df_csv_e40[df_csv_e40["active_matrix"] == "B"]
            if not df_a40.empty and "theta_phase" in df_a40.columns:
                ax.plot(df_a40["round"], df_a40["theta_phase"], label=r"FDA E3TO40: $\theta_A$ (Matrix A)", 
                        color=c_fda40_odd, linestyle="-", marker="D", markersize=4.5, linewidth=1.8, zorder=5)
                plotted = True
            if not df_b40.empty and "theta_phase" in df_b40.columns:
                ax.plot(df_b40["round"], df_b40["theta_phase"], label=r"FDA E3TO40: $\theta_B$ (Matrix B)", 
                        color=c_fda40_even, linestyle="-", marker="s", markersize=4.5, linewidth=1.8, zorder=5)
                plotted = True

        for orun in old_runs:
            df_old = orun["df"]
            if "theta_phase" in df_old.columns:
                df_a = df_old[df_old["active_matrix"] == "A"]
                df_b = df_old[df_old["active_matrix"] == "B"]
                lbl_tag = orun["label"].replace("FDA ", "")
                if not df_a.empty:
                    ax.plot(df_a["round"], df_a["theta_phase"], label=rf"{lbl_tag}: $\theta_A$", 
                            color=orun["c_odd"], linestyle="-.", marker=orun["marker"], markersize=4.0, linewidth=1.6, zorder=4)
                    plotted = True
                if not df_b.empty:
                    ax.plot(df_b["round"], df_b["theta_phase"], label=rf"{lbl_tag}: $\theta_B$", 
                            color=orun["c_even"], linestyle=":", marker=orun["marker"], markersize=4.0, linewidth=1.6, zorder=4)
                    plotted = True

        if not plotted:
            ax.text(0.5, 0.5, "No Adaptive FDA Telemetry", transform=ax.transAxes, ha="center", va="center", color="gray", fontsize=11)
        else:
            ax.legend(fontsize=7.0 if is_standalone else 6.0, loc="upper left", framealpha=0.88, facecolor="white", edgecolor="#cccccc")

    # -------------------------------------------------------------
    # 6. Helper: draw variance vs cumulative local epochs
    # -------------------------------------------------------------
    def draw_var_vs_epoch(ax, is_standalone=False):
        ax.set_title(f"{display_name} (Clients={c}, Rounds={r}, lr={lr}, α=1000) — Variance vs. Cumulative Epochs ({comparison_tag})", 
                     fontsize=11.5 if is_standalone else 10.0, fontweight="bold", color="#333333")
        ax.set_ylabel(r"Variance ($\sigma^2$)", fontsize=9.5)
        ax.set_xlabel("Cumulative Local Epochs", fontsize=9.5)
        ax.grid(True, linestyle="--", alpha=0.4)

        if df_csv_e30 is not None:
            cum_eps = np.cumsum(df_csv_e30['round_epochs'].values)
            mask_A = (df_csv_e30['active_matrix'] == 'A').values
            mask_B = (df_csv_e30['active_matrix'] == 'B').values
            ax.plot(cum_eps[mask_A], df_csv_e30['variance'].values[mask_A], label="FDA E3TO30 (A)", color=c_fda30_odd, linestyle="-", marker="^", markersize=4.0, linewidth=1.8, alpha=0.9, zorder=5)
            ax.plot(cum_eps[mask_B], df_csv_e30['variance'].values[mask_B], label="FDA E3TO30 (B)", color=c_fda30_even, linestyle="--", marker="^", markersize=3.8, linewidth=1.6, alpha=0.9, zorder=5)

        if df_csv_e20 is not None:
            cum_eps = np.cumsum(df_csv_e20['round_epochs'].values)
            mask_A = (df_csv_e20['active_matrix'] == 'A').values
            mask_B = (df_csv_e20['active_matrix'] == 'B').values
            ax.plot(cum_eps[mask_A], df_csv_e20['variance'].values[mask_A], label="FDA E3TO20 (A)", color=c_fda20_odd, linestyle="-", marker="o", markersize=3.8, linewidth=1.8, alpha=0.9, zorder=4)
            ax.plot(cum_eps[mask_B], df_csv_e20['variance'].values[mask_B], label="FDA E3TO20 (B)", color=c_fda20_even, linestyle="--", marker="o", markersize=3.6, linewidth=1.6, alpha=0.9, zorder=4)

        if df_csv_e40 is not None:
            cum_eps = np.cumsum(df_csv_e40['round_epochs'].values)
            mask_A = (df_csv_e40['active_matrix'] == 'A').values
            mask_B = (df_csv_e40['active_matrix'] == 'B').values
            ax.plot(cum_eps[mask_A], df_csv_e40['variance'].values[mask_A], label="FDA E3TO40 (A)", color=c_fda40_odd, linestyle="-", marker="D", markersize=4.0, linewidth=1.8, alpha=0.9, zorder=5)
            ax.plot(cum_eps[mask_B], df_csv_e40['variance'].values[mask_B], label="FDA E3TO40 (B)", color=c_fda40_even, linestyle="--", marker="D", markersize=3.8, linewidth=1.6, alpha=0.9, zorder=5)

        for orun in old_runs:
            df_old = orun["df"]
            cum_eps = np.cumsum(df_old['round_epochs'].values)
            mask_A = (df_old['active_matrix'] == 'A').values
            mask_B = (df_old['active_matrix'] == 'B').values
            lbl_tag = orun["label"].replace("FDA ", "")
            ax.plot(cum_eps[mask_A], df_old['variance'].values[mask_A], label=f"{lbl_tag} (A)", color=orun["c_odd"], linestyle="-.", marker=orun["marker"], markersize=3.8, linewidth=1.6, alpha=0.9, zorder=4)
            ax.plot(cum_eps[mask_B], df_old['variance'].values[mask_B], label=f"{lbl_tag} (B)", color=orun["c_even"], linestyle=":", marker=orun["marker"], markersize=3.6, linewidth=1.4, alpha=0.9, zorder=4)

        if df_csv_ro20 is not None:
            cum_eps = np.cumsum(df_csv_ro20['round_epochs'].values)
            mask_A = (df_csv_ro20['active_matrix'] == 'A').values
            mask_B = (df_csv_ro20['active_matrix'] == 'B').values
            ax.plot(cum_eps[mask_A], df_csv_ro20['variance'].values[mask_A], label="RoLoRA E20 (A)", color="#78350f", linestyle="-", marker="v", markersize=3.8, linewidth=1.5, alpha=0.85, zorder=3)
            ax.plot(cum_eps[mask_B], df_csv_ro20['variance'].values[mask_B], label="RoLoRA E20 (B)", color="#d97706", linestyle="--", marker="v", markersize=3.6, linewidth=1.4, alpha=0.85, zorder=3)

        if df_csv_ro15 is not None:
            cum_eps = np.cumsum(df_csv_ro15['round_epochs'].values)
            mask_A = (df_csv_ro15['active_matrix'] == 'A').values
            mask_B = (df_csv_ro15['active_matrix'] == 'B').values
            ax.plot(cum_eps[mask_A], df_csv_ro15['variance'].values[mask_A], label="RoLoRA E15 (A)", color="#c2410c", linestyle="-", marker=">", markersize=3.8, linewidth=1.5, alpha=0.85, zorder=3)
            ax.plot(cum_eps[mask_B], df_csv_ro15['variance'].values[mask_B], label="RoLoRA E15 (B)", color="#fdba74", linestyle="--", marker=">", markersize=3.6, linewidth=1.4, alpha=0.85, zorder=3)

        if df_csv_ro10 is not None:
            cum_eps = np.cumsum(df_csv_ro10['round_epochs'].values)
            mask_A = (df_csv_ro10['active_matrix'] == 'A').values
            mask_B = (df_csv_ro10['active_matrix'] == 'B').values
            ax.plot(cum_eps[mask_A], df_csv_ro10['variance'].values[mask_A], label="RoLoRA E10 (A)", color="#c2410c", linestyle="-", marker="s", markersize=3.8, linewidth=1.6, alpha=0.85, zorder=3)
            ax.plot(cum_eps[mask_B], df_csv_ro10['variance'].values[mask_B], label="RoLoRA E10 (B)", color="#fb923c", linestyle="--", marker="s", markersize=3.6, linewidth=1.5, alpha=0.85, zorder=3)

        if df_csv_ro5 is not None:
            cum_eps = np.cumsum(df_csv_ro5['round_epochs'].values)
            mask_A = (df_csv_ro5['active_matrix'] == 'A').values
            mask_B = (df_csv_ro5['active_matrix'] == 'B').values
            ax.plot(cum_eps[mask_A], df_csv_ro5['variance'].values[mask_A], label="RoLoRA E5 (A)", color="#991b1b", linestyle="-", marker="x", markersize=4.0, linewidth=1.5, alpha=0.85, zorder=2)
            ax.plot(cum_eps[mask_B], df_csv_ro5['variance'].values[mask_B], label="RoLoRA E5 (B)", color="#f87171", linestyle="--", marker="x", markersize=3.8, linewidth=1.4, alpha=0.85, zorder=2)

        if include_lora:
            if df_csv_lo20 is not None:
                cum_eps = np.cumsum(df_csv_lo20['round_epochs'].values)
                ax.plot(cum_eps, df_csv_lo20['variance'].values, label="LoRA E20", color=c_lora20_var, linestyle="-", marker="<", markersize=3.4, linewidth=1.5, alpha=0.85, zorder=3)
            if df_csv_lo10 is not None:
                cum_eps = np.cumsum(df_csv_lo10['round_epochs'].values)
                ax.plot(cum_eps, df_csv_lo10['variance'].values, label="LoRA E10", color=c_lora10_var, linestyle="-", marker="D", markersize=3.4, linewidth=1.5, alpha=0.85, zorder=3)
            if df_csv_lo5 is not None:
                cum_eps = np.cumsum(df_csv_lo5['round_epochs'].values)
                ax.plot(cum_eps, df_csv_lo5['variance'].values, label="LoRA E5", color=c_lora5_var, linestyle=":", marker="v", markersize=3.4, linewidth=1.5, alpha=0.85, zorder=2)
            if df_csv_ffa10 is not None:
                cum_eps = np.cumsum(df_csv_ffa10['round_epochs'].values)
                ax.plot(cum_eps, df_csv_ffa10['variance'].values, label="FFA-LoRA E10", color=c_ffa10_var, linestyle="-", marker="h", markersize=3.4, linewidth=1.5, alpha=0.85, zorder=3)
            if df_csv_ffa5 is not None:
                cum_eps = np.cumsum(df_csv_ffa5['round_epochs'].values)
                ax.plot(cum_eps, df_csv_ffa5['variance'].values, label="FFA-LoRA E5", color=c_ffa5_var, linestyle=":", marker="p", markersize=3.4, linewidth=1.5, alpha=0.85, zorder=2)

        ax.legend(ncol=2, fontsize=6.5 if is_standalone else 5.8, loc="upper left", framealpha=0.88, facecolor="white", edgecolor="#cccccc")

    # -------------------------------------------------------------
    # 7. Helper: draw train vs validation accuracy gap
    # -------------------------------------------------------------
    def draw_train_val_gap(ax, is_standalone=False):
        ax.set_title(f"{display_name} (Clients={c}, Rounds={r}, lr={lr}, α=1000) — Train vs. Validation Accuracy Gap ({comparison_tag})", 
                     fontsize=11.5 if is_standalone else 10.0, fontweight="bold")
        ax.set_ylabel("Accuracy", fontsize=9.5)
        ax.set_xlabel("Communication Round", fontsize=9.5)
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.set_xlim(xlim)

        def plot_method_tv(df, label, color, marker_val, ls_val="-"):
            if df is None:
                return
            v_acc = df["accuracy"]
            best_rnd = int(df.loc[v_acc.idxmax(), "round"])
            peak_val = v_acc.max()
            
            has_train = "training_accuracy" in df.columns
            if has_train:
                t_acc = df["training_accuracy"]
                peak_tr = t_acc.max()
                lbl = f"{label} (Val {peak_val*100:.1f}%, Tr {peak_tr*100:.1f}%)"
            else:
                lbl = f"{label} (Val {peak_val*100:.1f}%)"

            ax.plot(df["round"], v_acc, label=lbl, color=color, linestyle=ls_val, marker=marker_val, markersize=4.2, linewidth=1.8, zorder=4)
            ax.scatter(best_rnd, peak_val, marker="*", s=140, color=color, edgecolors="black", linewidth=0.5, zorder=6)

            if has_train:
                ax.plot(df["round"], t_acc, color=color, linestyle="--", marker="s", markersize=3.2, linewidth=1.4, alpha=0.85, zorder=3)
                ax.fill_between(df["round"], v_acc, t_acc, color=color, alpha=0.08, zorder=2)

        # FDA E3TO20
        plot_method_tv(df_csv_e20, "FDA E3TO20", c_fda20_acc, "o")

        # FDA E3TO30
        plot_method_tv(df_csv_e30, "FDA E3TO30", c_fda30_acc, "^")

        # FDA E3TO40
        plot_method_tv(df_csv_e40, "FDA E3TO40", c_fda40_acc, "D")

        # Old FDA runs
        for orun in old_runs:
            plot_method_tv(orun["df"], orun["label"], orun["c_acc"], orun["marker"], ls_val=orun["ls"])

        # RoLoRA
        plot_method_tv(df_csv_ro20, "RoLoRA E20", c_ro20_acc, "v")
        plot_method_tv(df_csv_ro20_rk8, "RoLoRA E20 RK8", c_ro20_rk8_acc, "v", ls_val="--")
        plot_method_tv(df_csv_ro15, "RoLoRA E15", c_ro15_acc, ">")
        plot_method_tv(df_csv_ro10, "RoLoRA E10", c_ro10_acc, "s")
        plot_method_tv(df_csv_ro5, "RoLoRA E5", c_ro5_acc, "x", ls_val="--")

        # LoRA & FFA-LoRA
        if include_lora:
            plot_method_tv(df_csv_lo20, "LoRA E20", c_lora20_acc, "<")
            plot_method_tv(df_csv_lo20_rk8, "LoRA E20 RK8", c_lora20_rk8_acc, "<", ls_val="--")
            plot_method_tv(df_csv_lo10, "LoRA E10", c_lora10_acc, "D")
            plot_method_tv(df_csv_lo5, "LoRA E5", c_lora5_acc, "v", ls_val="--")
            plot_method_tv(df_csv_ffa10, "FFA-LoRA E10", c_ffa10_acc, "h")
            plot_method_tv(df_csv_ffa5, "FFA-LoRA E5", c_ffa5_acc, "p", ls_val="--")

        ax.plot([], [], color="#222222", linestyle="-", linewidth=2.0, label="[Solid: Validation Acc]")
        ax.plot([], [], color="#222222", linestyle="--", linewidth=1.5, label="[Dashed: Train Acc]")

        ax.legend(fontsize=6.8 if is_standalone else 6.0, loc="best", ncol=2, framealpha=0.92, facecolor="white", edgecolor="#cccccc")

    # =============================================================
    # Render Standalone 1-Panel Plots (1 through 7)
    # =============================================================
    # 1. Accuracy vs Round
    fig, ax = plt.subplots(figsize=(7.5, 4.8), dpi=180)
    draw_acc(ax, is_standalone=True)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{cfg_prefix}_1_accuracy.png"), dpi=180)
    plt.close()

    # 2. Allocated Epochs vs Round
    fig, ax = plt.subplots(figsize=(7.5, 4.8), dpi=180)
    draw_ep(ax, is_standalone=True)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{cfg_prefix}_2_allocated_epochs.png"), dpi=180)
    plt.close()

    # 3. Client Drift Variance vs Round
    fig, ax = plt.subplots(figsize=(7.5, 4.8), dpi=180)
    draw_var(ax, is_standalone=True)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{cfg_prefix}_3_client_variance.png"), dpi=180)
    plt.close()

    # 4. Accuracy vs Communication Cost (MB)
    fig, ax = plt.subplots(figsize=(7.5, 4.8), dpi=180)
    draw_acc_vs_cost(ax, is_standalone=True)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{cfg_prefix}_4_accuracy_vs_cost.png"), dpi=180)
    plt.close()

    # 5. FDA Threshold per Matrix
    fig, ax = plt.subplots(figsize=(7.5, 4.8), dpi=180)
    draw_fda_threshold(ax, is_standalone=True)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{cfg_prefix}_5_fda_threshold.png"), dpi=180)
    plt.close()

    # 6. Variance vs Cumulative Local Epochs
    fig, ax = plt.subplots(figsize=(7.5, 4.8), dpi=180)
    draw_var_vs_epoch(ax, is_standalone=True)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{cfg_prefix}_6_variance_vs_epoch.png"), dpi=180)
    plt.close()

    # 7. Train vs. Validation Accuracy Gap
    fig, ax = plt.subplots(figsize=(8.0, 5.0), dpi=180)
    draw_train_val_gap(ax, is_standalone=True)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{cfg_prefix}_7_train_val_comparison.png"), dpi=180)
    plt.savefig(os.path.join(out_dir, f"{cfg_prefix}_7_train_accuracy.png"), dpi=180)
    plt.close()

    # 8. Inter-Client Drift Variance [Log-Scale] vs. Round (matching fda_opt.pdf)
    fig, ax = plt.subplots(figsize=(7.8, 5.0), dpi=180)
    draw_var_log(ax, is_standalone=True)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{cfg_prefix}_8_log_client_variance.png"), dpi=180)
    plt.close()

    # -------------------------------------------------------------
    # 7. Standalone 3-Panel Stacked Plot (Acc, Epochs, Var vs Round)
    # -------------------------------------------------------------
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(8.0, 11.5), dpi=180,
                                        gridspec_kw={'height_ratios': [1.1, 0.95, 0.85]})
    draw_acc(ax1, is_standalone=False)
    draw_ep(ax2, is_standalone=False)
    draw_var(ax3, is_standalone=False)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{cfg_prefix}_all3_telemetry_stacked.png"), dpi=180)
    plt.close()

    # -------------------------------------------------------------
    # 8. Comprehensive 6-Panel (3x2) Grid Plot
    # -------------------------------------------------------------
    fig, axes = plt.subplots(3, 2, figsize=(16.0, 13.5), dpi=180)
    draw_acc(axes[0, 0], is_standalone=False)
    draw_acc_vs_cost(axes[0, 1], is_standalone=False)
    draw_ep(axes[1, 0], is_standalone=False)
    draw_fda_threshold(axes[1, 1], is_standalone=False)
    draw_var(axes[2, 0], is_standalone=False)
    draw_var_vs_epoch(axes[2, 1], is_standalone=False)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{cfg_prefix}_all6_telemetry_stacked.png"), dpi=180)
    plt.close()

def main():
    base_dir = "figures/individual_plots"
    dir_rolora = os.path.join(base_dir, "fda_vs_rolora")
    dir_master = os.path.join(base_dir, "master_all_methods")
    
    os.makedirs(dir_rolora, exist_ok=True)
    os.makedirs(dir_master, exist_ok=True)

    print("Generating individual plots for Section 2: FDA vs. RoLoRA Baselines...")
    for cfg in all_configs:
        cfg_prefix, display_name, ds, c, r, lr = cfg
        print(f"  Rendering {cfg_prefix} (FDA vs RoLoRA)...")
        render_config_plots(cfg_prefix, display_name, ds, c, r, lr, include_lora=False, out_dir=dir_rolora)

    print("Generating individual plots for Section 1: Master All Methods (FDA, RoLoRA, LoRA, FFA-LoRA)...")
    for cfg in all_configs:
        cfg_prefix, display_name, ds, c, r, lr = cfg
        print(f"  Rendering {cfg_prefix} (All Methods)...")
        render_config_plots(cfg_prefix, display_name, ds, c, r, lr, include_lora=True, out_dir=dir_master)

    print("All individual and composite plots successfully exported!")

    try:
        from export_standalone_figures import export_all
        print("Syncing standalone figures to figures/all_individual_figures...")
        export_all("figures/all_individual_figures")
    except Exception as e:
        print(f"Note: Could not sync standalone figures: {e}")

if __name__ == "__main__":
    main()
