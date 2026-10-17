import os
import glob
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Load evaluation payload
payload_path = "new_llm_evaluation_payload.json"
with open(payload_path, "r", encoding="utf-8") as f:
    data = json.load(f)

runs = {r["run_id"]: r for r in data["runs"]}

tier_configs = [
    # Tier 1 (Row 0)
    [
        ("01_MNLI_C20_R50", "MNLI", "mnli", 20, 50),
        ("02_MNLI_C50_R30", "MNLI", "mnli", 50, 30),
        ("03_QNLI_C20_R50", "QNLI", "qnli", 20, 50),
        ("04_QNLI_C50_R30", "QNLI", "qnli", 50, 30),
    ],
    # Tier 2 (Row 1)
    [
        ("05_QQP_C20_R50",  "QQP",  "qqp",  20, 50),
        ("06_QQP_C50_R30",  "QQP",  "qqp",  50, 30),
        ("07_SST2_C20_R50", "SST-2", "sst2", 20, 50),
        ("08_SST2_C50_R30", "SST-2", "sst2", 50, 30),
    ],
    # Tier 3 (Row 2)
    [
        ("09_MRPC_C20_R30", "MRPC", "mrpc", 20, 30),
        ("10_MRPC_C50_R12", "MRPC", "mrpc", 50, 12),
        ("11_RTE_C20_R30",  "RTE",  "rte",  20, 30),
        ("12_RTE_C50_R12",  "RTE",  "rte",  50, 12),
    ],
]

# Color schemes
c_fda20_acc = "#1f77b4"
c_fda20_odd  = "#1e40af"
c_fda20_even = "#60a5fa"

c_fda30_acc = "#7b1fa2"
c_fda30_odd  = "#581c87"
c_fda30_even = "#c084fc"

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

c_ro20_acc   = "#9a3412"
c_ro20_rk8_acc = "#c2410c"
c_ro20_ep    = "#9a3412"
c_lora20_acc = "#15803d"
c_lora20_rk8_acc = "#166534"
c_lora20_ep  = "#15803d"
c_lora20_var = "#14532d"

comm_rates = {
    (20, False): 4.6875,
    (20, True): 9.375,
    (50, False): 11.71875,
    (50, True): 23.4375,
}

lr_map = {
    "mnli": "0.1", "qnli": "0.1", "qqp": "0.1",
    "sst2": "0.1", "mrpc": "0.05", "rte": "0.05",
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
        
        ep = float(np.mean([df["round_epochs"].iloc[r_idx] for df in active_dfs]))
        var = float(np.mean([df["variance"].iloc[r_idx] for df in active_dfs]))
        
        row = {
            "round": rnd,
            "active_matrix": act_mat,
            "accuracy": acc,
            "round_epochs": ep,
            "variance": var,
        }
        thetas = [df["theta_phase"].iloc[r_idx] for df in active_dfs if "theta_phase" in df.columns]
        if thetas:
            row["theta_phase"] = float(np.mean(thetas))
        records.append(row)
    return pd.DataFrame(records)

def load_fda_csv(ds, c, r, e_str):
    pat = f"results/round_metrics/*CALIB-DS-{ds}-C{c}-R{r}*{e_str}*CRATIO1*BDISC3*.csv"
    matches = sorted(glob.glob(pat))
    if matches:
        dfs = [prep(pd.read_csv(f)) for f in matches]
        return average_runs(dfs)
    return None

def load_baseline_csv(ds, c, r, method, e_str):
    pat = f"results/round_metrics/*MAIN-DS-{ds}-C{c}-R{r}-M-{method}-LR*-RK4-S*-{e_str}*.csv"
    matches = sorted(glob.glob(pat))
    if matches:
        dfs = [prep(pd.read_csv(f)) for f in matches]
        return average_runs(dfs)
    return None

def load_baseline_e20_csv(ds, c, r, method, rk=4):
    lr = lr_map[ds]
    pat1 = f"results/round_metrics/*MAIN-DS-{ds}-C{c}-*-M-{method}-LR{lr}-RK{rk}-S*-E20*.csv"
    pat2 = f"results/round_metrics/*MAIN-DS-{ds}-C{c}-*-M-{method}-LR{lr}-RK{rk}-S*_[0-9a-f]*.csv"
    matches = sorted(list(set(glob.glob(pat1) + glob.glob(pat2))))
    valid = []
    for m in matches:
        if "-E5-" in m or "-E10-" in m:
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

def load_old_fda_csv(ds, c, r, e_str="E3TO20", rk=4):
    lr = lr_map[ds]
    pat = f"results/round_metrics/*THESIS-RBASE-MAIN-DS-{ds}-C{c}-R{r}-M-fda-standard-LR{lr}-RK{rk}-{e_str}-S*-*.csv"
    matches = sorted(glob.glob(pat))
    if matches:
        dfs = [prep(pd.read_csv(f)) for f in matches]
        return average_runs(dfs)
    return None

def get_old_fda_runs(ds, c, r):
    runs_found = []
    df_20 = load_old_fda_csv(ds, c, r, "E3TO20", 4)
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
    df_30 = load_old_fda_csv(ds, c, r, "E3TO30", 4)
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
    df_rk8 = load_old_fda_csv(ds, c, r, "E3TO30", 8)
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

def draw_panel_acc_vs_cost(ax, display_name, ds, c, r, include_lora):
    rate_fda_ro = comm_rates[(c, False)]
    rate_lora = comm_rates[(c, True)]

    ax.set_title(f"{display_name} (C={c}, R={r}, lr={lr_map[ds]})", fontsize=11, fontweight="bold")
    ax.set_ylabel("Accuracy", fontsize=9.5)
    ax.set_xlabel("Bi-dir Cost (MB)", fontsize=9.5)
    ax.grid(True, linestyle="--", alpha=0.4)

    # FDA CRATIO E3TO20
    df_e20 = load_fda_csv(ds, c, r, "E3TO20")
    if df_e20 is not None:
        acc = df_e20["accuracy"]
        cost_x = df_e20["round"] * rate_fda_ro
        best_rnd = int(df_e20.loc[acc.idxmax(), "round"])
        peak_acc = acc.max()
        best_cost = best_rnd * rate_fda_ro
        ax.plot(cost_x, acc, label=f"FDA E3TO20 ({peak_acc*100:.1f}%)", color=c_fda20_acc, linestyle="-", marker="o", markersize=3.8, linewidth=1.8, zorder=4)
        ax.scatter(best_cost, peak_acc, marker="*", s=140, color="#0047AB", edgecolors="black", linewidth=0.5, zorder=7)

    # FDA CRATIO E3TO30
    df_e30 = load_fda_csv(ds, c, r, "E3TO30")
    if df_e30 is not None:
        acc = df_e30["accuracy"]
        cost_x = df_e30["round"] * rate_fda_ro
        best_rnd = int(df_e30.loc[acc.idxmax(), "round"])
        peak_acc = acc.max()
        best_cost = best_rnd * rate_fda_ro
        ax.plot(cost_x, acc, label=f"FDA E3TO30 ({peak_acc*100:.1f}%)", color=c_fda30_acc, linestyle="-", marker="^", markersize=4.2, linewidth=2.0, zorder=5)
        ax.scatter(best_cost, peak_acc, marker="*", s=150, color="#4a148c", edgecolors="gold", linewidth=0.8, zorder=8)

    # Old FDA runs
    old_runs = get_old_fda_runs(ds, c, r)
    for orun in old_runs:
        df_old = orun["df"]
        acc = df_old["accuracy"]
        cost_x = df_old["round"] * rate_fda_ro
        best_rnd = int(df_old.loc[acc.idxmax(), "round"])
        peak_acc = acc.max()
        best_cost = best_rnd * rate_fda_ro
        ax.plot(cost_x, acc, label=f"{orun['label']} ({peak_acc*100:.1f}%)", color=orun["c_acc"], linestyle=orun["ls"], marker=orun["marker"], markersize=3.8, linewidth=1.6, zorder=4)
        ax.scatter(best_cost, peak_acc, marker="*", s=130, color=orun["c_odd"], edgecolors="black", linewidth=0.5, zorder=7)

    # Baselines
    df_ro20 = load_baseline_e20_csv(ds, c, r, "rolora", 4)
    if df_ro20 is not None:
        acc = df_ro20["accuracy"]
        cost_x = df_ro20["round"] * rate_fda_ro
        best_rnd = int(df_ro20.loc[acc.idxmax(), "round"])
        peak_acc = acc.max()
        best_cost = best_rnd * rate_fda_ro
        ax.plot(cost_x, acc, label=f"RoLoRA E20 ({peak_acc*100:.1f}%)", color=c_ro20_acc, linestyle="-", marker="v", markersize=3.5, linewidth=1.5, zorder=3)
        ax.scatter(best_cost, peak_acc, marker="*", s=130, color="#78350f", edgecolors="black", linewidth=0.5, zorder=6)

    df_ro20_rk8 = load_baseline_e20_csv(ds, c, r, "rolora", 8)
    if df_ro20_rk8 is not None:
        acc = df_ro20_rk8["accuracy"]
        cost_x = df_ro20_rk8["round"] * rate_fda_ro
        peak_acc = acc.max()
        ax.plot(cost_x, acc, label=f"RoLoRA E20 RK8 ({peak_acc*100:.1f}%)", color=c_ro20_rk8_acc, linestyle="--", marker="v", markersize=3.3, linewidth=1.3, zorder=3)

    df_ro10 = load_baseline_csv(ds, c, r, "rolora", "E10")
    if df_ro10 is not None:
        acc = df_ro10["accuracy"]
        cost_x = df_ro10["round"] * rate_fda_ro
        best_rnd = int(df_ro10.loc[acc.idxmax(), "round"])
        peak_acc = acc.max()
        best_cost = best_rnd * rate_fda_ro
        ax.plot(cost_x, acc, label=f"RoLoRA E10 ({peak_acc*100:.1f}%)", color=c_ro10_acc, linestyle="-", marker="s", markersize=3.5, linewidth=1.5, zorder=3)
        ax.scatter(best_cost, peak_acc, marker="*", s=130, color="#b23b00", edgecolors="black", linewidth=0.5, zorder=6)

    df_ro5 = load_baseline_csv(ds, c, r, "rolora", "E5")
    if df_ro5 is not None:
        acc = df_ro5["accuracy"]
        cost_x = df_ro5["round"] * rate_fda_ro
        peak_acc = acc.max()
        ax.plot(cost_x, acc, label=f"RoLoRA E5 ({peak_acc*100:.1f}%)", color=c_ro5_acc, linestyle="--", marker="x", markersize=3.8, linewidth=1.4, zorder=2)

    if include_lora:
        df_lo20 = load_baseline_e20_csv(ds, c, r, "lora", 4)
        if df_lo20 is not None:
            acc = df_lo20["accuracy"]
            cost_x = df_lo20["round"] * rate_lora
            peak_acc = acc.max()
            ax.plot(cost_x, acc, label=f"LoRA E20 ({peak_acc*100:.1f}%)", color=c_lora20_acc, linestyle="-", marker="<", markersize=3.2, linewidth=1.4, zorder=3)

        df_lo20_rk8 = load_baseline_e20_csv(ds, c, r, "lora", 8)
        if df_lo20_rk8 is not None:
            acc = df_lo20_rk8["accuracy"]
            cost_x = df_lo20_rk8["round"] * rate_lora
            peak_acc = acc.max()
            ax.plot(cost_x, acc, label=f"LoRA E20 RK8 ({peak_acc*100:.1f}%)", color=c_lora20_rk8_acc, linestyle="--", marker="<", markersize=3.0, linewidth=1.2, zorder=2)

        df_lo10 = load_baseline_csv(ds, c, r, "lora", "E10")
        if df_lo10 is not None:
            acc = df_lo10["accuracy"]
            cost_x = df_lo10["round"] * rate_lora
            peak_acc = acc.max()
            ax.plot(cost_x, acc, label=f"LoRA E10 ({peak_acc*100:.1f}%)", color=c_lora10_acc, linestyle="-", marker="D", markersize=3.2, linewidth=1.5, zorder=3)

        df_lo5 = load_baseline_csv(ds, c, r, "lora", "E5")
        if df_lo5 is not None:
            acc = df_lo5["accuracy"]
            cost_x = df_lo5["round"] * rate_lora
            peak_acc = acc.max()
            ax.plot(cost_x, acc, label=f"LoRA E5 ({peak_acc*100:.1f}%)", color=c_lora5_acc, linestyle="--", marker="v", markersize=3.4, linewidth=1.3, zorder=2)

    ax.legend(fontsize=6.8, loc="lower right", framealpha=0.85, facecolor="white", edgecolor="#cccccc")

def draw_panel_fda_threshold(ax, display_name, ds, c, r):
    xlim = (1, 12) if r == 12 else (0, r)
    df_csv_e20 = load_fda_csv(ds, c, r, "E3TO20")
    df_csv_e30 = load_fda_csv(ds, c, r, "E3TO30")

    ax.set_title(f"{display_name} (C={c}, R={r}, lr={lr_map[ds]})", fontsize=11, fontweight="bold")
    ax.set_ylabel(r"Threshold ($\theta$)", fontsize=9.5)
    ax.set_xlabel("Communication Round", fontsize=9.5)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.set_xlim(xlim)

    plotted = False
    if df_csv_e20 is not None:
        df_a20 = df_csv_e20[df_csv_e20["active_matrix"] == "A"]
        df_b20 = df_csv_e20[df_csv_e20["active_matrix"] == "B"]
        if not df_a20.empty:
            ax.plot(df_a20["round"], df_a20["theta_phase"], label=r"E3TO20: $\theta_A$", color=c_fda20_odd, linestyle="-", marker="o", markersize=4.0, linewidth=1.6, zorder=4)
            plotted = True
        if not df_b20.empty:
            ax.plot(df_b20["round"], df_b20["theta_phase"], label=r"E3TO20: $\theta_B$", color=c_fda20_even, linestyle="-", marker="s", markersize=4.0, linewidth=1.6, zorder=4)
            plotted = True

    if df_csv_e30 is not None:
        df_a30 = df_csv_e30[df_csv_e30["active_matrix"] == "A"]
        df_b30 = df_csv_e30[df_csv_e30["active_matrix"] == "B"]
        if not df_a30.empty:
            ax.plot(df_a30["round"], df_a30["theta_phase"], label=r"E3TO30: $\theta_A$", color=c_fda30_odd, linestyle="-", marker="^", markersize=4.2, linewidth=1.6, zorder=5)
            plotted = True
        if not df_b30.empty:
            ax.plot(df_b30["round"], df_b30["theta_phase"], label=r"E3TO30: $\theta_B$", color=c_fda30_even, linestyle="-", marker="d", markersize=4.2, linewidth=1.6, zorder=5)
            plotted = True

    # Old FDA runs thresholds
    old_runs = get_old_fda_runs(ds, c, r)
    for orun in old_runs:
        df_old = orun["df"]
        df_a = df_old[df_old["active_matrix"] == "A"]
        df_b = df_old[df_old["active_matrix"] == "B"]
        lbl_tag = orun["label"].replace("FDA ", "")
        if not df_a.empty:
            ax.plot(df_a["round"], df_a["theta_phase"], label=rf"{lbl_tag}: $\theta_A$", color=orun["c_odd"], linestyle="-.", marker=orun["marker"], markersize=3.8, linewidth=1.5, zorder=4)
            plotted = True
        if not df_b.empty:
            ax.plot(df_b["round"], df_b["theta_phase"], label=rf"{lbl_tag}: $\theta_B$", color=orun["c_even"], linestyle=":", marker=orun["marker"], markersize=3.8, linewidth=1.5, zorder=4)
            plotted = True

    if plotted:
        ax.legend(fontsize=6.8, loc="upper left", framealpha=0.85, facecolor="white", edgecolor="#cccccc")

def draw_panel_var_vs_epoch(ax, display_name, ds, c, r, include_lora):
    df_csv_e20 = load_fda_csv(ds, c, r, "E3TO20")
    df_csv_e30 = load_fda_csv(ds, c, r, "E3TO30")
    df_csv_ro10 = load_baseline_csv(ds, c, r, "rolora", "E10")
    df_csv_ro5 = load_baseline_csv(ds, c, r, "rolora", "E5")
    df_csv_lo10 = load_baseline_csv(ds, c, r, "lora", "E10") if include_lora else None
    df_csv_lo5 = load_baseline_csv(ds, c, r, "lora", "E5") if include_lora else None

    ax.set_title(f"{display_name} (C={c}, R={r}, lr={lr_map[ds]})", fontsize=11, fontweight="bold")
    ax.set_ylabel(r"Variance ($\sigma^2$)", fontsize=9.5)
    ax.set_xlabel("Cumulative Local Epochs", fontsize=9.5)
    ax.grid(True, linestyle="--", alpha=0.4)

    # FDA E3TO30
    if df_csv_e30 is not None:
        cum_eps = np.cumsum(df_csv_e30['round_epochs'].values)
        mask_A = (df_csv_e30['active_matrix'] == 'A').values
        mask_B = (df_csv_e30['active_matrix'] == 'B').values
        ax.plot(cum_eps[mask_A], df_csv_e30['variance'].values[mask_A], label="FDA E3TO30 (A)", color=c_fda30_odd, linestyle="-", marker="^", markersize=3.6, linewidth=1.6, alpha=0.9, zorder=5)
        ax.plot(cum_eps[mask_B], df_csv_e30['variance'].values[mask_B], label="FDA E3TO30 (B)", color=c_fda30_even, linestyle="--", marker="^", markersize=3.4, linewidth=1.4, alpha=0.9, zorder=5)

    # FDA E3TO20
    if df_csv_e20 is not None:
        cum_eps = np.cumsum(df_csv_e20['round_epochs'].values)
        mask_A = (df_csv_e20['active_matrix'] == 'A').values
        mask_B = (df_csv_e20['active_matrix'] == 'B').values
        ax.plot(cum_eps[mask_A], df_csv_e20['variance'].values[mask_A], label="FDA E3TO20 (A)", color=c_fda20_odd, linestyle="-", marker="o", markersize=3.5, linewidth=1.6, alpha=0.9, zorder=4)
        ax.plot(cum_eps[mask_B], df_csv_e20['variance'].values[mask_B], label="FDA E3TO20 (B)", color=c_fda20_even, linestyle="--", marker="o", markersize=3.3, linewidth=1.4, alpha=0.9, zorder=4)

    # Old FDA runs
    old_runs = get_old_fda_runs(ds, c, r)
    for orun in old_runs:
        df_old = orun["df"]
        cum_eps = np.cumsum(df_old['round_epochs'].values)
        mask_A = (df_old['active_matrix'] == 'A').values
        mask_B = (df_old['active_matrix'] == 'B').values
        lbl_tag = orun["label"].replace("FDA ", "")
        ax.plot(cum_eps[mask_A], df_old['variance'].values[mask_A], label=f"{lbl_tag} (A)", color=orun["c_odd"], linestyle="-.", marker=orun["marker"], markersize=3.5, linewidth=1.5, alpha=0.9, zorder=4)
        ax.plot(cum_eps[mask_B], df_old['variance'].values[mask_B], label=f"{lbl_tag} (B)", color=orun["c_even"], linestyle=":", marker=orun["marker"], markersize=3.3, linewidth=1.3, alpha=0.9, zorder=4)

    # RoLoRA E20
    df_csv_ro20 = load_baseline_e20_csv(ds, c, r, "rolora", 4)
    if df_csv_ro20 is not None:
        cum_eps = np.cumsum(df_csv_ro20['round_epochs'].values)
        mask_A = (df_csv_ro20['active_matrix'] == 'A').values
        mask_B = (df_csv_ro20['active_matrix'] == 'B').values
        ax.plot(cum_eps[mask_A], df_csv_ro20['variance'].values[mask_A], label="RoLoRA E20 (A)", color="#78350f", linestyle="-", marker="v", markersize=3.6, linewidth=1.5, alpha=0.85, zorder=3)
        ax.plot(cum_eps[mask_B], df_csv_ro20['variance'].values[mask_B], label="RoLoRA E20 (B)", color="#d97706", linestyle="--", marker="v", markersize=3.4, linewidth=1.3, alpha=0.85, zorder=3)

    # RoLoRA E10
    if df_csv_ro10 is not None:
        cum_eps = np.cumsum(df_csv_ro10['round_epochs'].values)
        mask_A = (df_csv_ro10['active_matrix'] == 'A').values
        mask_B = (df_csv_ro10['active_matrix'] == 'B').values
        ax.plot(cum_eps[mask_A], df_csv_ro10['variance'].values[mask_A], label="RoLoRA E10 (A)", color="#c2410c", linestyle="-", marker="s", markersize=3.5, linewidth=1.5, alpha=0.85, zorder=3)
        ax.plot(cum_eps[mask_B], df_csv_ro10['variance'].values[mask_B], label="RoLoRA E10 (B)", color="#fb923c", linestyle="--", marker="s", markersize=3.3, linewidth=1.3, alpha=0.85, zorder=3)

    # RoLoRA E5
    if df_csv_ro5 is not None:
        cum_eps = np.cumsum(df_csv_ro5['round_epochs'].values)
        mask_A = (df_csv_ro5['active_matrix'] == 'A').values
        mask_B = (df_csv_ro5['active_matrix'] == 'B').values
        ax.plot(cum_eps[mask_A], df_csv_ro5['variance'].values[mask_A], label="RoLoRA E5 (A)", color="#991b1b", linestyle="-", marker="x", markersize=3.6, linewidth=1.4, alpha=0.85, zorder=2)
        ax.plot(cum_eps[mask_B], df_csv_ro5['variance'].values[mask_B], label="RoLoRA E5 (B)", color="#f87171", linestyle="--", marker="x", markersize=3.4, linewidth=1.2, alpha=0.85, zorder=2)

    # LoRA E20, E10 & E5
    if include_lora:
        df_csv_lo20 = load_baseline_e20_csv(ds, c, r, "lora", 4)
        if df_csv_lo20 is not None:
            cum_eps = np.cumsum(df_csv_lo20['round_epochs'].values)
            ax.plot(cum_eps, df_csv_lo20['variance'].values, label="LoRA E20 (Joint)", color=c_lora20_var, linestyle="-", marker="<", markersize=3.2, linewidth=1.4, alpha=0.85, zorder=3)
        if df_csv_lo10 is not None:
            cum_eps = np.cumsum(df_csv_lo10['round_epochs'].values)
            ax.plot(cum_eps, df_csv_lo10['variance'].values, label="LoRA E10 (Joint)", color=c_lora10_var, linestyle="-", marker="D", markersize=3.2, linewidth=1.4, alpha=0.85, zorder=3)
        if df_csv_lo5 is not None:
            cum_eps = np.cumsum(df_csv_lo5['round_epochs'].values)
            ax.plot(cum_eps, df_csv_lo5['variance'].values, label="LoRA E5 (Joint)", color=c_lora5_var, linestyle=":", marker="v", markersize=3.2, linewidth=1.4, alpha=0.85, zorder=2)

    ax.legend(ncol=2, fontsize=6.5, loc="upper left", framealpha=0.85, facecolor="white", edgecolor="#cccccc")

def generate_master_grid(kind, include_lora, out_path, title_main):
    fig, axes = plt.subplots(3, 4, figsize=(22.0, 14.5), dpi=180)
    fig.suptitle(title_main, fontsize=15, fontweight="bold", y=0.995)

    for row_idx, tier in enumerate(tier_configs):
        for col_idx, cfg in enumerate(tier):
            cfg_prefix, display_name, ds, c, r = cfg
            ax = axes[row_idx, col_idx]
            if kind == "acc_vs_cost":
                draw_panel_acc_vs_cost(ax, display_name, ds, c, r, include_lora)
            elif kind == "fda_threshold":
                draw_panel_fda_threshold(ax, display_name, ds, c, r)
            elif kind == "var_vs_epoch":
                draw_panel_var_vs_epoch(ax, display_name, ds, c, r, include_lora)

    plt.tight_layout()
    plt.subplots_adjust(top=0.96)
    plt.savefig(out_path, dpi=180)
    plt.close()
    print(f"Generated: {out_path}")

# ==============================================================================
# Helper to draw Accuracy vs Round for a single panel
# ==============================================================================
def draw_panel_acc(ax, display_name, ds, c, r, include_lora):
    xlim = (1, 12) if r == 12 else (0, r)
    ax.set_title(f"{display_name} (C={c}, R={r}, lr={lr_map[ds]}) — Accuracy", fontsize=10.5, fontweight="bold")
    ax.set_ylabel("Accuracy", fontsize=9.5)
    ax.set_xlabel("Communication Round", fontsize=9.5)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.set_xlim(xlim)

    # FDA CRATIO E3TO20
    df_e20 = load_fda_csv(ds, c, r, "E3TO20")
    if df_e20 is not None:
        acc = df_e20["accuracy"]
        best_rnd = int(df_e20.loc[acc.idxmax(), "round"])
        peak = acc.max()
        ax.plot(df_e20["round"], acc, label=f"FDA E3TO20 ({peak*100:.1f}%)", color=c_fda20_acc, linestyle="-", marker="o", markersize=3.8, linewidth=1.8, zorder=4)
        ax.scatter(best_rnd, peak, marker="*", s=140, color="#0047AB", edgecolors="black", linewidth=0.5, zorder=7)

    # FDA CRATIO E3TO30
    df_e30 = load_fda_csv(ds, c, r, "E3TO30")
    if df_e30 is not None:
        acc = df_e30["accuracy"]
        best_rnd = int(df_e30.loc[acc.idxmax(), "round"])
        peak = acc.max()
        ax.plot(df_e30["round"], acc, label=f"FDA E3TO30 ({peak*100:.1f}%)", color=c_fda30_acc, linestyle="-", marker="^", markersize=4.2, linewidth=2.0, zorder=5)
        ax.scatter(best_rnd, peak, marker="*", s=150, color="#4a148c", edgecolors="gold", linewidth=0.8, zorder=8)

    # Old FDA runs
    old_runs = get_old_fda_runs(ds, c, r)
    for orun in old_runs:
        df_old = orun["df"]
        acc = df_old["accuracy"]
        best_rnd = int(df_old.loc[acc.idxmax(), "round"])
        peak = acc.max()
        ax.plot(df_old["round"], acc, label=f"{orun['label']} ({peak*100:.1f}%)", color=orun["c_acc"], linestyle=orun["ls"], marker=orun["marker"], markersize=3.8, linewidth=1.6, zorder=4)
        ax.scatter(best_rnd, peak, marker="*", s=130, color=orun["c_odd"], edgecolors="black", linewidth=0.5, zorder=7)

    # RoLoRA
    df_ro20 = load_baseline_e20_csv(ds, c, r, "rolora", 4)
    if df_ro20 is not None:
        acc = df_ro20["accuracy"]
        best_rnd = int(df_ro20.loc[acc.idxmax(), "round"])
        peak = acc.max()
        ax.plot(df_ro20["round"], acc, label=f"RoLoRA E20 ({peak*100:.1f}%)", color=c_ro20_acc, linestyle="-", marker="v", markersize=3.5, linewidth=1.5, zorder=3)
        ax.scatter(best_rnd, peak, marker="*", s=130, color="#78350f", edgecolors="black", linewidth=0.5, zorder=6)

    df_ro20_rk8 = load_baseline_e20_csv(ds, c, r, "rolora", 8)
    if df_ro20_rk8 is not None:
        acc = df_ro20_rk8["accuracy"]
        peak = acc.max()
        ax.plot(df_ro20_rk8["round"], acc, label=f"RoLoRA E20 RK8 ({peak*100:.1f}%)", color=c_ro20_rk8_acc, linestyle="--", marker="v", markersize=3.3, linewidth=1.3, zorder=3)

    df_ro10 = load_baseline_csv(ds, c, r, "rolora", "E10")
    if df_ro10 is not None:
        acc = df_ro10["accuracy"]
        best_rnd = int(df_ro10.loc[acc.idxmax(), "round"])
        peak = acc.max()
        ax.plot(df_ro10["round"], acc, label=f"RoLoRA E10 ({peak*100:.1f}%)", color=c_ro10_acc, linestyle="-", marker="s", markersize=3.5, linewidth=1.5, zorder=3)
        ax.scatter(best_rnd, peak, marker="*", s=130, color="#b23b00", edgecolors="black", linewidth=0.5, zorder=6)

    df_ro5 = load_baseline_csv(ds, c, r, "rolora", "E5")
    if df_ro5 is not None:
        acc = df_ro5["accuracy"]
        peak = acc.max()
        ax.plot(df_ro5["round"], acc, label=f"RoLoRA E5 ({peak*100:.1f}%)", color=c_ro5_acc, linestyle="--", marker="x", markersize=3.8, linewidth=1.4, zorder=2)

    # LoRA
    if include_lora:
        df_lo20 = load_baseline_e20_csv(ds, c, r, "lora", 4)
        if df_lo20 is not None:
            acc = df_lo20["accuracy"]
            peak = acc.max()
            ax.plot(df_lo20["round"], acc, label=f"LoRA E20 ({peak*100:.1f}%)", color=c_lora20_acc, linestyle="-", marker="<", markersize=3.2, linewidth=1.4, zorder=3)

        df_lo20_rk8 = load_baseline_e20_csv(ds, c, r, "lora", 8)
        if df_lo20_rk8 is not None:
            acc = df_lo20_rk8["accuracy"]
            peak = acc.max()
            ax.plot(df_lo20_rk8["round"], acc, label=f"LoRA E20 RK8 ({peak*100:.1f}%)", color=c_lora20_rk8_acc, linestyle="--", marker="<", markersize=3.0, linewidth=1.2, zorder=2)

        df_lo10 = load_baseline_csv(ds, c, r, "lora", "E10")
        if df_lo10 is not None:
            acc = df_lo10["accuracy"]
            peak = acc.max()
            ax.plot(df_lo10["round"], acc, label=f"LoRA E10 ({peak*100:.1f}%)", color=c_lora10_acc, linestyle="-", marker="D", markersize=3.2, linewidth=1.5, zorder=3)
        df_lo5 = load_baseline_csv(ds, c, r, "lora", "E5")
        if df_lo5 is not None:
            acc = df_lo5["accuracy"]
            peak = acc.max()
            ax.plot(df_lo5["round"], acc, label=f"LoRA E5 ({peak*100:.1f}%)", color=c_lora5_acc, linestyle="--", marker="v", markersize=3.4, linewidth=1.3, zorder=2)

    ax.legend(fontsize=6.8, loc="lower right", framealpha=0.85, facecolor="white", edgecolor="#cccccc")

# ==============================================================================
# Helper to draw Allocated Epochs for a single panel
# ==============================================================================
def draw_panel_ep(ax, display_name, ds, c, r, include_lora):
    xlim = (1, 12) if r == 12 else (0, r)
    ax.set_title(f"{display_name} (C={c}, R={r}, lr={lr_map[ds]}) — Allocated Local Epochs", fontsize=10.5, fontweight="bold")
    ax.set_ylabel("Local Epochs", fontsize=9.5)
    ax.set_xlabel("Communication Round", fontsize=9.5)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.set_xlim(xlim)

    fda_runs_bars = []
    df_e20 = load_fda_csv(ds, c, r, "E3TO20")
    if df_e20 is not None:
        fda_runs_bars.append(("FDA E3TO20", df_e20, c_fda20_odd, c_fda20_even))
    df_e30 = load_fda_csv(ds, c, r, "E3TO30")
    if df_e30 is not None:
        fda_runs_bars.append(("FDA E3TO30", df_e30, c_fda30_odd, c_fda30_even))

    old_runs = get_old_fda_runs(ds, c, r)
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

    # Baseline lines
    df_ro20 = load_baseline_e20_csv(ds, c, r, "rolora", 4)
    if df_ro20 is not None:
        ax.plot(df_ro20["round"], [20]*len(df_ro20), label="RoLoRA E20", color=c_ro20_ep, linestyle="-", linewidth=1.6, zorder=4)

    df_ro10 = load_baseline_csv(ds, c, r, "rolora", "E10")
    if df_ro10 is not None:
        ax.plot(df_ro10["round"], [10]*len(df_ro10), label="RoLoRA E10", color=c_ro10_ep, linestyle="--", linewidth=1.6, zorder=4)

    df_ro5 = load_baseline_csv(ds, c, r, "rolora", "E5")
    if df_ro5 is not None:
        ax.plot(df_ro5["round"], [5]*len(df_ro5), label="RoLoRA E5", color=c_ro5_ep, linestyle=":", linewidth=1.6, zorder=4)

    if include_lora:
        df_lo20 = load_baseline_e20_csv(ds, c, r, "lora", 4)
        if df_lo20 is not None:
            ax.plot(df_lo20["round"], [20.3]*len(df_lo20), label="LoRA E20", color=c_lora20_ep, linestyle="-.", linewidth=1.6, zorder=4)
        df_lo10 = load_baseline_csv(ds, c, r, "lora", "E10")
        if df_lo10 is not None:
            ax.plot(df_lo10["round"], [10.2]*len(df_lo10), label="LoRA E10", color=c_lora10_ep, linestyle="-.", linewidth=1.6, zorder=4)
        df_lo5 = load_baseline_csv(ds, c, r, "lora", "E5")
        if df_lo5 is not None:
            ax.plot(df_lo5["round"], [5.2]*len(df_lo5), label="LoRA E5", color=c_lora5_ep, linestyle="-.", linewidth=1.6, zorder=4)

    ax.legend(fontsize=6.5, loc="upper right", framealpha=0.85, facecolor="white", edgecolor="#cccccc")

# ==============================================================================
# Helper to draw Client Drift Variance for a single panel
# ==============================================================================
def draw_panel_var(ax, display_name, ds, c, r, include_lora):
    xlim = (1, 12) if r == 12 else (0, r)
    ax.set_title(f"{display_name} (C={c}, R={r}, lr={lr_map[ds]}) — Client Drift Variance", fontsize=10.5, fontweight="bold")
    ax.set_ylabel(r"Variance ($\sigma^2_t$)", fontsize=9.5)
    ax.set_xlabel("Communication Round", fontsize=9.5)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.set_xlim(xlim)

    # FDA E3TO30
    df_e30 = load_fda_csv(ds, c, r, "E3TO30")
    if df_e30 is not None:
        df_A = df_e30[df_e30['active_matrix'] == 'A']
        df_B = df_e30[df_e30['active_matrix'] == 'B']
        ax.plot(df_A['round'], df_A['variance'], label="FDA E3TO30 (A)", color=c_fda30_odd, linestyle="-", marker="^", markersize=4.2, linewidth=1.8, alpha=0.9, zorder=5)
        ax.plot(df_B['round'], df_B['variance'], label="FDA E3TO30 (B)", color=c_fda30_even, linestyle="--", marker="^", markersize=4.0, linewidth=1.6, alpha=0.9, zorder=5)

    # FDA E3TO20
    df_e20 = load_fda_csv(ds, c, r, "E3TO20")
    if df_e20 is not None:
        df_A = df_e20[df_e20['active_matrix'] == 'A']
        df_B = df_e20[df_e20['active_matrix'] == 'B']
        ax.plot(df_A['round'], df_A['variance'], label="FDA E3TO20 (A)", color=c_fda20_odd, linestyle="-", marker="o", markersize=4.0, linewidth=1.8, alpha=0.9, zorder=4)
        ax.plot(df_B['round'], df_B['variance'], label="FDA E3TO20 (B)", color=c_fda20_even, linestyle="--", marker="o", markersize=3.8, linewidth=1.6, alpha=0.9, zorder=4)

    # Old FDA runs
    old_runs = get_old_fda_runs(ds, c, r)
    for orun in old_runs:
        df_old = orun["df"]
        df_A = df_old[df_old['active_matrix'] == 'A']
        df_B = df_old[df_old['active_matrix'] == 'B']
        lbl_tag = orun["label"].replace("FDA ", "")
        ax.plot(df_A['round'], df_A['variance'], label=f"{lbl_tag} (A)", color=orun["c_odd"], linestyle="-.", marker=orun["marker"], markersize=3.8, linewidth=1.6, alpha=0.9, zorder=4)
        ax.plot(df_B['round'], df_B['variance'], label=f"{lbl_tag} (B)", color=orun["c_even"], linestyle=":", marker=orun["marker"], markersize=3.6, linewidth=1.4, alpha=0.9, zorder=4)

    # RoLoRA
    df_ro20 = load_baseline_e20_csv(ds, c, r, "rolora", 4)
    if df_ro20 is not None:
        df_A = df_ro20[df_ro20['active_matrix'] == 'A']
        df_B = df_ro20[df_ro20['active_matrix'] == 'B']
        ax.plot(df_A['round'], df_A['variance'], label="RoLoRA E20 (A)", color="#78350f", linestyle="-", marker="v", markersize=3.8, linewidth=1.5, alpha=0.85, zorder=3)
        ax.plot(df_B['round'], df_B['variance'], label="RoLoRA E20 (B)", color="#d97706", linestyle="--", marker="v", markersize=3.6, linewidth=1.4, alpha=0.85, zorder=3)

    df_ro10 = load_baseline_csv(ds, c, r, "rolora", "E10")
    if df_ro10 is not None:
        df_A = df_ro10[df_ro10['active_matrix'] == 'A']
        df_B = df_ro10[df_ro10['active_matrix'] == 'B']
        ax.plot(df_A['round'], df_A['variance'], label="RoLoRA E10 (A)", color="#c2410c", linestyle="-", marker="s", markersize=3.8, linewidth=1.6, alpha=0.85, zorder=3)
        ax.plot(df_B['round'], df_B['variance'], label="RoLoRA E10 (B)", color="#fb923c", linestyle="--", marker="s", markersize=3.6, linewidth=1.5, alpha=0.85, zorder=3)

    df_ro5 = load_baseline_csv(ds, c, r, "rolora", "E5")
    if df_ro5 is not None:
        df_A = df_ro5[df_ro5['active_matrix'] == 'A']
        df_B = df_ro5[df_ro5['active_matrix'] == 'B']
        ax.plot(df_A['round'], df_A['variance'], label="RoLoRA E5 (A)", color="#991b1b", linestyle="-", marker="x", markersize=4.0, linewidth=1.5, alpha=0.85, zorder=2)
        ax.plot(df_B['round'], df_B['variance'], label="RoLoRA E5 (B)", color="#f87171", linestyle="--", marker="x", markersize=3.8, linewidth=1.4, alpha=0.85, zorder=2)

    # LoRA
    if include_lora:
        df_lo20 = load_baseline_e20_csv(ds, c, r, "lora", 4)
        if df_lo20 is not None:
            ax.plot(df_lo20['round'], df_lo20['variance'], label="LoRA E20 (Joint)", color=c_lora20_var, linestyle="-", marker="<", markersize=3.2, linewidth=1.4, alpha=0.85, zorder=3)
        df_lo10 = load_baseline_csv(ds, c, r, "lora", "E10")
        if df_lo10 is not None:
            ax.plot(df_lo10['round'], df_lo10['variance'], label="LoRA E10 (Joint)", color=c_lora10_var, linestyle="-", marker="D", markersize=3.4, linewidth=1.5, alpha=0.85, zorder=3)
        df_lo5 = load_baseline_csv(ds, c, r, "lora", "E5")
        if df_lo5 is not None:
            ax.plot(df_lo5['round'], df_lo5['variance'], label="LoRA E5 (Joint)", color=c_lora5_var, linestyle=":", marker="v", markersize=3.4, linewidth=1.5, alpha=0.85, zorder=2)

    ax.legend(ncol=2, fontsize=6.2, loc="upper left", framealpha=0.85, facecolor="white", edgecolor="#cccccc")

# ==============================================================================
# Generate Master 9x4 Multi-Metric Stacked Grid Figure
# ==============================================================================
def generate_master_stacked_grid(include_lora, out_path, title_main):
    # 9 rows, 4 columns
    # Rows 0-2: Tier 1 (Acc, Ep, Var)
    # Rows 3-5: Tier 2 (Acc, Ep, Var)
    # Rows 6-8: Tier 3 (Acc, Ep, Var)
    fig, axes = plt.subplots(9, 4, figsize=(24.0, 36.0), dpi=180)
    fig.suptitle(title_main, fontsize=18, fontweight="bold", y=0.998)

    for tier_idx, tier in enumerate(tier_configs):
        r_acc = tier_idx * 3
        r_ep  = tier_idx * 3 + 1
        r_var = tier_idx * 3 + 2

        for col_idx, cfg in enumerate(tier):
            cfg_prefix, display_name, ds, c, r = cfg
            draw_panel_acc(axes[r_acc, col_idx], display_name, ds, c, r, include_lora)
            draw_panel_ep(axes[r_ep, col_idx], display_name, ds, c, r, include_lora)
            draw_panel_var(axes[r_var, col_idx], display_name, ds, c, r, include_lora)

    plt.tight_layout()
    plt.subplots_adjust(top=0.985)
    plt.savefig(out_path, dpi=180)
    plt.close()
    print(f"Generated Master Stacked Grid: {out_path}")

def main():
    os.makedirs("figures", exist_ok=True)

    print("Generating Master 9x4 Stacked Telemetry Grids...")
    generate_master_stacked_grid(include_lora=True,
                                 out_path="figures/master_telemetry_all_methods_acc_epochs_var_v2.png",
                                 title_main="Master Head-to-Head Convergence, Epoch Allocations & Client Variance Across All 12 Configurations")

    generate_master_stacked_grid(include_lora=False,
                                 out_path="figures/fda_vs_rolora_e5_e10_acc_epochs_var_v2.png",
                                 title_main="Architectural Convergence Curves, Local Epochs & Client Variance Across All 12 Configurations")

    print("\nGenerating Master Figures for Section 1 (Master All Methods)...")
    generate_master_grid("acc_vs_cost", include_lora=True, 
                         out_path="figures/master_telemetry_all_methods_acc_vs_cost.png",
                         title_main="Master Accuracy vs. Cumulative Communication Cost (MB, Bi-directional) Across All 12 Configurations")
    generate_master_grid("fda_threshold", include_lora=True, 
                         out_path="figures/master_telemetry_fda_thresholds.png",
                         title_main=r"FDA Closed-Loop Dynamic Thresholds ($\theta_A$ on Matrix A, $\theta_B$ on Matrix B) Across All 12 Configurations")
    generate_master_grid("var_vs_epoch", include_lora=True, 
                         out_path="figures/master_telemetry_all_methods_var_vs_epoch.png",
                         title_main=r"Master Client Drift Variance ($\sigma^2_t$) vs. Cumulative Local Epochs Across All 12 Configurations")

    print("\nGenerating Master Figures for Section 2 (FDA vs. RoLoRA Baselines)...")
    generate_master_grid("acc_vs_cost", include_lora=False, 
                         out_path="figures/fda_vs_rolora_acc_vs_cost.png",
                         title_main="Architectural Accuracy vs. Cumulative Communication Cost (MB, Bi-directional) Across All 12 Configurations")
    generate_master_grid("fda_threshold", include_lora=False, 
                         out_path="figures/fda_vs_rolora_thresholds.png",
                         title_main=r"Architectural FDA Dynamic Thresholds ($\theta_A$ on Matrix A, $\theta_B$ on Matrix B) Across All 12 Configurations")
    generate_master_grid("var_vs_epoch", include_lora=False, 
                         out_path="figures/fda_vs_rolora_var_vs_epoch.png",
                         title_main=r"Architectural Client Drift Variance ($\sigma^2_t$) vs. Cumulative Local Epochs Across All 12 Configurations")

    print("\nAll master expanded and stacked figures generated successfully!")

if __name__ == "__main__":
    main()
