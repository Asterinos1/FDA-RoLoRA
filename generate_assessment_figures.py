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

with open("new_llm_evaluation_payload.json", "r") as f:
    payload = json.load(f)

runs = [r for r in payload["runs"] if str(r.get("run_id", "")).startswith("THESIS-RBASE-MAIN-DS-")]

csv_map = {}
for f in glob.glob("results/round_metrics/*.csv"):
    base = os.path.basename(f).replace(".csv", "")
    csv_map[base] = f
    parts = base.rsplit("_", 1)
    if len(parts) == 2:
        csv_map[parts[0]] = f

def get_base_id(rid):
    b = re.sub(r"-S\d+(-|$)", r"", rid)
    b = re.sub(r"--+", "-", b).rstrip("-")
    return b

def get_style(method, rk, epochs, es):
    label = f"{method} (r={rk}, E={epochs}"
    if es != "None":
        label += f", ES={es})"
    else:
        label += ")"
        
    if method == "FDA-Standard":
        if "3TO30" in epochs and rk == 8:
            return label, "#0047AB", "--", "^"
        elif "3TO30" in epochs:
            return label, "#1f77b4", "-", "o"
        else:
            return label, "#0e879c", "-.", "s"
    elif method == "LoRA":
        if epochs == "20":
            return label, "#b2182b", "-", "d"
        elif epochs == "10":
            return label, "#d62728", "--", "s"
        elif epochs == "5":
            return label, "#ff7f0e", "-.", "o"
        else:
            return label, "#f46d43", ":", "x"
    elif method == "RoLoRA":
        if epochs == "20":
            return label, "#005a32", "-", "d"
        elif epochs == "10":
            return label, "#1b7837", "--", "s"
        elif epochs == "5":
            return label, "#2ca02c", "-.", "o"
        else:
            return label, "#66c2a5", ":", "x"
    elif method == "FFA-LoRA":
        if epochs == "10":
            return label, "#7570b3", "-", "s"
        else:
            return label, "#9467bd", "--", "o"
            
    return label, "#333333", "-", "x"

from collections import defaultdict
combos = defaultdict(lambda: defaultdict(list))

for r in runs:
    rid = r["run_id"]
    s = rid.replace("THESIS-RBASE-MAIN-DS-", "")
    m = re.match(r"^(?P<ds>[^-]+)-C(?P<clients>\d+)-R(?P<rounds>\d+)-M-(?P<rest>.*)$", s)
    ds = m.group("ds")
    c = int(m.group("clients"))
    rounds = int(m.group("rounds"))
    rest = m.group("rest")
    
    if rest.startswith("fda-standard"):
        method = "FDA-Standard"
        rem = rest[len("fda-standard"):].lstrip("-")
    elif rest.startswith("ffa"):
        method = "FFA-LoRA"
        rem = rest[len("ffa"):].lstrip("-")
    elif rest.startswith("rolora"):
        method = "RoLoRA"
        rem = rest[len("rolora"):].lstrip("-")
    elif rest.startswith("lora"):
        method = "LoRA"
        rem = rest[len("lora"):].lstrip("-")
    else:
        method = "Other"
        rem = rest
        
    rk_m = re.search(r"RK(\d+)", rem)
    rk = int(rk_m.group(1)) if rk_m else 4
    
    seed_m = re.search(r"S(\d+)", rem)
    seed = int(seed_m.group(1)) if seed_m else None
    
    es_m = re.search(r"ES([\d\.]+)", rem)
    es = es_m.group(1) if es_m else "None"
    
    base_id = get_base_id(rid)
    
    cf = csv_map.get(rid)
    if not cf:
        matches = [f for f in glob.glob("results/round_metrics/*.csv") if os.path.basename(f).startswith(rid)]
        cf = matches[0]
        
    df = pd.read_csv(cf)
    if ds == "mnli" and "accuracy_m" in df.columns:
        df["accuracy"] = (df["accuracy_m"] + df["accuracy_mm"]) / 2.0
        
    ep_m = re.search(r"E(\d+TO\d+|\d+)", rem)
    epochs = ep_m.group(1) if ep_m else str(df["local_epochs"].iloc[0])
        
    combos[(ds, c, rounds)][base_id].append({
        "rid": rid,
        "seed": seed,
        "method": method,
        "rank": rk,
        "epochs": epochs,
        "es": es,
        "rounds": df["round"].values,
        "accuracy": df["accuracy"].values * 100.0,
        "cum_time_min": (df["time_round_total"].cumsum().values) / 60.0,
        "max_acc": float(r["max_accuracy"]) * 100.0,
        "time_to_best_min": float(r["time_to_best_round_seconds"]) / 60.0,
        "best_round": int(df.loc[df["accuracy"].idxmax(), "round"]),
        "total_rounds": int(df["round"].iloc[-1])
    })

combo_order = [
    ("mnli", 50, 30, "fig1_mnli_c50_r30"),
    ("mrpc", 20, 30, "fig2_mrpc_c20_r30"),
    ("mrpc", 50, 12, "fig3_mrpc_c50_r12"),
    ("qnli", 50, 30, "fig4_qnli_c50_r30"),
    ("qqp", 50, 30, "fig5_qqp_c50_r30"),
    ("rte", 20, 30, "fig6_rte_c20_r30"),
    ("rte", 50, 12, "fig7_rte_c50_r12"),
    ("sst2", 20, 75, "fig8_sst2_c20_r75"),
    ("sst2", 50, 30, "fig9_sst2_c50_r30")
]

for ds, c, rounds, fig_basename in combo_order:
    cfgs = combos[(ds, c, rounds)]
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

    ax1.set_title(f"{ds.upper()} (C={c}, R={rounds}) : Accuracy vs. Round", fontweight='bold')
    ax1.set_xlabel("Communication Round")
    ax1.set_ylabel("Test Accuracy (%)")
    ax1.set_xlim(left=1, right=rounds)
    
    ax2.set_title(f"{ds.upper()} (C={c}, R={rounds}) : Accuracy vs. Wall-Clock Time", fontweight='bold')
    ax2.set_xlabel("Elapsed Wall-Clock Time (Minutes)")
    ax2.set_ylabel("Test Accuracy (%)")
    
    ax1.legend(loc="best", framealpha=0.85, edgecolor='#cccccc', fontsize=8)
    ax2.legend(loc="best", framealpha=0.85, edgecolor='#cccccc', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, f"{fig_basename}.pdf"), dpi=300, bbox_inches="tight")
    plt.savefig(os.path.join(FIG_DIR, f"{fig_basename}.png"), dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Generated {fig_basename}")

# Global Summary
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5.5))
b_names = [
    "MNLI\n(C50,R30)", "MRPC\n(C20,R30)", "MRPC\n(C50,R12)",
    "QNLI\n(C50,R30)", "QQP\n(C50,R30)", "RTE\n(C20,R30)",
    "RTE\n(C50,R12)", "SST2\n(C20,R75)", "SST2\n(C50,R30)"
]
methods_tracked = ["FDA-Standard", "LoRA", "RoLoRA", "FFA-LoRA"]
colors_tracked = {"FDA-Standard": "#1f77b4", "LoRA": "#ff7f0e", "RoLoRA": "#2ca02c", "FFA-LoRA": "#9467bd"}

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
width = 0.2

for i, m in enumerate(methods_tracked):
    offset = (i - 1.5) * width
    vals = best_peaks[m]
    ax1.bar(x + offset, vals, width, label=m, color=colors_tracked[m], alpha=0.9, edgecolor='black', linewidth=0.5)

ax1.set_title("Peak Test Accuracy Across Parameter Combinations", fontweight='bold', fontsize=12, pad=10)
ax1.set_ylabel("Peak Accuracy (%)")
ax1.set_xticks(x)
ax1.set_xticklabels(b_names, fontsize=9)
ax1.set_ylim(30, 105)
ax1.legend(loc="upper left", framealpha=0.92, edgecolor='#cccccc')

for i, m in enumerate(methods_tracked):
    offset = (i - 1.5) * width
    vals = best_times[m]
    ax2.bar(x + offset, vals, width, label=m, color=colors_tracked[m], alpha=0.9, edgecolor='black', linewidth=0.5)

ax2.set_title("Time to Peak Accuracy Across Parameter Combinations", fontweight='bold', fontsize=12, pad=10)
ax2.set_ylabel("Elapsed Time to Peak (Minutes, Log Scale)")
ax2.set_yscale('log')
ax2.set_xticks(x)
ax2.set_xticklabels(b_names, fontsize=9)
ax2.set_ylim(1, 40000)
ax2.legend(loc="upper right", framealpha=0.92, edgecolor='#cccccc')

plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "fig0_cross_benchmark_summary.pdf"), dpi=300, bbox_inches="tight")
plt.savefig(os.path.join(FIG_DIR, "fig0_cross_benchmark_summary.png"), dpi=200, bbox_inches="tight")
plt.close()
print("Generated fig0_cross_benchmark_summary")
