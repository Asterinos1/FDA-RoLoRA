import os, re, glob
import pandas as pd
import numpy as np

def prep(df):
    if "round" in df.columns:
        df = df.sort_values("round").reset_index(drop=True)
    if "accuracy" not in df.columns and {"accuracy_m", "accuracy_mm"}.issubset(df.columns):
        df["accuracy"] = (df["accuracy_m"] + df["accuracy_mm"]) / 2.0
    if "round_epochs" not in df.columns:
        if "epochs" in df.columns:
            df["round_epochs"] = df["epochs"]
        elif "local_epochs" in df.columns:
            df["round_epochs"] = df["local_epochs"]
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
        records.append(row)
    return pd.DataFrame(records)

# The 16 scenarios where FDA Pure was tested
pure_scenarios = [
    # MNLI IID
    ("01_MNLI_C20_R50_LR0.1", "MNLI ($C=20, R=50, \\text{lr}=0.1, \\alpha=1000$)", "mnli", 20, 50, "0.1", [0.65, 0.72, 0.77], ("R@65%", "R@72%", "R@77%")),
    ("03_MNLI_C50_R30_LR0.1", "MNLI ($C=50, R=30, \\text{lr}=0.1, \\alpha=1000$)", "mnli", 50, 30, "0.1", [0.55, 0.65, 0.72], ("R@55%", "R@65%", "R@72%")),
    # QNLI IID
    ("05_QNLI_C20_R50_LR0.1", "QNLI ($C=20, R=50, \\text{lr}=0.1, \\alpha=1000$)", "qnli", 20, 50, "0.1", [0.80, 0.83, 0.85], ("R@80%", "R@83%", "R@85%")),
    ("07_QNLI_C50_R30_LR0.1", "QNLI ($C=50, R=30, \\text{lr}=0.1, \\alpha=1000$)", "qnli", 50, 30, "0.1", [0.75, 0.82, 0.86], ("R@75%", "R@82%", "R@86%")),
    # QQP IID
    ("09_QQP_C20_R50_LR0.1",  "QQP ($C=20, R=50, \\text{lr}=0.1, \\alpha=1000$)",  "qqp",  20, 50, "0.1", [0.65, 0.74, 0.76], ("R@65%", "R@74%", "R@76%")),
    ("10_QQP_C50_R30_LR0.1",  "QQP ($C=50, R=30, \\text{lr}=0.1, \\alpha=1000$)",  "qqp",  50, 30, "0.1", [0.75, 0.78, 0.80], ("R@75%", "R@78%", "R@80%")),
    # SST-2 IID
    ("11_SST2_C20_R50_LR0.1", "SST-2 ($C=20, R=50, \\text{lr}=0.1, \\alpha=1000$)", "sst2", 20, 50, "0.1", [0.85, 0.90, 0.924], ("R@85%", "R@90%", "R@92.4%")),
    ("12_SST2_C50_R30_LR0.1", "SST-2 ($C=50, R=30, \\text{lr}=0.1, \\alpha=1000$)", "sst2", 50, 30, "0.1", [0.85, 0.90, 0.91], ("R@85%", "R@90%", "R@91%")),
    # MNLI Non-IID
    ("17_MNLI_C20_R50_LR0.1_A0.5", "MNLI Non-IID ($C=20, R=50, \\text{lr}=0.1, \\alpha=0.5$)", "mnli", 20, 50, "0.1", [0.65, 0.72, 0.74], ("R@65%", "R@72%", "R@74%")),
    ("19_MNLI_C50_R30_LR0.1_A0.5", "MNLI Non-IID ($C=50, R=30, \\text{lr}=0.1, \\alpha=0.5$)", "mnli", 50, 30, "0.1", [0.50, 0.53, 0.55], ("R@50%", "R@53%", "R@55%")),
    # QNLI Non-IID
    ("18_QNLI_C20_R50_LR0.1_A0.5", "QNLI Non-IID ($C=20, R=50, \\text{lr}=0.1, \\alpha=0.5$)", "qnli", 20, 50, "0.1", [0.80, 0.83, 0.84], ("R@80%", "R@83%", "R@84%")),
    ("20_QNLI_C50_R30_LR0.1_A0.5", "QNLI Non-IID ($C=50, R=30, \\text{lr}=0.1, \\alpha=0.5$)", "qnli", 50, 30, "0.1", [0.55, 0.60, 0.65], ("R@55%", "R@60%", "R@65%")),
    # QQP Non-IID
    ("21_QQP_C20_R20_LR0.1_A0.5",  "QQP Non-IID ($C=20, R=20, \\text{lr}=0.1, \\alpha=0.5$)",  "qqp",  20, 20, "0.1", [0.65, 0.72, 0.76], ("R@65%", "R@72%", "R@76%")),
    ("22_QQP_C50_R24_LR0.1_A0.5",  "QQP Non-IID ($C=50, R=24, \\text{lr}=0.1, \\alpha=0.5$)",  "qqp",  50, 24, "0.1", [0.65, 0.70, 0.73], ("R@65%", "R@70%", "R@73%")),
    # SST-2 Non-IID
    ("23_SST2_C20_R25_LR0.1_A0.5", "SST-2 Non-IID ($C=20, R=25, \\text{lr}=0.1, \\alpha=0.5$)", "sst2", 20, 25, "0.1", [0.80, 0.85, 0.88], ("R@80%", "R@85%", "R@88%")),
    ("24_SST2_C50_R20_LR0.1_A0.5", "SST-2 Non-IID ($C=50, R=20, \\text{lr}=0.1, \\alpha=0.5$)", "sst2", 50, 20, "0.1", [0.55, 0.65, 0.80], ("R@55%", "R@65%", "R@80%")),
]

comm_rates = {(20, False): 4.6875, (20, True): 9.375, (50, False): 11.71875, (50, True): 23.4375}

def get_color(mname):
    if "Pure" in mname and ("Emax/1" in mname or "k=1.0" in mname):
        return "#15803d"  # Forest Green (FDA Pure k=1.0)
    elif "Pure" in mname and ("Emax/2" in mname or "k=0.5" in mname):
        return "#059669"  # Emerald Green (FDA Pure k=0.5)
    elif "Pure" in mname:
        return "#16a34a"  # Green
    elif "RoLoRA" in mname:
        return "#b91c1c"  # Red
    elif "LoRA" in mname and "FFA" not in mname:
        return "#b45309"  # Amber/Orange
    elif "FFA" in mname:
        return "#0891b2"  # Cyan
    return "#000000"

COLOR_MATRIX_A = "#047857"
BG_MATRIX_A    = "rgba(4, 120, 87, 0.08)"
COLOR_MATRIX_B = "#7c3aed"
BG_MATRIX_B    = "rgba(124, 58, 237, 0.08)"

def get_best_files_per_seed(files):
    by_seed = {}
    for f in files:
        m = re.search(r'-S(\d+)', os.path.basename(f))
        seed = m.group(1) if m else '0'
        sz = os.path.getsize(f)
        if seed not in by_seed or sz > by_seed[seed][1]:
            by_seed[seed] = (f, sz)
    return sorted([item[0] for item in by_seed.values()])

def find_e20_matches(ds, c, lr, method, rank):
    pat1 = f"results/round_metrics/*MAIN-DS-{ds}-C{c}-*-M-{method}-LR{lr}-RK{rank}-S*-E20*.csv"
    pat2 = f"results/round_metrics/*MAIN-DS-{ds}-C{c}-*-M-{method}-LR{lr}-RK{rank}-S*_[0-9a-f]*.csv"
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
    return sorted(valid)

def find_fda_pure_matches(ds, c, r, lr, rk, ratio_str):
    pats = [
        f"results/round_metrics/*THESIS-RBASE-PURE-DS-{ds}-C{c}-*-M-fda-pure-LR{lr}-RK{rk}-*CRATIO{ratio_str}-*.csv"
    ]
    matches = []
    for p in pats:
        matches.extend(glob.glob(p))
    return sorted(list(set(matches)))

def get_pure_suite_methods(ds, c, r, lr, include_lora=True, cfg_prefix=None):
    # Non-IID suites
    if cfg_prefix == "17_MNLI_C20_R50_LR0.1_A0.5":
        return [
            ("FDA Pure E1TO40 [k=0.5]", get_best_files_per_seed(glob.glob("results/round_metrics/*THESIS-NONIID-A0.5-DS-mnli-C20-*-M-fda-pure-LR0.1-RK4-E1TO40-CRATIO0.5-*.csv")), False),
            ("FDA Pure E1TO20 [k=1.0]", get_best_files_per_seed(glob.glob("results/round_metrics/*THESIS-NONIID-A0.5-DS-mnli-C20-*-M-fda-pure-LR0.1-RK4-E1TO20-CRATIO1-*.csv")), False),
            ("FDA Pure E1TO20 [k=0.5]", get_best_files_per_seed(glob.glob("results/round_metrics/*THESIS-NONIID-A0.5-DS-mnli-C20-*-M-fda-pure-LR0.1-RK4-E1TO20-CRATIO0.5-*.csv")), False),
            ("RoLoRA E20", get_best_files_per_seed(glob.glob("results/round_metrics/*THESIS-NONIID-A0.5-DS-mnli-C20-*-M-rolora-E20-LR0.1-RK4-*.csv")), False),
            ("RoLoRA E10", get_best_files_per_seed(glob.glob("results/round_metrics/*THESIS-NONIID-A0.5-DS-mnli-C20-*-M-rolora-E10-LR0.1-RK4-*.csv")), False),
        ]
    if cfg_prefix == "18_QNLI_C20_R50_LR0.1_A0.5":
        return [
            ("FDA Pure E1TO40 [k=1.0]", get_best_files_per_seed(glob.glob("results/round_metrics/*THESIS-NONIID-A0.5-DS-qnli-C20-*-M-fda-pure-LR0.1-RK4-E1TO40-CRATIO1-*.csv")), False),
            ("FDA Pure E1TO40 [k=0.5]", get_best_files_per_seed(glob.glob("results/round_metrics/*THESIS-NONIID-A0.5-DS-qnli-C20-*-M-fda-pure-LR0.1-RK4-E1TO40-CRATIO0.5-*.csv")), False),
            ("FDA Pure E1TO20 [k=1.0]", get_best_files_per_seed(glob.glob("results/round_metrics/*THESIS-NONIID-A0.5-DS-qnli-C20-*-M-fda-pure-LR0.1-RK4-E1TO20-CRATIO1-*.csv")), False),
            ("FDA Pure E1TO20 [k=0.5]", get_best_files_per_seed(glob.glob("results/round_metrics/*THESIS-NONIID-A0.5-DS-qnli-C20-*-M-fda-pure-LR0.1-RK4-E1TO20-CRATIO0.5-*.csv")), False),
            ("RoLoRA E20", get_best_files_per_seed(glob.glob("results/round_metrics/*THESIS-NONIID-A0.5-DS-qnli-C20-*-M-rolora-E20-LR0.1-RK4-*.csv")), False),
            ("RoLoRA E10", get_best_files_per_seed(glob.glob("results/round_metrics/*THESIS-NONIID-A0.5-DS-qnli-C20-*-M-rolora-E10-LR0.1-RK4-*.csv")), False),
        ]
    if cfg_prefix in ["19_MNLI_C50_R30_LR0.1_A0.5", "20_QNLI_C50_R30_LR0.1_A0.5", "21_QQP_C20_R20_LR0.1_A0.5", "22_QQP_C50_R24_LR0.1_A0.5", "23_SST2_C20_R25_LR0.1_A0.5", "24_SST2_C50_R20_LR0.1_A0.5"]:
        return [
            ("FDA Pure E1TO20 [k=1.0]", get_best_files_per_seed(glob.glob(f"results/round_metrics/*THESIS-NONIID-A0.5-DS-{ds}-C{c}-*-M-fda-pure-LR0.1-RK4-E1TO20-CRATIO1-*.csv")), False),
            ("FDA Pure E1TO20 [k=0.5]", get_best_files_per_seed(glob.glob(f"results/round_metrics/*THESIS-NONIID-A0.5-DS-{ds}-C{c}-*-M-fda-pure-LR0.1-RK4-E1TO20-CRATIO0.5-*.csv")), False),
            ("RoLoRA E10", get_best_files_per_seed(glob.glob(f"results/round_metrics/*THESIS-NONIID-A0.5-DS-{ds}-C{c}-*-M-rolora-E10-LR0.1-RK4-*.csv")), False),
        ]
    
    # IID suites (ONLY Pure, RoLoRA, LoRA, FFA-LoRA)
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

def build_pure_summary_table(include_lora=True):
    lines = [
        f"| Dataset / Configuration | Method | Seeds | Max Acc | Targeted Milestone 1 (Time, Complete MB) | Targeted Milestone 2 (Time, Complete MB) | Targeted Milestone 3 (Time, Complete MB) | Total Rounds | Complete MB | Best Round | Total Time (s / h) | Steps | Mean Ep/Rd | <span style=\"color: {COLOR_MATRIX_A};\">Mean Ep (A)/Rd</span> | <span style=\"color: {COLOR_MATRIX_B};\">Mean Ep (B)/Rd</span> |",
        "| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |"
    ]
    
    total_table_runs = 0
    total_unique_configs = 0
    
    for cfg_tuple in pure_scenarios:
        cfg_prefix, title, ds, c, r, lr, milestones, m_headers = cfg_tuple
        lines.append(f"| **Dataset / Configuration** | **Method** | **Seeds** | **Max Acc** | **{m_headers[0]} (Time, Complete MB)** | **{m_headers[1]} (Time, Complete MB)** | **{m_headers[2]} (Time, Complete MB)** | **Total Rounds** | **Complete MB** | **Best Round** | **Total Time (s / h)** | **Steps** | **Mean Ep/Rd** | <span style=\"color: {COLOR_MATRIX_A};\">**Mean Ep (A)/Rd**</span> | <span style=\"color: {COLOR_MATRIX_B};\">**Mean Ep (B)/Rd**</span> |")
        
        methods = get_pure_suite_methods(ds, c, r, lr, include_lora, cfg_prefix)
        
        rows = []
        for mname, pat, is_lora in methods:
            if isinstance(pat, list):
                matches = pat
            else:
                matches = sorted(glob.glob(pat))
            if not matches:
                continue
                
            dfs = [prep(pd.read_csv(f)) for f in matches]
            num_seeds = len(dfs)
            total_table_runs += num_seeds
            total_unique_configs += 1
            rate = comm_rates[(c, is_lora)]
            
            df_avg = average_runs(dfs)
            max_acc = float(df_avg["accuracy"].max())
            best_rnd = int(df_avg.loc[df_avg["accuracy"].idxmax(), "round"])
            tot_r = len(df_avg)
            mb = tot_r * rate
            sec = float(np.mean([df["time_round_total"].sum() for df in dfs])) if "time_round_total" in dfs[0].columns else 0.0
            steps = int(round(np.mean([df["round_steps"].sum() for df in dfs]))) if "round_steps" in dfs[0].columns else 0
            ep = float(np.mean([df["round_epochs"].mean() for df in dfs]))
            
            ep_a_list = []
            ep_b_list = []
            for df in dfs:
                if "active_matrix" in df.columns:
                    df_a = df[df["active_matrix"] == "A"]
                    df_b = df[df["active_matrix"] == "B"]
                    ea = df_a["round_epochs"].mean() if len(df_a) > 0 else 0.0
                    eb = df_b["round_epochs"].mean() if len(df_b) > 0 else 0.0
                elif "FFA" in mname:
                    ea = 0.0
                    eb = df["round_epochs"].mean() if "round_epochs" in df.columns else 0.0
                else:
                    ea = df["round_epochs"].mean() if "round_epochs" in df.columns else 0.0
                    eb = ea
                ep_a_list.append(ea)
                ep_b_list.append(eb)
            ep_a = float(np.mean(ep_a_list))
            ep_b = float(np.mean(ep_b_list))
            
            m_res = []
            for th in milestones:
                crossing = np.where(df_avg["accuracy"].values >= th)[0]
                if len(crossing) > 0:
                    rhit = int(crossing[0]) + 1
                    times = [df.loc[:rhit-1, "time_round_total"].sum() if len(df) >= rhit else df["time_round_total"].sum() for df in dfs] if "time_round_total" in dfs[0].columns else [0.0]
                    th_h = float(np.mean(times)) / 3600.0
                    th_mb = rhit * rate
                    m_res.append((rhit, th_h, th_mb))
                else:
                    m_res.append(None)
                    
            rows.append({
                "mname": mname,
                "color": get_color(mname),
                "num_seeds": num_seeds,
                "max_acc": max_acc,
                "best_rnd": best_rnd,
                "tot_r": tot_r,
                "mb": mb,
                "sec": sec,
                "time_h": sec / 3600.0,
                "steps": steps,
                "ep": ep,
                "ep_a": ep_a,
                "ep_b": ep_b,
                "milestones": m_res,
            })
            
        if not rows:
            continue
            
        best_acc = max(r["max_acc"] for r in rows)
        min_tot_r = min(r["tot_r"] for r in rows)
        min_mb = min(r["mb"] for r in rows)
        
        best_m_rounds = []
        for m_idx in range(len(milestones)):
            hits = [r["milestones"][m_idx][0] for r in rows if r["milestones"][m_idx] is not None]
            best_m_rounds.append(min(hits) if hits else None)
            
        for idx, row in enumerate(rows):
            col = row["color"]
            mname_str = f'<span style="color: {col};">{row["mname"]}</span>'
            seeds_str = f'<span style="color: {col};">{row["num_seeds"]}</span>'
            
            acc_val_str = f'{row["max_acc"]:.4f}'
            if abs(row["max_acc"] - best_acc) < 1e-5:
                acc_str = f'**<span style="color: {col};">{acc_val_str}</span>**'
                mname_str = f'**{mname_str}**'
            else:
                acc_str = f'<span style="color: {col};">{acc_val_str}</span>'
                
            m_cells = []
            for m_idx in range(len(milestones)):
                m_info = row["milestones"][m_idx]
                if m_info is None:
                    m_cells.append(f'<span style="color: {col};">—</span>')
                else:
                    rhit, th_h, th_mb = m_info
                    if best_m_rounds[m_idx] is not None and rhit == best_m_rounds[m_idx]:
                        b_tag = f'<b style="font-size: 1.22em; text-decoration: underline;">{rhit}</b>'
                    else:
                        b_tag = f'<b style="font-size: 1.18em;">{rhit}</b>'
                    m_cells.append(f'<span style="color: {col};">{b_tag} <span style="font-size: 0.85em; opacity: 0.9;">({th_h:.2f} h, {th_mb:.1f} MB)</span></span>')
                    
            tot_r_str = f'{row["tot_r"]}'
            if row["tot_r"] == min_tot_r:
                tot_r_cell = f'**<span style="color: {col};">{tot_r_str}</span>**'
            else:
                tot_r_cell = f'<span style="color: {col};">{tot_r_str}</span>'
                
            mb_str = f'{row["mb"]:.1f} MB'
            if abs(row["mb"] - min_mb) < 1e-4:
                mb_cell = f'**<span style="color: {col};">{mb_str}</span>**'
            else:
                mb_cell = f'<span style="color: {col};">{mb_str}</span>'
                
            best_rnd_cell = f'<span style="color: {col};">{row["best_rnd"]}</span>'
            time_cell = f'<span style="color: {col};">{row["sec"]:,.0f} s ({row["time_h"]:.2f} h)</span>'
            steps_cell = f'<span style="color: {col};">{row["steps"]:,}</span>'
            ep_cell = f'<span style="color: {col};"><b style="font-size: 1.12em;">{row["ep"]:.2f}</b></span>'
            ep_a_cell = f'<span style="color: {COLOR_MATRIX_A}; background-color: {BG_MATRIX_A}; padding: 1px 5px; border-radius: 3px;">{row["ep_a"]:.2f}</span>'
            ep_b_cell = f'<span style="color: {COLOR_MATRIX_B}; background-color: {BG_MATRIX_B}; padding: 1px 5px; border-radius: 3px;">{row["ep_b"]:.2f}</span>'
            
            cfg_cell = f'**{title}**' if idx == 0 else ''
            
            line = f"| {cfg_cell} | {mname_str} | {seeds_str} | {acc_str} | {m_cells[0]} | {m_cells[1]} | {m_cells[2]} | {tot_r_cell} | {mb_cell} | {best_rnd_cell} | {time_cell} | {steps_cell} | {ep_cell} | {ep_a_cell} | {ep_b_cell} |"
            lines.append(line)
            
    print(f"Total table rows built successfully! Total runs: {total_table_runs}, Unique configs: {total_unique_configs}")
    return "\n".join(lines)

if __name__ == "__main__":
    tbl = build_pure_summary_table(include_lora=True)
    with open("scratch/pure_table.txt", "w", encoding="utf-8") as f:
        f.write(tbl)
    print("Saved to scratch/pure_table.txt")
