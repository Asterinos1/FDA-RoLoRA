import sys
from pathlib import Path
import pandas as pd
import re
from collections import defaultdict

# Pre-completed (13)
skipped_extras = set([
    'THESIS-RBASE-PURE-DS-qnli-C20-R50-M-fda-pure-LR0.1-RK4-E1TO20-CRATIO1-NODISC-BSTART-NOGUARD-S42-ES0.8-MR6-P3-D0.005-W3',
    'THESIS-RBASE-PURE-DS-qnli-C20-R50-M-fda-pure-LR0.1-RK4-E1TO20-CRATIO0.5-NODISC-BSTART-NOGUARD-S42-ES0.8-MR6-P3-D0.005-W3',
    'THESIS-RBASE-PURE-DS-qnli-C50-R30-M-fda-pure-LR0.1-RK4-E1TO20-CRATIO1-NODISC-BSTART-NOGUARD-S42-ES0.8-MR6-P3-D0.005-W3',
    'THESIS-RBASE-PURE-DS-qnli-C50-R30-M-fda-pure-LR0.1-RK4-E1TO20-CRATIO0.5-NODISC-BSTART-NOGUARD-S42-ES0.8-MR6-P3-D0.005-W3',
    'THESIS-RBASE-PURE-DS-mnli-C20-R50-M-fda-pure-LR0.1-RK4-E1TO20-CRATIO1-NODISC-BSTART-NOGUARD-S42-ES0.6-MR6-P3-D0.005-W3',
    'THESIS-RBASE-PURE-DS-mnli-C20-R50-M-fda-pure-LR0.1-RK4-E1TO20-CRATIO0.5-NODISC-BSTART-NOGUARD-S42-ES0.6-MR6-P3-D0.005-W3',
    'THESIS-RBASE-PURE-DS-mnli-C50-R30-M-fda-pure-LR0.1-RK4-E1TO20-CRATIO1-NODISC-BSTART-NOGUARD-S42-ES0.55-MR6-P3-D0.005-W3',
    'THESIS-RBASE-PURE-DS-mnli-C50-R30-M-fda-pure-LR0.1-RK4-E1TO20-CRATIO0.5-NODISC-BSTART-NOGUARD-S42-ES0.55-MR6-P3-D0.005-W3',
    'THESIS-NONIID-A0.5-DS-qnli-C20-R50-M-rolora-E10-LR0.1-RK4-BSTART-S42-ES0.8-MR30-P5-D0.005-W5',
    'THESIS-NONIID-A0.5-DS-qnli-C20-R50-M-fda-pure-LR0.1-RK4-E1TO20-CRATIO1-NODISC-BSTART-NOGUARD-S42-ES0.8-MR30-P5-D0.005-W5',
    'THESIS-NONIID-A0.5-DS-qnli-C20-R50-M-fda-pure-LR0.1-RK4-E1TO20-CRATIO0.5-NODISC-BSTART-NOGUARD-S42-ES0.8-MR30-P5-D0.005-W5',
    'THESIS-NONIID-A0.5-DS-mnli-C20-R50-M-rolora-E10-LR0.1-RK4-BSTART-S42-ES0.6-MR30-P5-D0.005-W5',
    'THESIS-NONIID-A0.5-DS-mnli-C20-R50-M-fda-pure-LR0.1-RK4-E1TO20-CRATIO0.5-NODISC-BSTART-NOGUARD-S42-ES0.6-MR30-P5-D0.005-W5'
])

# Finished in this session (22)
prompt_dones_text = """
[b93d6ab692] THESIS-RBASE-PURE-DS-qqp-C20-R50-M-fda-pure-LR0.1-RK4-E1TO20-CRATIO0.5-NODISC-BSTART-NOGUARD-S42-ES0.72-MR6-P3-D0.005-W3; round=13 test_accuracy=0.7338 test_f1=0.6835 test_loss=0.5285 train_accuracy=0.6943 round_epochs=13.0000
[06bf14ce6c] THESIS-RBASE-PURE-DS-qqp-C20-R50-M-fda-pure-LR0.1-RK4-E1TO20-CRATIO0.5-NODISC-BSTART-NOGUARD-S43-ES0.72-MR6-P3-D0.005-W3; round=15 test_accuracy=0.7255 test_f1=0.7008 test_loss=0.5090 train_accuracy=0.7095 round_epochs=9.0000
[f7ea35904a] THESIS-RBASE-PURE-DS-qqp-C20-R50-M-fda-pure-LR0.1-RK4-E1TO20-CRATIO1-NODISC-BSTART-NOGUARD-S42-ES0.72-MR6-P3-D0.005-W3; round=13 test_accuracy=0.7258 test_f1=0.7090 test_loss=0.5177 train_accuracy=0.7318 round_epochs=17.0000
[ee20ad2344] THESIS-RBASE-PURE-DS-qqp-C50-R30-M-fda-pure-LR0.1-RK4-E1TO20-CRATIO1-NODISC-BSTART-NOGUARD-S42-ES0.72-MR6-P3-D0.005-W3; round=14 test_accuracy=0.7182 test_f1=0.6791 test_loss=0.5254 train_accuracy=0.7266 round_epochs=8.0000
[1886dc1a81] THESIS-RBASE-PURE-DS-qqp-C50-R30-M-fda-pure-LR0.1-RK4-E1TO20-CRATIO1-NODISC-BSTART-NOGUARD-S43-ES0.72-MR6-P3-D0.005-W3; round=14 test_accuracy=0.7347 test_f1=0.6857 test_loss=0.5164 train_accuracy=0.7346 round_epochs=10.0000
[0234788fa4] THESIS-RBASE-PURE-DS-qqp-C50-R30-M-fda-pure-LR0.1-RK4-E1TO20-CRATIO0.5-NODISC-BSTART-NOGUARD-S43-ES0.72-MR6-P3-D0.005-W3; round=17 test_accuracy=0.7194 test_f1=0.6834 test_loss=0.5285 train_accuracy=0.6955 round_epochs=9.0000
[935da48338] THESIS-RBASE-PURE-DS-qqp-C50-R30-M-fda-pure-LR0.1-RK4-E1TO20-CRATIO0.5-NODISC-BSTART-NOGUARD-S42-ES0.72-MR6-P3-D0.005-W3; round=19 test_accuracy=0.7222 test_f1=0.6866 test_loss=0.5241 train_accuracy=0.6948 round_epochs=10.0000
[830cf672d7] THESIS-RBASE-PURE-DS-qqp-C20-R50-M-fda-pure-LR0.1-RK4-E1TO20-CRATIO1-NODISC-BSTART-NOGUARD-S43-ES0.72-MR6-P3-D0.005-W3; round=22 test_accuracy=0.8011 test_f1=0.7634 test_loss=0.4224 train_accuracy=0.7961 round_epochs=20.0000
[ad56b10b8a] THESIS-RBASE-PURE-DS-qnli-C20-R50-M-fda-pure-LR0.1-RK4-E1TO20-CRATIO0.5-NODISC-BSTART-NOGUARD-S43-ES0.8-MR6-P3-D0.005-W3; round=17 test_accuracy=0.8109 test_loss=0.4587 train_accuracy=0.6150 round_epochs=6.0000
[be2cce5120] THESIS-RBASE-PURE-DS-qnli-C50-R30-M-fda-pure-LR0.1-RK4-E1TO20-CRATIO1-NODISC-BSTART-NOGUARD-S43-ES0.8-MR6-P3-D0.005-W3; round=17 test_accuracy=0.8126 test_loss=0.4615 train_accuracy=0.7626 round_epochs=14.0000
[64d7ae452d] THESIS-RBASE-PURE-DS-qnli-C20-R50-M-fda-pure-LR0.1-RK4-E1TO20-CRATIO1-NODISC-BSTART-NOGUARD-S43-ES0.8-MR6-P3-D0.005-W3; round=19 test_accuracy=0.8338 test_loss=0.4127 train_accuracy=0.7909 round_epochs=20.0000
[037713d967] THESIS-RBASE-PURE-DS-qnli-C50-R30-M-fda-pure-LR0.1-RK4-E1TO20-CRATIO0.5-NODISC-BSTART-NOGUARD-S43-ES0.8-MR6-P3-D0.005-W3; round=23 test_accuracy=0.8063 test_loss=0.4487 train_accuracy=0.7255 round_epochs=9.0000
[b10ed2964b] THESIS-RBASE-PURE-DS-mnli-C50-R30-M-fda-pure-LR0.1-RK4-E1TO20-CRATIO0.5-NODISC-BSTART-NOGUARD-S43-ES0.55-MR6-P3-D0.005-W3; round=30 test_accuracy=0.6181 matched_accuracy=0.6100 mismatched_accuracy=0.6262 train_accuracy=0.5927 round_epochs=7.0000
[700e794734] THESIS-RBASE-PURE-DS-mnli-C20-R50-M-fda-pure-LR0.1-RK4-E1TO20-CRATIO0.5-NODISC-BSTART-NOGUARD-S43-ES0.6-MR6-P3-D0.005-W3; round=38 test_accuracy=0.7632 matched_accuracy=0.7586 mismatched_accuracy=0.7677 train_accuracy=0.7566 round_epochs=11.0000
[b773b52275] THESIS-RBASE-PURE-DS-mnli-C20-R50-M-fda-pure-LR0.1-RK4-E1TO20-CRATIO1-NODISC-BSTART-NOGUARD-S43-ES0.6-MR6-P3-D0.005-W3; round=29 test_accuracy=0.7678 matched_accuracy=0.7644 mismatched_accuracy=0.7713 train_accuracy=0.8169 round_epochs=19.0000
[b146cb13ea] THESIS-RBASE-PURE-DS-mnli-C50-R30-M-fda-pure-LR0.1-RK4-E1TO20-CRATIO1-NODISC-BSTART-NOGUARD-S43-ES0.55-MR6-P3-D0.005-W3; round=30 test_accuracy=0.7223 matched_accuracy=0.7148 mismatched_accuracy=0.7299 train_accuracy=0.7032 round_epochs=17.0000
[cf0fab3c6f] THESIS-NONIID-A0.5-DS-qnli-C20-R50-M-rolora-E10-LR0.1-RK4-BSTART-S43-ES0.8-MR30-P5-D0.005-W5; round=34/50 matrix=A epochs=10 train_loss=0.2262 acc=0.7805 var=9.1883e-01
[e9ca25f6b3] THESIS-NONIID-A0.5-DS-qnli-C50-R30-M-rolora-E10-LR0.1-RK4-BSTART-S42-ES0.8-MR18-P5-D0.005-W5; round=30/30 matrix=A epochs=10 train_loss=0.2861 acc=0.6744 var=1.1120e+00
[b05a8de440] THESIS-NONIID-A0.5-DS-qnli-C20-R50-M-fda-pure-LR0.1-RK4-E1TO20-CRATIO0.5-NODISC-BSTART-NOGUARD-S43-ES0.8-MR30-P5-D0.005-W5; round=30 test_accuracy=0.6414 test_loss=0.6749 train_accuracy=0.9036 round_epochs=10.0000
[2ec8e923a8] THESIS-NONIID-A0.5-DS-qnli-C50-R30-M-rolora-E10-LR0.1-RK4-BSTART-S43-ES0.8-MR18-P5-D0.005-W5; round=30/30 matrix=A epochs=10 train_loss=0.3637 acc=0.4946 var=6.9714e-01
[2605096cc2] THESIS-NONIID-A0.5-DS-qnli-C50-R30-M-fda-pure-LR0.1-RK4-E1TO20-CRATIO0.5-NODISC-BSTART-NOGUARD-S42-ES0.8-MR18-P5-D0.005-W5; round=30 test_accuracy=0.5080 test_loss=0.6916 train_accuracy=0.8958 round_epochs=10.0000
[9513561939] THESIS-NONIID-A0.5-DS-qnli-C20-R50-M-fda-pure-LR0.1-RK4-E1TO20-CRATIO1-NODISC-BSTART-NOGUARD-S43-ES0.8-MR30-P5-D0.005-W5; round=32 test_accuracy=0.7983 test_loss=0.5128 train_accuracy=0.9269 round_epochs=20.0000
"""

results = {}
for line in prompt_dones_text.strip().splitlines():
    line = line.strip()
    m = re.match(r"\[([a-f0-9]+)\] (.+?); (.*)", line)
    if m:
        h, ex, stats = m.groups()
        acc_m = re.search(r"(?:test_accuracy|acc)=([0-9\.]+)", stats)
        acc = float(acc_m.group(1)) if acc_m else None
        rnd_m = re.search(r"round=([0-9/]+)", stats)
        rnd = rnd_m.group(1) if rnd_m else None
        results[ex] = {"acc": acc, "round": rnd}

for ex in skipped_extras:
    files = list(Path("results/round_metrics").glob(f"{ex}_*.csv"))
    if files:
        df = pd.read_csv(files[0])
        last = df.iloc[-1]
        if "mnli" in ex:
            acc_m = last.get("accuracy_m")
            acc_mm = last.get("accuracy_mm")
            acc = (acc_m + acc_mm)/2.0 if (pd.notnull(acc_m) and pd.notnull(acc_mm)) else None
        else:
            acc = last.get("accuracy", last.get("test_accuracy"))
        rnd = last.get("round")
        results[ex] = {"acc": float(acc) if pd.notnull(acc) else None, "round": rnd}

active_extras = set([
    'THESIS-NONIID-A0.5-DS-qnli-C50-R30-M-fda-pure-LR0.1-RK4-E1TO20-CRATIO0.5-NODISC-BSTART-NOGUARD-S43-ES0.8-MR18-P5-D0.005-W5',
    'THESIS-NONIID-A0.5-DS-mnli-C20-R50-M-fda-pure-LR0.1-RK4-E1TO20-CRATIO1-NODISC-BSTART-NOGUARD-S42-ES0.6-MR30-P5-D0.005-W5',
    'THESIS-NONIID-A0.5-DS-mnli-C20-R50-M-rolora-E10-LR0.1-RK4-BSTART-S43-ES0.6-MR30-P5-D0.005-W5',
    'THESIS-NONIID-A0.5-DS-mnli-C20-R50-M-fda-pure-LR0.1-RK4-E1TO20-CRATIO1-NODISC-BSTART-NOGUARD-S43-ES0.6-MR30-P5-D0.005-W5',
    'THESIS-NONIID-A0.5-DS-mnli-C20-R50-M-fda-pure-LR0.1-RK4-E1TO20-CRATIO0.5-NODISC-BSTART-NOGUARD-S43-ES0.6-MR30-P5-D0.005-W5',
    'THESIS-NONIID-A0.5-DS-mnli-C50-R30-M-rolora-E10-LR0.1-RK4-BSTART-S42-ES0.55-MR18-P5-D0.005-W5'
])

sys.path.insert(0, "master_experiments")
import thesis_master_experiments as tm
configs = tm.build_master_configs()

groups = defaultdict(lambda: {"s42": {"status": "QUEUED", "acc": None, "round": None},
                              "s43": {"status": "QUEUED", "acc": None, "round": None}})

for c in configs:
    ds = c["ds_name"].upper()
    is_noniid = float(c.get("alpha", 1000.0)) < 10.0
    regime = "Non-IID (α=0.5)" if is_noniid else "IID"
    if c["METHOD_NAME"] == "FedAvg-RoLoRA":
        method = "RoLoRA (Fixed E=10)"
    else:
        k_val = c.get("controller_ratio", 1.0)
        method = f"FDA-Pure (k={k_val:g})"
    clients = f"C={c['num_clients']}"
    key = (ds, regime, method, clients)
    
    extra = c["extras"]
    s_key = f"s{c['seed']}"
    
    if extra in results:
        groups[key][s_key]["status"] = "FINISHED"
        groups[key][s_key]["acc"] = results[extra]["acc"]
        groups[key][s_key]["round"] = results[extra]["round"]
    elif extra in active_extras:
        groups[key][s_key]["status"] = "RUNNING"
    else:
        groups[key][s_key]["status"] = "QUEUED"

by_dataset = defaultdict(list)
for key, v in sorted(groups.items()):
    ds = key[0]
    by_dataset[ds].append((key, v))

for ds in ["QQP", "QNLI", "MNLI", "SST2"]:
    glist = by_dataset[ds]
    print(f"\n=======================================================")
    print(f"DATASET: {ds} ({len(glist)} Parameter Groups)")
    print(f"=======================================================")
    for key, v in glist:
        s42_info = v["s42"]
        s43_info = v["s43"]
        
        status_s42 = s42_info["status"]
        status_s43 = s43_info["status"]
        
        if status_s42 == "FINISHED" and status_s43 == "FINISHED":
            state = "COMPLETED"
            accs = [s42_info["acc"], s43_info["acc"]]
            if all(a is not None for a in accs):
                mean_acc = sum(accs)/2.0
                acc_str = f"S42={accs[0]*100:.2f}%, S43={accs[1]*100:.2f}% (Mean={mean_acc*100:.2f}%)"
            else:
                acc_str = f"S42={accs[0]}, S43={accs[1]}"
        elif status_s42 == "FINISHED" or status_s43 == "FINISHED":
            state = "PARTIAL (1/2)"
            done_s = "S42" if status_s42 == "FINISHED" else "S43"
            done_acc = s42_info["acc"] if status_s42 == "FINISHED" else s43_info["acc"]
            other_s = "S43" if status_s42 == "FINISHED" else "S42"
            other_status = status_s43 if status_s42 == "FINISHED" else status_s42
            acc_str = f"{done_s}={done_acc*100:.2f}% | {other_s}={other_status}"
        elif status_s42 == "RUNNING" or status_s43 == "RUNNING":
            state = "IN PROGRESS"
            acc_str = f"S42={status_s42}, S43={status_s43}"
        else:
            state = "QUEUED"
            acc_str = "Pending"
            
        print(f"[{state:12s}] {key[1]:17s} | {key[2]:20s} | {key[3]:4s} | {acc_str}")
