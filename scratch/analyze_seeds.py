import pandas as pd

df = pd.read_csv("scratch/parsed_thesis_runs.csv")
valid_df = df[df['rounds'] >= 5].copy()

def clean_method(row):
    m = str(row['method'])
    ep = row['local_epochs']
    fname = row['file']
    if "Pure" in fname or row['is_pure']:
        if "CRATIO1.0" in fname or "CRATIO1" in fname or "CR1" in fname:
            return "FDA Pure (k=1.0)"
        elif "CRATIO0.5" in fname or "CR0.5" in fname:
            return "FDA Pure (k=0.5)"
        else:
            return "FDA Pure (other)"
    elif "FDA" in m or "fda" in fname.lower():
        if "CRATIO1" in fname:
            return "FDA Old (k=1.0)"
        elif "CRATIO0.75" in fname:
            return "FDA Old (k=0.75)"
        elif "CRATIO0.5" in fname:
            return "FDA Old (k=0.5)"
        elif "E3TO40" in fname:
            return "FDA Old E3TO40"
        elif "E3TO30" in fname:
            return "FDA Old E3TO30"
        elif "E3TO20" in fname:
            return "FDA Old E3TO20"
        elif "E3TO10" in fname:
            return "FDA Old E3TO10"
        else:
            return f"FDA Old ({m})"
    elif "rolora" in m.lower():
        return f"RoLoRA E{ep}"
    elif "lora" in m.lower():
        if "ffa" in m.lower() or "ffa" in fname.lower():
            return f"FFA-LoRA E{ep}"
        return f"LoRA E{ep}"
    else:
        return f"{m} E{ep}"

valid_df['clean_method'] = valid_df.apply(clean_method, axis=1)

grouped = valid_df.groupby(['dataset', 'alpha', 'clients', 'lr', 'clean_method'])['seed'].unique().reset_index()
grouped['seed_count'] = grouped['seed'].apply(len)
grouped['seeds_list'] = grouped['seed'].apply(lambda s: sorted([int(x) for x in s if pd.notna(x)]))

print("=== ROWS 0 TO 60 ===")
for i in range(min(60, len(grouped))):
    r = grouped.iloc[i]
    print(f"{i:3d} | {r['dataset']:5s} | alpha={r['alpha']:<6.1f} | C={r['clients']:2d} | lr={r['lr']:<4.2f} | {r['clean_method']:30s} | n={r['seed_count']} | {r['seeds_list']}")
