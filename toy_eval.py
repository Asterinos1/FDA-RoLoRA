import os
import glob
import pandas as pd
import json

def export_toy_metrics_for_llm(metrics_dir="results/round_metrics", output_file="toy_metrics_llm_summary.json"):
    csv_files = glob.glob(os.path.join(metrics_dir, "TOY*.csv"))
    if not csv_files:
        print(f"No toy script CSV files found in {metrics_dir}")
        return

    df_list = [pd.read_csv(f) for f in csv_files]
    df = pd.concat(df_list, ignore_index=True)
    
    if df.empty or 'num_clients' not in df.columns:
        print("Dataset is empty or missing required columns.")
        return

    llm_summary = {}
    
    for num_clients in sorted(df['num_clients'].dropna().unique()):
        scenario_df = df[df['num_clients'] == num_clients]
        scenario_key = f"{int(num_clients)}_clients"
        llm_summary[scenario_key] = {}
        
        for method in scenario_df['method'].dropna().unique():
            method_df = scenario_df[scenario_df['method'] == method]
            
            agg_df = method_df.groupby('round')['accuracy'].agg(['mean', 'std']).reset_index()
            
            llm_summary[scenario_key][method] = {
                "round": agg_df['round'].tolist(),
                "accuracy_mean": agg_df['mean'].round(4).tolist(),
                "accuracy_std": agg_df['std'].fillna(0.0).round(4).tolist()
            }
            
    with open(output_file, 'w') as f:
        json.dump(llm_summary, f, indent=4)
        
    print(f"LLM-readable summary generated at: {output_file}")

if __name__ == "__main__":
    export_toy_metrics_for_llm()