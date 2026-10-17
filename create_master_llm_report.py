import json
import os
import shutil

with open('analysis/LLM_SUMMARY_NEW2FDA.json', 'r') as f:
    fl_summary = json.load(f)

with open('variance-simulator/results/LLM_SUMMARY_VARIANCE_SIMULATOR.json', 'r') as f:
    sim_summary = json.load(f)

master_dict = {
    'title': 'FDA-LoRA Research Program: Master LLM-Friendly Results',
    'date': '2026-08-18',
    'federated_learning_benchmarks': fl_summary,
    'variance_simulator_validation': sim_summary
}

with open('results/LLM_MASTER_RESEARCH_REPORT.json', 'w') as f:
    json.dump(master_dict, f, indent=2)

shutil.copy('analysis/LLM_SUMMARY_NEW2FDA.json', 'results/LLM_SUMMARY_NEW2FDA.json')
shutil.copy('analysis/LLM_SUMMARY_NEW2FDA.md', 'results/LLM_SUMMARY_NEW2FDA.md')

# Build master markdown report
with open('results/LLM_MASTER_RESEARCH_REPORT.md', 'w') as f:
    f.write('# FDA-LoRA Master LLM Research Report\n\n')
    f.write('## 1. Executive Summary & Core Insights\n```json\n')
    f.write(json.dumps({
        'FL_Key_Findings': fl_summary['key_takeaways'],
        'Simulator_Hypotheses': sim_summary['hypotheses_verdicts']
    }, indent=2))
    f.write('\n```\n\n')
    
    f.write('## 2. Federated Learning Benchmark Matrix (NEW2FDA vs Baselines)\n\n')
    for key, data in fl_summary['benchmarks'].items():
        f.write(f"### Setting: {key} ({data['architecture']} - {data['scenario']})\n\n")
        f.write('| Method | R=25 | R=50 | R=75 | R=100 (Final) | Peak Acc (Best Round) | Avg Local Epochs | Total Compute Steps |\n')
        f.write('|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|\n')
        for method in data['fixed_round_accuracy'].keys():
            r25_val = data['fixed_round_accuracy'][method].get('R25', {}).get('mean', 0.0)
            r50_val = data['fixed_round_accuracy'][method].get('R50', {}).get('mean', 0.0)
            r75_val = data['fixed_round_accuracy'][method].get('R75', {}).get('mean', 0.0)
            r100_val = data['fixed_round_accuracy'][method].get('R100', {}).get('mean', 0.0)
            
            r25 = f"{r25_val*100:.1f}%"
            r50 = f"{r50_val*100:.1f}%"
            r75 = f"{r75_val*100:.1f}%"
            r100 = f"{r100_val*100:.1f}%"
            
            peak = data['peak_accuracy'].get(method, {})
            peak_acc = peak.get('max_accuracy_mean', 0.0) * 100
            peak_r = peak.get('best_round_median', '-')
            peak_str = f"{peak_acc:.1f}% (R{peak_r})"
            
            comp = data['computation_efficiency'].get(method, {})
            ep_str = f"{comp.get('avg_local_epochs_per_round', '-')}"
            step_str = f"{comp.get('total_steps_100_rounds', '-')}"
            
            f.write(f"| **{method}** | {r25} | {r50} | {r75} | **{r100}** | {peak_str} | {ep_str} | {step_str} |\n")
        f.write('\n')

    f.write('## 3. Variance Simulator Policy Evaluation\n\n')
    for run, r_data in sim_summary['simulation_suites'].items():
        f.write(f"### Suite: `{run}` ({r_data['scenario']})\n\n")
        f.write('| Policy & Track | NMAE | Log-RMSE | On-Target Stop % | Mean Stopping Step | Threshold Std |\n')
        f.write('|:---|:---:|:---:|:---:|:---:|:---:|\n')
        for pol, p_metrics in r_data['policies'].items():
            f.write(f"| **{pol}** | {p_metrics['nmae']} | {p_metrics['log_rmse']} | {p_metrics['on_target_rate']} | {p_metrics['mean_stopping_step']} | {p_metrics['threshold_std']} |\n")
        f.write('\n')

print('Generated results/LLM_MASTER_RESEARCH_REPORT.json and results/LLM_MASTER_RESEARCH_REPORT.md')
