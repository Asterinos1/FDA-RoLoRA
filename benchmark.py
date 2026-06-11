import math
import numpy as np
import pandas as pd
import itertools
import sys
from tqdm import tqdm

class AdvancedDualTrackVarianceSimulator:
    def __init__(self, total_rounds=100, e_min=2, e_max=10, 
                 base_var_A=1.0, base_var_B=0.5, 
                 noise_std=0.05, total_clients=100, 
                 participation_rate=0.1, dirichlet_alpha=1.0,
                 rank_multiplier=1.0, jitter_factor=1.0,
                 sweet_spot_epoch=0, non_linear_drift=0.0,
                 straggler_prob=0.02, shock_prob=0.05):
        
        self.total_rounds = total_rounds
        self.e_min = e_min
        self.e_max = e_max
        self.noise_std = noise_std
        self.total_clients = total_clients
        self.participation_rate = participation_rate
        self.dirichlet_alpha = dirichlet_alpha
        self.jitter_factor = jitter_factor
        self.sweet_spot_epoch = sweet_spot_epoch
        self.non_linear_drift = non_linear_drift
        self.straggler_prob = straggler_prob
        self.shock_prob = shock_prob
        
        self.k_clients = max(1, int(self.total_clients * self.participation_rate))
        alpha_penalty = 1.0 + (1.0 / self.dirichlet_alpha) if self.dirichlet_alpha > 0 else 10.0
        
        self.base_var = {
            'A': base_var_A * alpha_penalty * rank_multiplier, 
            'B': base_var_B * alpha_penalty * rank_multiplier
        }
        self.growth_factor = 0.1 * alpha_penalty
        
    def simulate_intra_round_variance(self, base_v, e, active_matrix):
        expected_dropouts = int(self.k_clients * self.straggler_prob * max(0, e - self.e_min))
        current_k = max(1, self.k_clients - expected_dropouts)
        
        matrix_growth = self.growth_factor * 1.5 if active_matrix == 'B' else self.growth_factor
        linear_term = matrix_growth * e
        quadratic_term = self.non_linear_drift * ((e - self.sweet_spot_epoch) ** 2)
        
        drift_multiplier = max(0.1, 1.0 + linear_term + quadratic_term)
        structural_var = base_v * drift_multiplier
        effective_var = structural_var / current_k
        
        dynamic_noise_scale = (self.noise_std * self.jitter_factor * math.sqrt(e)) / math.sqrt(current_k)
        noise = np.random.standard_t(df=3) * dynamic_noise_scale
        
        if np.random.rand() < self.shock_prob:
            noise += np.random.uniform(1.5, 3.0) * effective_var
            
        return max(0.001, effective_var + noise)

    def run(self, update_method="original", **method_kwargs):
        history = {'A': {'rounds': [], 'e_star': [], 'theta': [], 'variance': []},
                   'B': {'rounds': [], 'e_star': [], 'theta': [], 'variance': []}}
        
        var_init_A = self.simulate_intra_round_variance(self.base_var['A'], self.e_min, 'A')
        var_init_B = self.simulate_intra_round_variance(self.base_var['B'], self.e_min, 'B')
        theta = {'A': var_init_A * 1.5, 'B': var_init_B * 1.5}
        tau = method_kwargs.get('tau', self.e_max / 2)
        
        for r in range(1, self.total_rounds + 1):
            active_matrix = 'A' if r % 2 != 0 else 'B'
            current_theta = theta[active_matrix]
            
            e_star = self.e_min
            var_exact = self.simulate_intra_round_variance(self.base_var[active_matrix], e_star, active_matrix)
            
            while e_star < self.e_max and var_exact <= current_theta:
                e_star += 1
                var_exact = self.simulate_intra_round_variance(self.base_var[active_matrix], e_star, active_matrix)
            
            history[active_matrix]['rounds'].append(r)
            history[active_matrix]['e_star'].append(e_star)
            history[active_matrix]['theta'].append(current_theta)
            history[active_matrix]['variance'].append(var_exact)
            
            if update_method == "original":
                next_theta = (tau / e_star) * var_exact
            elif update_method == "ema_standard":
                rho = method_kwargs.get('rho', 0.8)
                proposed_target = (tau / e_star) * var_exact
                next_theta = rho * current_theta + (1 - rho) * proposed_target
            elif update_method == "ema_pure":
                rho = method_kwargs.get('rho', 0.8)
                alpha_param = method_kwargs.get('alpha', 1.15)
                proposed_target = alpha_param * var_exact
                next_theta = rho * current_theta + (1 - rho) * proposed_target
            else:
                raise ValueError(f"Unknown update method: {update_method}")
                
            theta[active_matrix] = next_theta
            
        return history

def evaluate_multi_seed(scenario_kwargs, method_name, method_kwargs, num_seeds=3):
    metrics = {
        'A': {'ratios': [], 'opt_counts': 0, 'epochs': [], 'emax_counts': 0, 'emin_counts': 0, 'total': 0},
        'B': {'ratios': [], 'opt_counts': 0, 'epochs': [], 'emax_counts': 0, 'emin_counts': 0, 'total': 0}
    }
    
    for seed in range(num_seeds):
        np.random.seed(seed)
        sim = AdvancedDualTrackVarianceSimulator(total_rounds=100, **scenario_kwargs)
        history = sim.run(update_method=method_name, **method_kwargs)
        
        for matrix in ['A', 'B']:
            t_arr = np.array(history[matrix]['theta'])
            v_arr = np.array(history[matrix]['variance'])
            e_arr = np.array(history[matrix]['e_star'])
            
            r = t_arr / np.where(v_arr == 0, 1e-9, v_arr)
            metrics[matrix]['ratios'].extend(r.tolist())
            metrics[matrix]['opt_counts'] += np.sum((r >= 1.05) & (r <= 1.50))
            metrics[matrix]['emax_counts'] += np.sum(e_arr == sim.e_max)
            metrics[matrix]['emin_counts'] += np.sum(e_arr == sim.e_min)
            metrics[matrix]['epochs'].extend(e_arr.tolist())
            metrics[matrix]['total'] += len(r)
            
    compiled_metrics = {}
    for m in ['A', 'B']:
        total = metrics[m]['total']
        compiled_metrics[f'ratio_mean_{m}'] = np.mean(metrics[m]['ratios'])
        compiled_metrics[f'opt_pct_{m}'] = (metrics[m]['opt_counts'] / total) * 100
        compiled_metrics[f'epochs_mean_{m}'] = np.mean(metrics[m]['epochs'])
        compiled_metrics[f'epochs_std_{m}'] = np.std(metrics[m]['epochs'])
        compiled_metrics[f'fail_emax_pct_{m}'] = (metrics[m]['emax_counts'] / total) * 100
        compiled_metrics[f'fail_emin_pct_{m}'] = (metrics[m]['emin_counts'] / total) * 100
        
    compiled_metrics['opt_pct_mean'] = (compiled_metrics['opt_pct_A'] + compiled_metrics['opt_pct_B']) / 2
    compiled_metrics['avg_epochs_mean'] = (compiled_metrics['epochs_mean_A'] + compiled_metrics['epochs_mean_B']) / 2
    compiled_metrics['savings_mean'] = ((10.0 - compiled_metrics['avg_epochs_mean']) / 10.0) * 100
    
    return compiled_metrics

def run_grid_search():
    param_grid = {
        "dirichlet_alpha": [0.05, 0.5, 1.0],         
        "jitter_factor": [1.0, 5.0, 10.0],           
        "non_linear_drift": [0.15, 0.4],             
        "sweet_spot_epoch": [3, 6],                  
        "straggler_prob": [0.0, 0.05, 0.20],         
        "shock_prob": [0.0, 0.10, 0.25],             
        "participation_rate": [0.01, 0.1]            
    }

    methods = [
        {"name": "original", "kwargs": {"tau": 5}, "config": "tau=5"},
        {"name": "ema_standard", "kwargs": {"tau": 5, "rho": 0.7}, "config": "tau=5, rho=0.7"},
        {"name": "ema_standard", "kwargs": {"tau": 5, "rho": 0.85}, "config": "tau=5, rho=0.85"},
        {"name": "ema_pure", "kwargs": {"alpha": 1.15, "rho": 0.8}, "config": "a=1.15, rho=0.8"},
        {"name": "ema_pure", "kwargs": {"alpha": 1.25, "rho": 0.85}, "config": "a=1.25, rho=0.85"}
    ]

    keys, values = zip(*param_grid.items())
    combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    
    tasks = [(combo, m) for combo in combinations for m in methods]
    total_runs = len(tasks) * 3
    print(f"Executing Grid Search...\nTotal Scenarios: {len(combinations)} | Methods: {len(methods)} | Total Runs: {total_runs}\n")

    results = []
    try:
        for combo, m in tqdm(tasks, desc="Evaluating Configurations"):
            perf = evaluate_multi_seed(combo, m["name"], m["kwargs"], num_seeds=3)
            
            row = {
                "C": combo["participation_rate"],
                "α": combo["dirichlet_alpha"],
                "Jitter": combo["jitter_factor"],
                "Straggler P(x)": combo["straggler_prob"],
                "Shock P(x)": combo["shock_prob"],
                "Method": m["name"],
                "Config": m["config"],
                "Opt % (Total)": round(perf['opt_pct_mean'], 1),
                "Opt % (A)": round(perf['opt_pct_A'], 1),
                "Opt % (B)": round(perf['opt_pct_B'], 1),
                "Fail e_max % (B)": round(perf['fail_emax_pct_B'], 1),
                "Fail e_min % (A)": round(perf['fail_emin_pct_A'], 1),
                "Epochs μ": round(perf['avg_epochs_mean'], 2),
                "Epochs σ (A)": round(perf['epochs_std_A'], 2),
                "Epochs σ (B)": round(perf['epochs_std_B'], 2),
                "Savings %": round(perf['savings_mean'], 1)
            }
            results.append(row)
    except KeyboardInterrupt:
        print("\n\n[!] Execution interrupted by user. Processing collected data up to this point...")

    if not results:
        print("No metrics collected. Exiting program.")
        sys.exit(0)

    df = pd.DataFrame(results)
    df.to_csv("federated_variance_grid_search_results.csv", index=False)
    return df

def generate_automated_assessment(best_row):
    print("="*80)
    print("AUTOMATED BENCHMARK ASSESSMENT")
    print("="*80)
    
    method = best_row['Method']
    config = best_row['Config']
    opt_total = best_row['Opt % (Total)']
    epochs_mu = best_row['Epochs μ']
    savings = best_row['Savings %']
    
    alpha = best_row['α']
    jitter = best_row['Jitter']
    stragglers = best_row['Straggler P(x)']
    shocks = best_row['Shock P(x)']
    
    print(f"\nWINNING METHOD: {method.upper()} [{config}]")
    print(f"Optimal Tracking Percentage: {opt_total}%")
    print(f"Average Epochs Extracted: {epochs_mu} (Compute Savings: {savings}%)")
    
    print("\nENVIRONMENTAL CONDITIONS CONQUERED:")
    print(f"- Non-IID Severity (α): {alpha}")
    print(f"- Gradient Jitter: {jitter}")
    print(f"- Straggler Dropout Rate: {stragglers * 100}% per epoch")
    print(f"- Shock Probability: {shocks * 100}% per round")
    
    print("\nDETAILED ANALYSIS:")
    if method == "ema_pure":
        print("The Pure-EMA Geometric method triumphed because it successfully decoupled threshold ")
        print("calculations from the discrete epoch count (e*). In environments with stragglers ")
        print(f"({stragglers*100}% dropout) and periodic catastrophic shocks ({shocks*100}% probability), ")
        print("relying on inverse-epoch scaling causes cascading failures. By treating variance ")
        print("growth as a geometric constant (alpha) smoothed by historical momentum (rho), ")
        print("this method prevented erratic threshold collapse, allowing Matrix A and B to ")
        print(f"safely extract an average of {epochs_mu} epochs despite severe instability.")
        
    elif method == "ema_standard":
        print("The Standard EMA method achieved the highest stability by applying exponential ")
        print("smoothing to the inverse-epoch scaling target. While it still relies on the ")
        print("halting epoch (e*) to project the next threshold, the momentum parameter (rho) ")
        print(f"acted as a critical low-pass filter against the high jitter ({jitter}). ")
        print("It successfully ignored anomalous gradient noise without permanently crippling ")
        print(f"local exploitation, resulting in consistent optimal tracking ({opt_total}%).")
        
    elif method == "original":
        print("The Original inverse-epoch method won, indicating a highly stable environmental ")
        print("scenario. Because data distribution was near-IID and shocks were minimal, ")
        print("historical momentum was unnecessary. The strictly responsive mathematical ")
        print("relationship (tau / e*) allowed for hyper-efficient halting precisely at the ")
        print("sweet spot, maximizing compute savings without risk of the stale momentum trap.")
        
    print("\nMATRIX ASYMMETRY NOTE:")
    print(f"Matrix A Optimal Tracking: {best_row['Opt % (A)']}% | E_min Failure Rate: {best_row['Fail e_min % (A)']}%")
    print(f"Matrix B Optimal Tracking: {best_row['Opt % (B)']}% | E_max Failure Rate: {best_row['Fail e_max % (B)']}%")
    if best_row['Fail e_min % (A)'] > 10.0:
        print("-> Warning: Matrix A frequently suffocated early in rounds. Consider lowering its base variance multiplier.")
    if best_row['Fail e_max % (B)'] > 10.0:
        print("-> Warning: Matrix B frequently hit the epoch ceiling, indicating its threshold grew too loose.")
    print("="*80)

if __name__ == "__main__":
    benchmark_df = run_grid_search()
    
    best_run = benchmark_df.sort_values(by="Opt % (Total)", ascending=False).iloc[0]
    
    top_performers = benchmark_df.sort_values(by="Opt % (Total)", ascending=False).head(10)
    print("\nTop 10 Configurations:")
    print(top_performers.to_markdown(index=False))
    print("\n")
    
    generate_automated_assessment(best_run)