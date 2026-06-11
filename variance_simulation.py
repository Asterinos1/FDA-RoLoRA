import math
import numpy as np
import matplotlib.pyplot as plt

class AdvancedDualTrackVarianceSimulator:
    """
    Simulates intra-round variance and early-stopping dynamics in a dual-track 
    Federated Learning environment, supporting non-linear drift, structural jitter,
    straggler dropouts, heavy-tailed noise, and non-IID shocks.
    """
    
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
        """
        Calculates effective variance incorporating heavy-tails, stragglers, and shocks.
        """
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
        """
        Executes the federated training simulation across all rounds.
        """
        history = {
            'A': {'rounds': [], 'e_star': [], 'theta': [], 'variance': []},
            'B': {'rounds': [], 'e_star': [], 'theta': [], 'variance': []}
        }
        
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
                raise ValueError("Unknown update method")

            theta[active_matrix] = next_theta

        return history

def plot_simulation(history, method_name, config_title):
    """
    Generates diagnostic plots mapping variance thresholds and halting dynamics.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharex='col')
    fig.suptitle(f'Dual-Track Variance Simulation: {method_name}\n({config_title})', fontsize=14)

    for i, matrix in enumerate(['A', 'B']):
        rounds = history[matrix]['rounds']
        theta_array = np.array(history[matrix]['theta'])
        var_array = np.array(history[matrix]['variance'])
        
        axes[0, i].plot(rounds, theta_array, label='Threshold (θ)', color='blue', linestyle='--')
        axes[0, i].plot(rounds, var_array, label='Realized Var', color='red', alpha=0.7)
        axes[0, i].set_title(f'Matrix {matrix} - Variance Space')
        axes[0, i].set_ylabel('Variance')
        axes[0, i].legend(loc='lower right')
        axes[0, i].grid(True, alpha=0.3)

        mean_ratio = np.mean(theta_array / np.where(var_array == 0, 1e-9, var_array))
        
        if mean_ratio < 1.05:
            eval_text = "Threshold too low"
            text_color = "darkred"
        elif mean_ratio > 1.50:
            eval_text = "Threshold too large"
            text_color = "darkred"
        else:
            eval_text = "Optimal Dynamics"
            text_color = "darkgreen"

        axes[0, i].text(0.5, 0.95, eval_text, 
                        transform=axes[0, i].transAxes, 
                        ha='center', va='top', 
                        fontsize=12, fontweight='bold', color=text_color,
                        bbox=dict(facecolor='white', alpha=0.8, edgecolor=text_color, boxstyle='round,pad=0.3'))

        axes[1, i].plot(rounds, history[matrix]['e_star'], label='Realized Epochs (e*)', color='green', marker='o', markersize=3)
        axes[1, i].axhline(10, color='black', linestyle=':', label='e_max (10)')
        axes[1, i].axhline(2, color='gray', linestyle=':', label='e_min (2)')
        axes[1, i].set_title(f'Matrix {matrix} - Halting Dynamics')
        axes[1, i].set_xlabel('Global Round (r)')
        axes[1, i].set_ylabel('Epochs')
        axes[1, i].set_ylim(0, 12)
        axes[1, i].legend(loc='lower right')
        axes[1, i].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

def configure_simulation():
    """
    Interactive prompt to define standard federated learning environment parameters,
    including advanced radical behavior settings. Features pre-configured scenarios
    and expansive explanatory comments for manual tuning.
    """
    print("="*60)
    print("FDA-RoLoRA Dual-Track Environment Configuration")
    print("="*60)
    
    presets = {
        '2': {
            'name': "High-Straggler Edge Network (High Dropout, Non-IID)",
            'params': {
                'total_clients': 200, 'participation_rate': 0.05, 'dirichlet_alpha': 0.1,
                'rank_multiplier': 0.8, 'sweet_spot_epoch': 3, 'non_linear_drift': 0.2,
                'jitter_factor': 6.0, 'base_var_A': 1.0, 'base_var_B': 0.5,
                'noise_std': 0.05, 'straggler_prob': 0.15, 'shock_prob': 0.02
            }
        },
        '3': {
            'name': "Stable Cross-Silo (Zero Dropout, Near-IID)",
            'params': {
                'total_clients': 20, 'participation_rate': 1.0, 'dirichlet_alpha': 10.0,
                'rank_multiplier': 1.0, 'sweet_spot_epoch': 5, 'non_linear_drift': 0.05,
                'jitter_factor': 1.0, 'base_var_A': 0.5, 'base_var_B': 0.2,
                'noise_std': 0.01, 'straggler_prob': 0.0, 'shock_prob': 0.0
            }
        },
        '4': {
            'name': "Catastrophic Shock Stress Test (High Volatility)",
            'params': {
                'total_clients': 100, 'participation_rate': 0.1, 'dirichlet_alpha': 0.5,
                'rank_multiplier': 1.0, 'sweet_spot_epoch': 4, 'non_linear_drift': 0.15,
                'jitter_factor': 8.0, 'base_var_A': 1.2, 'base_var_B': 0.6,
                'noise_std': 0.08, 'straggler_prob': 0.02, 'shock_prob': 0.20
            }
        },
        '5': {
            'name': "High-Rank Capacity Expansion (r=64, Steep Drift)",
            'params': {
                'total_clients': 100, 'participation_rate': 0.1, 'dirichlet_alpha': 1.0,
                'rank_multiplier': 1.5, 'sweet_spot_epoch': 2, 'non_linear_drift': 0.3,
                'jitter_factor': 4.0, 'base_var_A': 1.5, 'base_var_B': 0.8,
                'noise_std': 0.05, 'straggler_prob': 0.02, 'shock_prob': 0.05
            }
        }
    }

    print("Select a simulation scenario:")
    print("1) Interactive Manual Configuration")
    for key, preset in presets.items():
        print(f"{key}) Pre-configured: {preset['name']}")

    choice = input("\nEnter choice (1-5) [Default: 1]: ").strip()

    if choice in presets:
        print(f"\nLoading {presets[choice]['name']}...")
        return presets[choice]['params'], presets[choice]['name']

    try:
        print("\n[Total Clients (N)]")
        print("Explanation: The total number of available devices in the federated pool.")
        print("Impact: A larger pool often means more diverse data but requires careful sampling.")
        total_clients = int(input("Enter Total Clients (N) [Default: 100]: ").strip() or 100)

        print("\n[Participation Rate (C)]")
        print("Explanation: The fraction of the total client pool sampled in a single communication round.")
        print("Impact: Determines active clients per round (k = N * C). Lower rates increase variance due to less statistical representation.")
        participation_rate = float(input("Enter Participation Rate (C) [0.0 - 1.0, Default: 0.1]: ").strip() or 0.1)

        print("\n[Dirichlet Alpha (α)]")
        print("Explanation: Controls the non-IID data distribution among clients.")
        print("Impact: A lower value (e.g., 0.1) forces highly skewed, non-IID data, leading to severe structural variance penalties. A higher value (e.g., 10.0) approximates IID data.")
        dirichlet_alpha = float(input("Enter Dirichlet Alpha (α) [Default: 1.0]: ").strip() or 1.0)
        
        print("\n" + "-"*30 + "\nRadical Behavior Parameters\n" + "-"*30)

        print("\n[Sweet-Spot Epoch]")
        print("Explanation: The local epoch at which the global structural variance is mathematically minimized before local overfitting begins.")
        print("Impact: Models the classic FL 'U-curve' where variance drops initially then rises. Setting this to 4 means variance is lowest at epoch 4.")
        sweet_spot_epoch = int(input("Enter Sweet-Spot Epoch [Default: 4]: ").strip() or 4)

        print("\n[Non-Linear Drift Coefficient]")
        print("Explanation: Controls the steepness of the quadratic variance explosion after passing the sweet-spot epoch.")
        print("Impact: Higher values simulate models that are highly sensitive to local overfitting (client drift).")
        non_linear_drift = float(input("Enter Non-Linear Drift Coefficient [Default: 0.15]: ").strip() or 0.15)

        print("\n[Jitter Factor]")
        print("Explanation: A multiplier for epoch-dependent noise volatility.")
        print("Impact: Simulates the chaotic, heavy-tailed gradient noise found in LLM fine-tuning. Higher values create more erratic round-to-round variance.")
        jitter_factor = float(input("Enter Jitter Factor [Default: 5.0]: ").strip() or 5.0)

        print("\n[Straggler Probability]")
        print("Explanation: The probability that a participating client drops out or times out for every additional local epoch requested.")
        print("Impact: Simulates real-world system constraints. More epochs requested = fewer returning clients = higher aggregation variance.")
        straggler_prob = float(input("Enter Straggler Probability [Default: 0.02]: ").strip() or 0.02)

        print("\n[Shock Probability]")
        print("Explanation: The chance that a client encounters an anomalous, highly non-IID data batch during local training.")
        print("Impact: Introduces sudden, catastrophic spikes in the variance curve, specifically testing the robustness of the EMA thresholding.")
        shock_prob = float(input("Enter Shock Probability [Default: 0.05]: ").strip() or 0.05)

        print("\n[LoRA Rank Selection]")
        print("Explanation: Defines the capacity of the Low-Rank Adaptation matrices.")
        print("Impact: High rank (e.g., r=64) geometrically expands the parameter space, increasing baseline variance compared to low rank (e.g., r=8).")
        print("1) Low Rank (Tighter parameter space, lower baseline variance)")
        print("2) High Rank (Larger geometric capacity, higher baseline variance)")
        rank_choice = input("Enter 1 or 2 [Default: 1]: ").strip() or '1'

    except ValueError:
        print("\n[!] Invalid input detected. Falling back to default configuration.")
        total_clients, participation_rate, dirichlet_alpha = 100, 0.1, 1.0
        sweet_spot_epoch, non_linear_drift, jitter_factor = 4, 0.15, 5.0
        straggler_prob, shock_prob = 0.02, 0.05
        rank_choice = '1'

    rank_multiplier = 1.5 if rank_choice == '2' else 0.8
    rank_str = "High Rank" if rank_choice == '2' else "Low Rank"
    
    params = {
        'total_clients': total_clients,
        'participation_rate': participation_rate,
        'dirichlet_alpha': dirichlet_alpha,
        'rank_multiplier': rank_multiplier,
        'sweet_spot_epoch': sweet_spot_epoch,
        'non_linear_drift': non_linear_drift,
        'jitter_factor': jitter_factor,
        'base_var_A': 1.0, 
        'base_var_B': 0.5,
        'noise_std': 0.05,
        'straggler_prob': straggler_prob,
        'shock_prob': shock_prob
    }

    config_title = f"Manual: N={total_clients} | C={participation_rate} | α={dirichlet_alpha} | Jitter={jitter_factor} | {rank_str}"
    return params, config_title

if __name__ == "__main__":
    np.random.seed(42)
    
    sim_params, config_title = configure_simulation()
    
    simulator = AdvancedDualTrackVarianceSimulator(total_rounds=100, **sim_params)
    history_original = simulator.run(update_method="original")
    plot_simulation(history_original, "Original (τ/e*) * Var", config_title)
    
    simulator = AdvancedDualTrackVarianceSimulator(total_rounds=100, **sim_params)
    history_ema_std = simulator.run(update_method="ema_standard", rho=0.8, tau=5)
    plot_simulation(history_ema_std, "Standard EMA (ρ=0.8, τ=5)", config_title)

    simulator = AdvancedDualTrackVarianceSimulator(total_rounds=100, **sim_params)
    history_ema_pure = simulator.run(update_method="ema_pure", rho=0.8, alpha=1.15)
    plot_simulation(history_ema_pure, "Pure-EMA Geometric (ρ=0.8, α=1.15)", config_title)