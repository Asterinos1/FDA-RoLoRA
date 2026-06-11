import math
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

class DualTrackVarianceSimulator:
    def __init__(self, total_rounds=100, e_min=2, e_max=10, 
                 base_var_A=1.0, base_var_B=0.5, 
                 decay_rate_A=0.03, decay_rate_B=0.015,
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
        self.decay_rate = {'A': decay_rate_A, 'B': decay_rate_B}
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
            
            self.base_var[active_matrix] *= (1 - self.decay_rate[active_matrix])
            
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
                
            theta[active_matrix] = next_theta

        return history

def plot_simulation(history, method_name, config_title):
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
    return fig

# --- STREAMLIT UI CONFIGURATION ---
st.set_page_config(layout="wide", page_title="FDA-RoLoRA Simulator")
st.title("FDA-RoLoRA Dual-Track Environment")

PRESETS = {
    "Manual Configuration": {},
    "High-Straggler Edge Network": {
        "total_clients": 200, "participation_rate": 0.05, "dirichlet_alpha": 0.1,
        "sweet_spot_epoch": 3, "non_linear_drift": 0.2, "jitter_factor": 6.0,
        "straggler_prob": 0.15, "shock_prob": 0.02, "rank_choice": "Low Rank (r=8)"
    },
    "Stable Cross-Silo": {
        "total_clients": 20, "participation_rate": 1.0, "dirichlet_alpha": 10.0,
        "sweet_spot_epoch": 5, "non_linear_drift": 0.05, "jitter_factor": 1.0,
        "straggler_prob": 0.0, "shock_prob": 0.0, "rank_choice": "Low Rank (r=8)"
    },
    "Catastrophic Shock Stress Test": {
        "total_clients": 100, "participation_rate": 0.1, "dirichlet_alpha": 0.5,
        "sweet_spot_epoch": 4, "non_linear_drift": 0.15, "jitter_factor": 8.0,
        "straggler_prob": 0.02, "shock_prob": 0.20, "rank_choice": "Low Rank (r=8)"
    },
    "High-Rank Capacity Expansion": {
        "total_clients": 100, "participation_rate": 0.1, "dirichlet_alpha": 1.0,
        "sweet_spot_epoch": 2, "non_linear_drift": 0.3, "jitter_factor": 4.0,
        "straggler_prob": 0.02, "shock_prob": 0.05, "rank_choice": "High Rank (r=64)"
    }
}

st.sidebar.header("Pre-configured Scenarios")
selected_scenario = st.sidebar.selectbox("Load Preset", list(PRESETS.keys()))
p = PRESETS[selected_scenario]

def def_val(key, default):
    return p.get(key, default) if p else default

st.sidebar.header("Environment Parameters")
total_clients = st.sidebar.number_input("Total Clients (N)", min_value=1, value=int(def_val("total_clients", 100)))
participation_rate = st.sidebar.slider("Participation Rate (C)", 0.01, 1.0, float(def_val("participation_rate", 0.1)))
dirichlet_alpha = st.sidebar.slider("Dirichlet Alpha (α)", 0.1, 10.0, float(def_val("dirichlet_alpha", 1.0)))

st.sidebar.header("Radical Behavior")
sweet_spot_epoch = st.sidebar.number_input("Sweet-Spot Epoch", min_value=0, value=int(def_val("sweet_spot_epoch", 4)))
non_linear_drift = st.sidebar.slider("Non-Linear Drift", 0.0, 1.0, float(def_val("non_linear_drift", 0.15)))
jitter_factor = st.sidebar.slider("Jitter Factor", 0.0, 10.0, float(def_val("jitter_factor", 5.0)))
straggler_prob = st.sidebar.slider("Straggler Prob P(x)", 0.0, 0.5, float(def_val("straggler_prob", 0.02)))
shock_prob = st.sidebar.slider("Shock Prob P(x)", 0.0, 0.5, float(def_val("shock_prob", 0.05)))

rank_choice = st.sidebar.radio("LoRA Rank", ["Low Rank (r=8)", "High Rank (r=64)"], 
                               index=0 if def_val("rank_choice", "Low Rank (r=8)") == "Low Rank (r=8)" else 1)
rank_multiplier = 1.5 if "High" in rank_choice else 0.8

st.sidebar.header("Update Method")
update_method = st.sidebar.selectbox(
    "Threshold Algorithm", 
    ["original", "ema_standard", "ema_pure"]
)

rho = 0.8
alpha_param = 1.15
tau_divisor = 2.0
e_max = 10 

if update_method == "original":
    tau_divisor = st.sidebar.slider("Tau Divisor (x) for e_max / x", 1.0, 10.0, 2.0, step=0.1)
elif update_method == "ema_standard":
    rho = st.sidebar.slider("EMA Rho (ρ)", 0.1, 0.99, 0.8)
    tau_divisor = st.sidebar.slider("Tau Divisor (x) for e_max / x", 1.0, 10.0, 2.0, step=0.1)
elif update_method == "ema_pure":
    rho = st.sidebar.slider("EMA Rho (ρ)", 0.1, 0.99, 0.8)
    alpha_param = st.sidebar.slider("Alpha Param (α)", 1.0, 2.0, 1.15)

tau = e_max / tau_divisor

sim_params = {
    'total_clients': total_clients,
    'participation_rate': participation_rate,
    'dirichlet_alpha': dirichlet_alpha,
    'rank_multiplier': rank_multiplier,
    'sweet_spot_epoch': sweet_spot_epoch,
    'non_linear_drift': non_linear_drift,
    'jitter_factor': jitter_factor,
    'straggler_prob': straggler_prob,
    'shock_prob': shock_prob,
    'base_var_A': 1.0, 
    'base_var_B': 0.5,
    'decay_rate_A': 0.03, 
    'decay_rate_B': 0.015,
    'noise_std': 0.05
}

config_title = f"{selected_scenario} | N={total_clients} | C={participation_rate} | α={dirichlet_alpha} | Jitter={jitter_factor}"

simulator = DualTrackVarianceSimulator(total_rounds=100, e_max=e_max, **sim_params)
history = simulator.run(update_method=update_method, tau=tau, rho=rho, alpha=alpha_param)

fig = plot_simulation(history, update_method.replace('_', ' ').title(), config_title)
st.pyplot(fig)