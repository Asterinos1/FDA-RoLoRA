import os
import glob
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def get_available_csv_files(directory_path: str) -> list[str]:
    if not os.path.exists(directory_path):
        print(f"Directory '{directory_path}' does not exist.")
        return []
    return sorted(glob.glob(os.path.join(directory_path, "*.csv")))

def prompt_selection(items: list[str], prompt_text: str) -> list[int] | None:
    for i, item in enumerate(items):
        print(f"[{i}] {item}")
        
    while True:
        choices = input(f"{prompt_text} (or enter 'q' to quit): ").strip()
        
        if choices.lower() in ['q', 'quit']:
            return None
            
        try:
            selected_indices = [int(x.strip()) for x in choices.split(",")]
            if all(0 <= idx < len(items) for idx in selected_indices):
                return selected_indices
            else:
                print("Error: One or more indices are out of range.")
        except ValueError:
            print("Error: Invalid input. Please enter comma-separated numbers or 'q' to quit.")

def extract_dynamic_metrics(dataframes: dict[str, pd.DataFrame]) -> list[str]:
    all_metrics = set()
    whitelist = [
        'f1', 'matthews_correlation', 'testing_loss', 'training_loss', 'variance'
    ]
    
    has_acc = False
    has_train_acc = False
    has_epochs = False

    for df in dataframes.values():
        numeric_cols = df.select_dtypes(include=['number']).columns
        for col in numeric_cols:
            col_lower = col.lower()
            if col_lower == 'round' or 'unnamed' in col_lower:
                continue
            
            if col_lower == 'accuracy':
                has_acc = True
                continue
            if col_lower == 'training_accuracy':
                has_train_acc = True
                continue
            if col_lower in ['local_epochs', 'round_epochs', 'max_epochs', 'e_min']:
                if col_lower in ['local_epochs', 'round_epochs']:
                    has_epochs = True
                continue
            
            if df[col].nunique() > 1 or col_lower in whitelist:
                all_metrics.add(col)
                
    if has_acc and has_train_acc and has_epochs:
        all_metrics.add('combined_acc_epochs')
    else:
        if has_acc: all_metrics.add('accuracy')
        if has_train_acc: all_metrics.add('training_accuracy')
            
    return sorted(list(all_metrics))

def format_plot_label(run_name: str) -> str:
    parts = run_name.split("_")
    if len(parts) > 5:
        if parts[-1].startswith("v") and parts[-1][1:].isdigit():
            return "_".join(parts[-5:])
        return "_".join(parts[-4:])
    return run_name

def get_grid_dimensions(n: int) -> tuple[int, int]:
    if n == 0:
        return 1, 1
    cols = math.ceil(math.sqrt(n))
    rows = math.ceil(n / cols)
    return rows, cols

def _safe_minmax_norm(series: pd.Series, higher_is_better: bool = True) -> pd.Series:
    """
    Normalize a series to [0, 1].  When all values are identical the range is
    zero, so every run gets 0.5 (neutral) instead of forcing one to 0 and
    another to 1 — avoiding meaningless score spread caused by tiny differences.
    """
    lo, hi = series.min(), series.max()
    rng = hi - lo
    if rng < 1e-9:
        return pd.Series(0.5, index=series.index)
    normalized = (series - lo) / rng
    return normalized if higher_is_better else (1.0 - normalized)


def _convergence_speed(acc: np.ndarray, rounds: np.ndarray, threshold: float = 0.90) -> float:
    """
    Return the *round index* at which accuracy first crosses `threshold` *
    relative to the run's own peak accuracy (not absolute 90 %).
    Lower is better (converges sooner).  Returns len(rounds) when the
    threshold is never reached, so late/non-converging runs are penalised.
    """
    target = threshold * acc.max()
    crossed = np.where(acc >= target)[0]
    if len(crossed) == 0:
        return float(rounds[-1])          # never converged — worst case
    # Normalise by the total number of rounds so runs of different lengths
    # are on the same scale.
    return rounds[crossed[0]] / rounds[-1]


def evaluate_methods(dataframes: dict[str, pd.DataFrame]):
    print("\n=> Evaluating Method Performance...")

    valid_runs = {
        k: v for k, v in dataframes.items()
        if 'accuracy' in v.columns and 'round' in v.columns
    }

    if len(valid_runs) < 2:
        print("At least two valid runs with 'accuracy' and 'round' columns are required.")
        return

    stats = {}
    for run_name, df in valid_runs.items():
        df_sorted = df.sort_values(by='round')
        acc    = df_sorted['accuracy'].values
        rounds = df_sorted['round'].values

        if len(acc) < 5:
            print(f"  Skipping '{run_name}': fewer than 5 rounds.")
            continue

        # ── 1. Final accuracy ─────────────────────────────────────────────
        # Smooth over the last 10 % of rounds (min 3) to reduce noise while
        # still reflecting the end-state rather than a single lucky round.
        tail = max(3, int(round(len(acc) * 0.10)))
        final_acc = float(np.mean(acc[-tail:]))

        # ── 2. Full-run stability (ALL rounds) ────────────────────────────
        # Global std captures the entire training trajectory, not just the
        # tail.  We also compute the tail std separately and take the *max*
        # so a run that was unstable early AND unstable late is penalised on
        # both counts, while a run that calmed down after initial turbulence
        # is credited appropriately.
        global_std = float(np.std(acc))
        tail_std   = float(np.std(acc[-tail:]))
        instability = max(global_std, tail_std)

        # ── 3. Convergence AUC (normalised, ALL rounds) ───────────────────
        # Divide by the *number of rounds*, not the round-index range, so
        # runs that started at round 0 vs round 1 are treated equally.
        _trapz = getattr(np, 'trapezoid', getattr(np, 'trapz', None))
        auc = float(_trapz(acc, rounds))
        n_rounds = len(rounds)
        convergence_auc = auc / n_rounds if n_rounds > 1 else float(acc[0])

        # ── 4. Convergence speed ──────────────────────────────────────────
        # Fraction of training elapsed before reaching 90 % of own peak.
        # Lower = faster convergence.
        conv_speed = _convergence_speed(acc, rounds, threshold=0.90)

        # ── 5. Generalisation gap (ALL rounds) ────────────────────────────
        # Use the *mean absolute* gap over all rounds, clipped to ≥ 0.
        # This distinguishes a model that consistently overfits from one
        # that only briefly diverges.  Absolute value prevents a negative
        # gap (test > train, common in dropout/FL) from masking real issues.
        gen_gap = float('nan')
        if 'training_accuracy' in df_sorted.columns:
            train_acc = df_sorted['training_accuracy'].values
            per_round_gap = np.clip(train_acc - acc, 0, None)   # floor at 0
            gen_gap = float(np.mean(per_round_gap))

        stats[run_name] = {
            'final_accuracy'   : final_acc,
            'convergence_auc'  : convergence_auc,
            'convergence_speed': conv_speed,        # lower = faster
            'instability'      : instability,       # lower = better
            'generalization_gap': gen_gap,
        }

    if not stats:
        print("Insufficient data for evaluation. All runs must have at least 5 rounds.")
        return

    df_stats = pd.DataFrame.from_dict(stats, orient='index')

    # ── Composite score ───────────────────────────────────────────────────
    # Each dimension is normalised to [0, 1] with 0.5 when all runs tie.
    norm = pd.DataFrame(index=df_stats.index)
    norm['acc_score']   = _safe_minmax_norm(df_stats['final_accuracy'],    higher_is_better=True)
    norm['auc_score']   = _safe_minmax_norm(df_stats['convergence_auc'],   higher_is_better=True)
    norm['speed_score'] = _safe_minmax_norm(df_stats['convergence_speed'], higher_is_better=False)
    norm['stab_score']  = _safe_minmax_norm(df_stats['instability'],       higher_is_better=False)

    if not df_stats['generalization_gap'].isna().all():
        norm['gap_score'] = _safe_minmax_norm(
            df_stats['generalization_gap'].fillna(df_stats['generalization_gap'].median()),
            higher_is_better=False
        )
    else:
        norm['gap_score'] = 0.5

    # Weights: final accuracy still dominates, but convergence speed and
    # full-run stability now each carry meaningful weight.
    df_stats['composite_score'] = (
        norm['acc_score']   * 0.35 +
        norm['auc_score']   * 0.20 +
        norm['speed_score'] * 0.20 +
        norm['stab_score']  * 0.15 +
        norm['gap_score']   * 0.10
    )

    df_stats = df_stats.sort_values(by='composite_score', ascending=False)

    # ── Print results ─────────────────────────────────────────────────────
    print("\n--- Performance Metrics Table ---")
    print(df_stats.round(4).to_string())

    best_run   = df_stats.index[0]
    runner_up  = df_stats.index[1] if len(df_stats) > 1 else None

    print("\n--- Evaluation Summary ---")
    print(f"Best Performing Method : {best_run}")
    print(f"Composite Score        : {df_stats.loc[best_run, 'composite_score']:.3f} / 1.000")

    reasons = []
    if runner_up is not None:
        acc_diff = (df_stats.loc[best_run, 'final_accuracy']
                    - df_stats.loc[runner_up, 'final_accuracy'])
        if norm.loc[best_run, 'acc_score'] > 0.7:
            reasons.append(
                f"Superior final accuracy ({df_stats.loc[best_run, 'final_accuracy']:.2%},"
                f" {acc_diff:+.2%} vs runner-up)."
            )
    if norm.loc[best_run, 'speed_score'] > 0.7:
        reasons.append(
            f"Faster convergence (reached 90 % of peak at"
            f" {df_stats.loc[best_run, 'convergence_speed']:.0%} of training)."
        )
    if norm.loc[best_run, 'stab_score'] > 0.7:
        reasons.append(
            f"Higher overall stability across all rounds"
            f" (full-run std: {df_stats.loc[best_run, 'instability']:.4f})."
        )
    if norm.loc[best_run, 'gap_score'] > 0.7 and not np.isnan(df_stats.loc[best_run, 'generalization_gap']):
        reasons.append(
            f"Lower generalisation gap over all rounds"
            f" (mean abs. gap: {df_stats.loc[best_run, 'generalization_gap']:.4f})."
        )

    if reasons:
        print("Reasons:")
        for r in reasons:
            print(f"  - {r}")
    else:
        print("  (Runs are closely matched — no single dimension stands out.)")

def plot_metrics(selected_metrics: list[str], dataframes: dict[str, pd.DataFrame]):
    if 'combined_acc_epochs' in selected_metrics:
        print("\n=> Generating Combined Train vs Test Accuracy & Epochs Plots...")
        valid_runs = {k: v for k, v in dataframes.items() if 'accuracy' in v.columns and 'training_accuracy' in v.columns and ('round_epochs' in v.columns or 'local_epochs' in v.columns)}
        
        n_runs = len(valid_runs)
        if n_runs > 0:
            rows, cols = get_grid_dimensions(n_runs)
            fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 5 * rows))
            
            if n_runs == 1:
                axes = [axes]
            else:
                axes = axes.flatten()
                
            for idx, (run_name, df) in enumerate(valid_runs.items()):
                ax1 = axes[idx]
                epoch_col = 'round_epochs' if 'round_epochs' in df.columns else 'local_epochs'
                
                df_sorted = df.sort_values(by='round')
                clean_label = format_plot_label(run_name)
                
                time_str = ""
                if 'time_round_total' in df.columns:
                    total_time = df['time_round_total'].sum()
                    avg_time = df['time_round_total'].mean()
                    time_str = f"\n(Avg: {avg_time:.1f}s | Total: {total_time:.1f}s)"
                
                ax1.set_title(f'{clean_label}{time_str}', fontweight='bold', fontsize=11)
                
                l1 = ax1.plot(df_sorted['round'], df_sorted['training_accuracy'], marker='o', linestyle='--', color='green', label='Train Accuracy', linewidth=2)
                l2 = ax1.plot(df_sorted['round'], df_sorted['accuracy'], marker='s', linestyle='-', color='purple', label='Test Accuracy', linewidth=2)
                
                ax1.set_xlabel('Round', fontweight='bold', fontsize=10)
                ax1.set_ylabel('Accuracy', fontweight='bold', fontsize=10)
                ax1.set_ylim(0.0, 1.05)
                ax1.set_axisbelow(True)
                ax1.grid(True, linestyle='--', alpha=0.7, color='gray')
                
                ax2 = ax1.twinx()
                bar_colors = ['darkgray' if r % 2 == 0 else 'lightgray' for r in df_sorted['round']]
                
                l3 = ax2.bar(df_sorted['round'], df_sorted[epoch_col], alpha=0.5, color=bar_colors, edgecolor='black', linewidth=1.2, label='Epochs per Round', width=1.0)
                ax2.set_ylabel('Local Epochs', fontweight='bold', fontsize=10)
                max_ep = df_sorted[epoch_col].max()
                ax2.set_ylim(0, max(max_ep * 1.5, 25))
                
                if idx == 0:
                    lines = l1 + l2 + [l3]
                    labels = [l.get_label() for l in lines]
                    fig.legend(lines, labels, loc='upper center', bbox_to_anchor=(0.5, 1.05), ncol=3, framealpha=0.9, fontsize=11)
            
            for idx in range(n_runs, len(axes)):
                fig.delaxes(axes[idx])
                
            plt.suptitle('Accuracy & Epochs', fontweight='bold', y=1.02 if n_runs > 1 else 1.1, fontsize=14)
            plt.tight_layout()
            plt.show()
                
        selected_metrics.remove('combined_acc_epochs')

    for metric in selected_metrics:
        valid_runs = {k: v for k, v in dataframes.items() if metric in v.columns and 'round' in v.columns}
        n_runs = len(valid_runs)
        
        if n_runs == 0:
            continue
            
        rows, cols = get_grid_dimensions(n_runs)
        fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 5 * rows))
        
        if n_runs == 1:
            axes = [axes]
        else:
            axes = axes.flatten()
            
        metric_title = metric.replace("_", " ").title()
        
        for idx, (run_name, df) in enumerate(valid_runs.items()):
            ax = axes[idx]
            df_sorted = df.sort_values(by='round')
            clean_label = format_plot_label(run_name)
            
            if 'time_round_total' in df.columns:
                avg_time = df['time_round_total'].mean()
                clean_label += f" (Avg: {avg_time:.1f}s)"
                
            ax.plot(df_sorted['round'], df_sorted[metric], marker='o', markersize=4, label=clean_label, linewidth=2)
            ax.set_title(clean_label, fontweight='bold', fontsize=11)
            ax.set_xlabel('Round', fontweight='bold', fontsize=10)
            ax.set_ylabel(metric_title, fontweight='bold', fontsize=10)
            
            if any(keyword in metric.lower() for keyword in ['loss', 'variance', 'norm', 'time']):
                ax.set_yscale('log')
                
            ax.grid(True, linestyle='--', alpha=0.7)
            
        for idx in range(n_runs, len(axes)):
            fig.delaxes(axes[idx])

        plt.suptitle(f'{metric_title} over Rounds', fontweight='bold', y=1.02, fontsize=14)
        plt.tight_layout()
        plt.show()

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    metrics_dir = os.path.normpath(os.path.join(script_dir, "results", "round_metrics"))
    
    print(f"Scanning for CSV files in: {metrics_dir}")
    csv_files = get_available_csv_files(metrics_dir)
    
    if not csv_files:
        print("No results found to process. Exiting.")
        return

    display_names = [os.path.basename(f).replace(".csv", "") for f in csv_files]

    while True:
        print("\n--- Available Runs ---")
        selected_file_indices = prompt_selection(
            display_names, 
            "\nEnter the numbers of the files you want to analyze (comma-separated)"
        )
        
        if selected_file_indices is None:
            print("Exiting.")
            break
        
        selected_files = [csv_files[i] for i in selected_file_indices]
        
        dataframes = {}
        for file_path in selected_files:
            run_name = os.path.basename(file_path).replace(".csv", "")
            dataframes[run_name] = pd.read_csv(file_path)

        print("\n--- Action Menu ---")
        print("[1] Plot Metrics")
        print("[2] Evaluate Methods (Compare Performance)")
        print("[q] Quit")
        
        action = input("Select an action: ").strip().lower()
        
        if action in ['q', 'quit']:
            print("Exiting.")
            break
            
        if action == '2':
            evaluate_methods(dataframes)
            continue
            
        if action == '1':
            available_metrics = extract_dynamic_metrics(dataframes)
            
            if not available_metrics:
                print("No plottable metrics found in the selected files.")
                continue

            print("\n--- Available Metrics ---")
            display_metrics = [m.replace("_", " ").title() for m in available_metrics]
            
            selected_metric_indices = prompt_selection(
                display_metrics, 
                "\nEnter the numbers of the metrics you want to plot (comma-separated)"
            )
            
            if selected_metric_indices is None:
                continue
            
            selected_metrics = [available_metrics[i] for i in selected_metric_indices]
            
            print("\nGenerating plots...")
            plot_metrics(selected_metrics, dataframes)

if __name__ == "__main__":
    main()