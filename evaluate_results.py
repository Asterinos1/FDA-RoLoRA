import os
import glob
import pandas as pd
import matplotlib.pyplot as plt

def evaluate_latest_run():
    metrics_dir = os.path.join("results", "round_metrics")
    
    csv_files = glob.glob(os.path.join(metrics_dir, "*.csv"))
    
    if not csv_files:
        print(f"No CSV files found in {metrics_dir}.")
        return

    latest_csv = max(csv_files, key=os.path.getctime)
    print(f"Analyzing results from: {latest_csv}\n")

    df = pd.read_csv(latest_csv)

    print("--- Quick Summary ---")
    print(f"Total Rounds Completed: {df['round'].max()}")
    if 'accuracy' in df.columns:
        print(f"Max Accuracy: {df['accuracy'].max():.4f}")
    if 'training_loss' in df.columns:
        print(f"Final Training Loss: {df['training_loss'].iloc[-1]:.4f}")
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    if 'accuracy' in df.columns:
        axes[0].plot(df['round'], df['accuracy'], marker='o', color='blue')
        axes[0].set_title('Accuracy over Rounds')
        axes[0].set_xlabel('Round')
        axes[0].set_ylabel('Accuracy')
        axes[0].grid(True, linestyle='--', alpha=0.7)
    else:
        axes[0].text(0.5, 0.5, 'Accuracy data not available', ha='center', va='center')

    if 'training_loss' in df.columns:
        axes[1].plot(df['round'], df['training_loss'], marker='s', color='red')
        axes[1].set_title('Training Loss over Rounds')
        axes[1].set_xlabel('Round')
        axes[1].set_ylabel('Loss')
        axes[1].set_yscale('log') 
        axes[1].grid(True, linestyle='--', alpha=0.7)
    else:
        axes[1].text(0.5, 0.5, 'Loss data not available', ha='center', va='center')

    if 'variance' in df.columns:
        axes[2].plot(df['round'], df['variance'], marker='^', color='green')
        axes[2].set_title('Model Variance over Rounds')
        axes[2].set_xlabel('Round')
        axes[2].set_ylabel('Variance')
        axes[2].set_yscale('log')
        axes[2].grid(True, linestyle='--', alpha=0.7)
    else:
        axes[2].text(0.5, 0.5, 'Variance data not available', ha='center', va='center')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    evaluate_latest_run()