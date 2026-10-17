import json
import os
import glob
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd

COMPACT_SERIES_FIELDS = frozenset(
    {
        "epochs_per_round",
        "variance_per_round",
        "accuracy_per_round",
        "training_loss_per_round",
    }
)

DATASET_STEP_UNIT = {
    "mnli": 50,
    "cola": 22,
    "mrpc": 50,
    "qnli": 56,
    "sst2": 88,
    "rte": 41,
}


def format_report_json(value: Any, level: int = 0, compact: bool = False) -> str:
    """Format JSON with compact series for LLM readability (matching assess.py style)."""
    if compact:
        return json.dumps(value)

    indentation = " " * (level * 4)
    child_indentation = " " * ((level + 1) * 4)

    if isinstance(value, dict):
        if not value:
            return "{}"
        entries = [
            (
                f"{child_indentation}{json.dumps(key)}: "
                f"{format_report_json(item, level + 1, key in COMPACT_SERIES_FIELDS)}"
            )
            for key, item in value.items()
        ]
        return "{\n" + ",\n".join(entries) + f"\n{indentation}}}"

    if isinstance(value, list):
        if not value:
            return "[]"
        entries = [
            f"{child_indentation}{format_report_json(item, level + 1)}"
            for item in value
        ]
        return "[\n" + ",\n".join(entries) + f"\n{indentation}]"

    return json.dumps(value)


def analyze_fda_opt_experiments(
    csv_dir: str = "FDA-Opt/results/round_metrics",
    output_file: str = "fda_opt_evaluation_payload.json",
) -> dict[str, Any]:
    csv_path = Path(csv_dir)
    if not csv_path.exists():
        print(f"Directory {csv_path} does not exist.")
        return {}

    run_records: list[dict[str, Any]] = []

    for file_path in sorted(csv_path.glob("*.csv")):
        try:
            df = pd.read_csv(file_path)
            if df.empty or "fda" not in df.columns or not bool(df["fda"].iloc[0]):
                continue

            if "round" in df.columns:
                df = df.sort_values("round").reset_index(drop=True)

            run_id = file_path.stem
            dataset = str(df["ds_name"].iloc[0])
            emax = int(df["max_epochs"].iloc[0])
            target_midpoint = emax / 2.0
            total_rounds = len(df)

            # Determine steps per epoch
            unit = DATASET_STEP_UNIT.get(dataset, df["round_steps"].iloc[0])
            raw_epochs = (df["round_steps"] / unit).round().astype(int).tolist()
            epochs_per_round = [max(1, min(emax, int(e))) for e in raw_epochs]

            # Primary metric
            primary_col = "accuracy"
            if dataset == "cola":
                primary_col = "matthews_correlation"
            elif dataset == "mnli" and {"accuracy_m", "accuracy_mm"}.issubset(df.columns):
                df["accuracy"] = (df["accuracy_m"] + df["accuracy_mm"]) / 2.0

            accuracy_per_round = [
                float(val) if pd.notna(val) else None
                for val in df.get(primary_col, [None] * len(df))
            ]
            training_loss_per_round = [
                float(val) if pd.notna(val) else None
                for val in df.get("training_loss", [None] * len(df))
            ]
            variance_per_round = [
                float(val) if pd.notna(val) else None
                for val in df.get("variance", [None] * len(df))
            ]

            # Oscillation Metrics
            overshoot_rounds = [
                r + 1 for r, ep in enumerate(epochs_per_round) if ep > target_midpoint
            ]
            overshoot_count = len(overshoot_rounds)
            overshoot_pct = (overshoot_count / total_rounds) * 100.0

            rounds_at_emax = sum(1 for ep in epochs_per_round if ep == emax)
            rounds_at_emin = sum(1 for ep in epochs_per_round if ep == 1)

            diffs = [
                epochs_per_round[i + 1] - epochs_per_round[i]
                for i in range(total_rounds - 1)
            ]
            abs_diffs = [abs(d) for d in diffs]
            mean_swing = float(np.mean(abs_diffs)) if abs_diffs else 0.0
            max_swing = max(abs_diffs) if abs_diffs else 0

            # Direction reversals (sign flips where delta_{t+1} * delta_t < 0)
            reversals = sum(
                1 for i in range(len(diffs) - 1) if diffs[i] * diffs[i + 1] < 0
            )
            reversal_pct = (
                (reversals / (len(diffs) - 1)) * 100.0 if len(diffs) > 1 else 0.0
            )

            # Boom-bust swings: round-to-round jump >= 40% of E_max
            boom_bust_threshold = max(2, int(round(0.4 * emax)))
            boom_bust_count = sum(1 for d in abs_diffs if d >= boom_bust_threshold)
            has_boom_bust = boom_bust_count >= 3

            record: dict[str, Any] = {
                "run_id": run_id,
                "dataset": dataset,
                "model": str(df["checkpoint"].iloc[0]),
                "num_clients": int(df["num_clients"].iloc[0]),
                "clients_per_round": int(df["clients_per_round"].iloc[0]),
                "server_opt": str(df.get("server_opt_name", ["Unknown"])[0]),
                "client_opt": str(df.get("client_opt_name", ["Unknown"])[0]),
                "emax": emax,
                "target_midpoint": target_midpoint,
                "total_rounds": total_rounds,
                "mean_epochs": float(np.mean(epochs_per_round)),
                "std_epochs": float(np.std(epochs_per_round)),
                "min_epoch": min(epochs_per_round),
                "max_epoch": max(epochs_per_round),
                "overshoot_count": overshoot_count,
                "overshoot_pct": round(overshoot_pct, 2),
                "rounds_at_emax": rounds_at_emax,
                "rounds_at_emin": rounds_at_emin,
                "reversal_count": reversals,
                "reversal_pct": round(reversal_pct, 2),
                "mean_round_swing": round(mean_swing, 2),
                "max_round_swing": max_swing,
                "boom_bust_count": boom_bust_count,
                "has_boom_bust_oscillations": has_boom_bust,
                "epochs_per_round": epochs_per_round,
                "variance_per_round": variance_per_round,
                "accuracy_per_round": accuracy_per_round,
                "training_loss_per_round": training_loss_per_round,
            }
            run_records.append(record)
        except Exception as e:
            print(f"Error parsing {file_path.name}: {e}")

    # Global summary statistics
    n_runs = len(run_records)
    runs_with_overshoot = sum(1 for r in run_records if r["overshoot_count"] > 0)
    runs_with_boom_bust = sum(1 for r in run_records if r["has_boom_bust_oscillations"])
    runs_hitting_emax = sum(1 for r in run_records if r["rounds_at_emax"] > 0)

    avg_overshoot_pct = float(np.mean([r["overshoot_pct"] for r in run_records]))
    avg_reversal_pct = float(np.mean([r["reversal_pct"] for r in run_records]))
    avg_mean_swing = float(np.mean([r["mean_round_swing"] for r in run_records]))

    summary = {
        "total_fda_runs": n_runs,
        "runs_exceeding_midpoint": runs_with_overshoot,
        "pct_runs_exceeding_midpoint": round(runs_with_overshoot / n_runs * 100.0, 2),
        "runs_hitting_emax": runs_hitting_emax,
        "pct_runs_hitting_emax": round(runs_hitting_emax / n_runs * 100.0, 2),
        "runs_with_boom_bust_oscillations": runs_with_boom_bust,
        "pct_runs_with_boom_bust_oscillations": round(runs_with_boom_bust / n_runs * 100.0, 2),
        "mean_pct_rounds_above_midpoint": round(avg_overshoot_pct, 2),
        "mean_reversal_rate_pct": round(avg_reversal_pct, 2),
        "mean_round_to_round_swing": round(avg_mean_swing, 2),
    }

    full_payload = {
        "summary": summary,
        "runs": run_records,
    }

    with open(output_file, "w", encoding="utf-8") as f:
        f.write(format_report_json(full_payload))

    print(f"Saved evaluation payload to {output_file} ({n_runs} runs).")
    print(json.dumps(summary, indent=2))
    return full_payload


if __name__ == "__main__":
    analyze_fda_opt_experiments()
