import json
import re
from pathlib import Path
from typing import Any, TypedDict

import pandas as pd


class RunMetadata(TypedDict):
    method: str
    fepoch: int | None
    emin: int | None
    emax: int | None


EpochValue = int | float
COMPACT_SERIES_FIELDS = frozenset(
    {
        "epochs_per_round",
        "odd_epochs",
        "even_epochs",
        "test_accuracy_per_round",
        "train_accuracy_per_round",
        "variance_per_round",
        "odd_round_variance",
        "even_round_variance",
    }
)


def parse_run_id(run_id: str) -> RunMetadata:
    method_match = re.search(r"_METHOD-(.*?)_DS-", run_id, re.IGNORECASE)
    if method_match is None:
        method_match = re.search(
            r"(?:^|[-_])M-(.*?)(?=-LR(?:\d|[-_]))", run_id, re.IGNORECASE
        )

    fixed_epoch_match = re.search(
        r"(?:^|[-_])FEPOCH[-_]?(\d+)(?=[-_]|$)", run_id, re.IGNORECASE
    )
    epoch_range_match = re.search(
        r"(?:^|[-_])E(\d+)TOE?(\d+)(?=[-_]|$)", run_id, re.IGNORECASE
    )
    if fixed_epoch_match is None:
        fixed_epoch_match = re.search(
            r"(?:^|[-_])E(\d+)(?=[-_]|$)", run_id, re.IGNORECASE
        )

    emin_match = re.search(
        r"(?:^|[-_])EMIN[-_]?(\d+)(?=[-_]|$)", run_id, re.IGNORECASE
    )
    emax_match = re.search(
        r"(?:^|[-_])EMAX[-_]?(\d+)(?=[-_]|$)", run_id, re.IGNORECASE
    )

    emin = int(emin_match.group(1)) if emin_match else None
    emax = int(emax_match.group(1)) if emax_match else None
    if epoch_range_match is not None:
        emin = emin if emin is not None else int(epoch_range_match.group(1))
        emax = emax if emax is not None else int(epoch_range_match.group(2))

    return {
        "method": method_match.group(1) if method_match else "Unknown",
        "fepoch": int(fixed_epoch_match.group(1)) if fixed_epoch_match else None,
        "emin": emin,
        "emax": emax,
    }


def first_text_value(df: pd.DataFrame, columns: tuple[str, ...]) -> str | None:
    for column in columns:
        if column not in df.columns:
            continue
        values = df[column].dropna()
        if not values.empty:
            return str(values.iloc[0])
    return None


def first_integer_value(df: pd.DataFrame, column: str) -> int | None:
    if column not in df.columns:
        return None
    values = pd.to_numeric(df[column], errors="coerce").dropna()
    if values.empty:
        return None
    return int(values.iloc[0])


def normalize_epoch_value(value: Any) -> EpochValue | None:
    if pd.isna(value):
        return None
    numeric_value = float(value)
    return int(numeric_value) if numeric_value.is_integer() else numeric_value


def extract_epochs_per_round(
    df: pd.DataFrame, fixed_epoch: int | None
) -> list[EpochValue | None]:
    if "round_epochs" in df.columns:
        recorded_epochs = pd.to_numeric(df["round_epochs"], errors="coerce")
        if recorded_epochs.notna().any():
            return [normalize_epoch_value(value) for value in recorded_epochs]
    if fixed_epoch is not None:
        return [fixed_epoch] * len(df)
    return []


def split_series_by_round(
    df: pd.DataFrame, values: list[EpochValue | None]
) -> tuple[list[EpochValue | None], list[EpochValue | None]]:
    if "round" not in df.columns or len(values) != len(df):
        return values[::2], values[1::2]

    round_numbers = pd.to_numeric(df["round"], errors="coerce")
    if round_numbers.isna().any():
        return values[::2], values[1::2]

    odd_values: list[EpochValue | None] = []
    even_values: list[EpochValue | None] = []
    for round_number, value in zip(round_numbers, values, strict=True):
        target = odd_values if int(round_number) % 2 == 1 else even_values
        target.append(value)
    return odd_values, even_values


def extract_metric_series(
    df: pd.DataFrame, columns: tuple[str, ...]
) -> list[float | None]:
    for column in columns:
        if column not in df.columns:
            continue
        values = pd.to_numeric(df[column], errors="coerce")
        return [None if pd.isna(value) else float(value) for value in values]
    return []


def sum_epochs_through_index(
    epochs_per_round: list[EpochValue | None], index: int | None
) -> EpochValue | None:
    if index is None or index < 0 or index >= len(epochs_per_round):
        return None
    epochs_to_index = epochs_per_round[: index + 1]
    if any(epoch is None for epoch in epochs_to_index):
        return None
    return normalize_epoch_value(
        sum(epoch for epoch in epochs_to_index if epoch is not None)
    )


def format_report_json(value: Any, level: int = 0, compact: bool = False) -> str:
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

def generate_llm_report(
    csv_dir: str | Path = "results/round_metrics",
    output_file: str | Path = "new_llm_evaluation_payload.json",
) -> None:
    csv_path = Path(csv_dir)
    if not csv_path.exists():
        print(f"Directory {csv_path} does not exist.")
        return

    report: dict[str, list[dict[str, Any]]] = {"runs": []}

    for file_path in sorted(csv_path.glob("*.csv")):
        df = pd.read_csv(file_path)
        if df.empty:
            continue
        if "round" in df.columns:
            df = df.sort_values("round").reset_index(drop=True)

        run_id = first_text_value(df, ("extras",)) or file_path.stem
        parsed_meta = parse_run_id(run_id)
        method = (
            first_text_value(df, ("method", "METHOD_NAME"))
            or parsed_meta["method"]
        )
        dataset = first_text_value(df, ("ds_name",)) or "Unknown"

        if (
            parsed_meta["fepoch"] is None
            and parsed_meta["emin"] is None
            and parsed_meta["emax"] is None
        ):
            parsed_meta["fepoch"] = first_integer_value(df, "local_epochs")

        epochs_per_round = extract_epochs_per_round(df, parsed_meta["fepoch"])
        odd_epochs, even_epochs = split_series_by_round(df, epochs_per_round)
        valid_epochs = [epoch for epoch in epochs_per_round if epoch is not None]
        mean_epochs_per_round = (
            float(sum(valid_epochs) / len(valid_epochs)) if valid_epochs else None
        )

        primary_metric = "accuracy"
        if dataset == "cola":
            primary_metric = "matthews_correlation"
        elif dataset == "mnli" and {
            "accuracy_m",
            "accuracy_mm",
        }.issubset(df.columns):
            df["accuracy"] = (df["accuracy_m"] + df["accuracy_mm"]) / 2.0

        test_accuracy_per_round = extract_metric_series(
            df, ("accuracy", "testing_accuracy", "test_accuracy")
        )
        train_accuracy_per_round = extract_metric_series(
            df, ("training_accuracy", "train_accuracy")
        )
        variance_per_round = extract_metric_series(df, ("variance",))
        odd_round_variance, even_round_variance = split_series_by_round(
            df, variance_per_round
        )

        total_rounds = int(df["round"].max())
        best_idx: int | None = None
        if primary_metric in df.columns:
            primary_metric_values = pd.to_numeric(df[primary_metric], errors="coerce")
            if primary_metric_values.notna().any():
                best_idx = int(primary_metric_values.idxmax())
        max_accuracy = (
            float(df.loc[best_idx, primary_metric]) if best_idx is not None else None
        )
        best_round = (
            int(df.loc[best_idx, "round"]) if best_idx is not None else None
        )
        epochs_to_best_accuracy = sum_epochs_through_index(
            epochs_per_round, best_idx
        )

        run_stats: dict[str, Any] = {
            "run_id": run_id,
            "method": method,
            "dataset": dataset,
            "fepoch": parsed_meta["fepoch"],
            "emin": parsed_meta["emin"],
            "emax": parsed_meta["emax"],
            "epochs_per_round": epochs_per_round,
            "odd_epochs": odd_epochs,
            "even_epochs": even_epochs,
            "test_accuracy_per_round": test_accuracy_per_round,
            "train_accuracy_per_round": train_accuracy_per_round,
            "variance_per_round": variance_per_round,
            "odd_round_variance": odd_round_variance,
            "even_round_variance": even_round_variance,
            "mean_epochs_per_round": mean_epochs_per_round,
            "total_rounds": total_rounds,
            "early_stopped": total_rounds < 70,
            "best_round": best_round,
            "rounds_to_best_accuracy": best_round,
            "epochs_to_best_accuracy": epochs_to_best_accuracy,
            "max_accuracy": max_accuracy,
            "min_testing_loss": (
                float(df["testing_loss"].min())
                if "testing_loss" in df.columns
                else None
            ),
            "min_training_loss": (
                float(df["training_loss"].min())
                if "training_loss" in df.columns
                else None
            ),
            "total_time_seconds": (
                float(df["time_round_total"].sum())
                if "time_round_total" in df.columns
                else None
            ),
            "time_to_best_round_seconds": (
                float(df.loc[:best_idx, "time_round_total"].sum())
                if best_idx is not None and "time_round_total" in df.columns
                else None
            ),
            "average_variance": (
                float(df["variance"].mean()) if "variance" in df.columns else None
            ),
            "total_computation_steps": (
                int(df["round_steps"].sum()) if "round_steps" in df.columns else None
            ),
            "computation_steps_to_best_round": (
                int(df.loc[:best_idx, "round_steps"].sum())
                if best_idx is not None and "round_steps" in df.columns
                else None
            ),
        }
        report["runs"].append(run_stats)

    with Path(output_file).open("w", encoding="utf-8") as output:
        output.write(format_report_json(report))

if __name__ == "__main__":
    generate_llm_report()
