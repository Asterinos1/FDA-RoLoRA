"""
Experiment metrics recording, serialization, and disk export handler.

Tracks per-round training and validation metrics across federated rounds,
associating hyperparameter configurations with unique MD5 experiment signatures
and exporting them to structured CSV files.
"""

import os
import json
import hashlib
from typing import Dict, Any, List
import pandas as pd

script_dir = os.path.dirname(os.path.realpath(__file__))
tmp_dir = '../../results'
ROUND_METRICS_PATH = os.path.normpath(os.path.join(script_dir, f'{tmp_dir}/round_metrics'))


class MetricsHandler:
    """
    Manages round-by-round metric tracking and CSV persistence.

    Attributes:
        hyperparams (dict): Experiment configuration parameters.
        filename (str): Unique CSV target filename derived from config hash.
        round_metrics (list of dict): Log of collected per-round metric dicts.
    """

    def __init__(self, hyperparams: Dict[str, Any]):
        """
        Initialize the metric handler with experiment hyperparameters and generate output filename.

        Args:
            hyperparams (dict): Dictionary of experiment hyperparameters.
        """
        self.hyperparams = hyperparams
        params_str = json.dumps(self.hyperparams, sort_keys=True).encode('utf-8')
        short_hash = hashlib.md5(params_str).hexdigest()[:10]
        prefix = self.hyperparams.get('extras', 'metrics')
        self.filename = f"{prefix}_{short_hash}.csv"
        self.round_metrics: List[Dict[str, Any]] = []

    def append_round_metrics(self, metrics_dict: Dict[str, Any]):
        """
        Append a dictionary of metrics for the completed communication round.

        Args:
            metrics_dict (dict): Metric key-value pairs (e.g. loss, accuracy, variance).
        """
        self.round_metrics.append(metrics_dict)

    def save_metrics(self, filepath: str = ROUND_METRICS_PATH):
        """
        Export accumulated round metrics combined with hyperparameters to a CSV file.

        Args:
            filepath (str): Destination directory for the metrics CSV file.
        """
        if not self.round_metrics:
            return

        os.makedirs(filepath, exist_ok=True)
        round_metrics_with_id = [
            {**self.hyperparams, **metrics}
            for metrics in self.round_metrics
        ]
        df = pd.DataFrame(
            round_metrics_with_id,
            columns=list(self.hyperparams.keys()) + list(self.round_metrics[0].keys())
        )
        df.to_csv(os.path.join(filepath, self.filename), index=False)
