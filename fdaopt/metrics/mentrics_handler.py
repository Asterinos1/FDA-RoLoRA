import pandas as pd
import os
import hashlib
import json
script_dir = os.path.dirname(os.path.realpath(__file__))
tmp_dir = '../../results'
ROUND_METRICS_PATH = os.path.normpath(os.path.join(script_dir, f'{tmp_dir}/round_metrics'))
class MetricsHandler:
    def __init__(self, hyperparams):
        self.hyperparams = hyperparams
        params_str = json.dumps(self.hyperparams, sort_keys=True).encode('utf-8')
        short_hash = hashlib.md5(params_str).hexdigest()[:10]
        prefix = self.hyperparams.get('extras', 'metrics')
        self.filename = f"{prefix}_{short_hash}.csv"
        self.round_metrics = []
    def append_round_metrics(self, metrics_dict):
        self.round_metrics.append(metrics_dict)
    def save_metrics(self, filepath=ROUND_METRICS_PATH):
        round_metrics_with_id = [
            {**self.hyperparams, **round_metrics}
            for round_metrics in self.round_metrics
        ]
        df = pd.DataFrame(
            round_metrics_with_id,
            columns=list(self.hyperparams.keys()) + list(self.round_metrics[0].keys())
        )
        df.to_csv(f"{filepath}/{self.filename}")

