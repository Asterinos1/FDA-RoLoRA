"""
Hyperparameter file loading and management utilities.

Provides helper routines to locate and parse JSON experiment configurations
from the repository's hyperparameters registry directory.
"""

import json
import os
from typing import Dict, Any

script_directory = os.path.dirname(os.path.abspath(__file__))
HYPERPARAM_DIR = os.path.normpath(os.path.join(script_directory, '../../hyperparameters'))


def get_hyperparameters(hyperparam_filename: str) -> Dict[str, Any]:
    """
    Load an experiment hyperparameter configuration dictionary from a JSON file.

    Args:
        hyperparam_filename (str): Name or path of the JSON file (e.g. '0.json').

    Returns:
        dict: Parsed hyperparameter key-value configuration.
    """
    file_path = os.path.join(HYPERPARAM_DIR, hyperparam_filename) if not os.path.isabs(hyperparam_filename) else hyperparam_filename
    with open(file_path, 'r', encoding='utf-8') as f:
        hyperparams = json.load(f)
    return hyperparams
