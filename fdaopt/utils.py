"""
Utility helpers and runtime environment configuration for FDA-LoRA.

This module manages global hardware acceleration devices, memory consumption
profiles, and pseudo-random seed initialization for deterministic execution.
"""

import random
import torch
import numpy as np
from datasets import load_dataset
from datasets.utils.logging import set_verbosity_error
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding
import evaluate

# Suppress verbose datasets logging
set_verbosity_error()

# Global device handle configured at runtime (e.g. 'cuda:0', 'cpu')
DEVICE = None

# Hardware memory profile: 'performance', 'moderate', or 'low'
DEVICE_RAM_PROGRAM = 'moderate'


def set_seed(seed: int):
    """
    Set random seeds across NumPy, PyTorch, and Python random for reproducibility.

    Args:
        seed (int): Seed value to initialize pseudo-random number generators.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)


def set_device(device: str):
    """
    Configure the global compute device for model tensors and training loops.

    Args:
        device (str): PyTorch device identifier (e.g. 'cuda:0', 'cpu').
    """
    global DEVICE
    DEVICE = device
