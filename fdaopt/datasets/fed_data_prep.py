"""
Federated dataset preparation, Dirichlet non-IID partitioning, and client sampling.

Provides utilities for:
  - Dirichlet non-IID label distribution partitioning across arbitrary numbers of clients.
  - Extreme non-IID partition synthesis for benchmark datasets (e.g. MNIST).
  - Hugging Face dataset tokenization, column formatting, and PyTorch DataLoader creation.
  - Infinite cyclic batch generators (`FederatedDataset`) and balanced round client samplers (`ClientSampler`).
"""

import random
from typing import List, Dict, Tuple, Any, Optional, Generator
import numpy as np
import torch
from torch.utils.data import DataLoader
from datasets import load_dataset, Dataset, DatasetDict
from transformers import AutoTokenizer, DataCollatorWithPadding

from fdaopt import utils


def calculate_prior_distribution(raw_train_dataset: Dataset) -> Tuple[np.ndarray, int]:
    """
    Calculate the empirical prior distribution of class labels in the training dataset.

    Args:
        raw_train_dataset (Dataset): Hugging Face Dataset containing a 'label' column.

    Returns:
        tuple: (prior_distribution_array, num_classes)
    """
    label_count_dict = dict(raw_train_dataset.to_pandas()['label'].value_counts())
    n = len(raw_train_dataset)
    label_count_tuple = sorted(label_count_dict.items(), key=lambda x: x[0])
    prior_distribution = np.array([c / n for _, c in label_count_tuple])
    num_labels = len(label_count_dict)
    return prior_distribution, num_labels


def sample_to_client(client_sample_counts: np.ndarray, sample_label: int) -> int:
    """
    Assign a dataset sample to a client based on remaining quota from Dirichlet distribution.

    Args:
        client_sample_counts (np.ndarray): 2D float array of shape (num_clients, num_classes)
            tracking the remaining quota of samples needed for each class by each client.
        sample_label (int): Integer class label of the current sample.

    Returns:
        int: Index of the chosen client.
    """
    num_clients = len(client_sample_counts)
    client_indices = np.random.permutation(num_clients)

    for client_idx in client_indices:
        if client_sample_counts[client_idx, sample_label] >= 1:
            client_sample_counts[client_idx, sample_label] -= 1
            return client_idx

    # If all quotas are exhausted below 1.0, assign to client with largest residual
    client_idx = int(np.argmax(client_sample_counts[:, sample_label]))
    client_sample_counts[client_idx, sample_label] -= 1
    return client_idx


def federated_dirichlet_datasets(
    raw_train_dataset: Dataset,
    prior_distribution: np.ndarray,
    num_clients: int,
    alpha: float
) -> List[Dataset]:
    """
    Partition a dataset across clients using a Dirichlet distribution: p_k ~ Dir(alpha * p).

    Args:
        raw_train_dataset (Dataset): Hugging Face Dataset to partition.
        prior_distribution (np.ndarray): Base label prior distribution array.
        num_clients (int): Total number of clients.
        alpha (float): Concentration parameter (smaller alpha = higher non-IID heterogeneity).

    Returns:
        list of Dataset: Subsets assigned to each client.
    """
    n = len(raw_train_dataset)
    client_num_samples = n / num_clients
    client_distributions = np.random.dirichlet(alpha * prior_distribution, num_clients)
    client_sample_indices: List[List[int]] = [[] for _ in range(num_clients)]

    client_sample_counts = np.array([
        client_distributions[client_idx, :] * client_num_samples
        for client_idx in range(num_clients)
    ])

    for sample_idx, sample in enumerate(raw_train_dataset):
        client_idx = sample_to_client(client_sample_counts, sample['label'])
        client_sample_indices[client_idx].append(sample_idx)

    client_datasets = [raw_train_dataset.select(indices) for indices in client_sample_indices]
    return client_datasets


def tokenize_client_datasets(client_datasets: List[Dataset], tokenize_fn) -> List[Dataset]:
    """
    Tokenize and format a list of client datasets for PyTorch tensor ingestion.

    Args:
        client_datasets (list of Dataset): Raw client datasets.
        tokenize_fn: Tokenization mapping function.

    Returns:
        list of Dataset: Formatted tokenized datasets.
    """
    expected_columns = ['labels', 'input_ids', 'token_type_ids', 'attention_mask']
    tok_client_datasets = []

    for client_dataset in client_datasets:
        tok_client_dataset = client_dataset.map(tokenize_fn, batched=True)
        if "label" in tok_client_dataset.column_names:
            tok_client_dataset = tok_client_dataset.rename_column("label", "labels")
        columns_to_remove = [
            column for column in tok_client_dataset.column_names
            if column not in expected_columns
        ]
        tok_client_dataset = tok_client_dataset.remove_columns(columns_to_remove)
        tok_client_dataset.set_format("torch")
        tok_client_datasets.append(tok_client_dataset)

    return tok_client_datasets


def preprocess_test_dataset(
    raw_test_dataset: Dataset,
    tokenize_fn,
    data_collator: DataCollatorWithPadding,
    batch_size: int
) -> DataLoader:
    """
    Tokenize and create a DataLoader for evaluation / test datasets.

    Args:
        raw_test_dataset (Dataset): Test or validation dataset split.
        tokenize_fn: Tokenization function.
        data_collator (DataCollatorWithPadding): Batch collator with padding.
        batch_size (int): Evaluation batch size.

    Returns:
        DataLoader: Test DataLoader.
    """
    expected_columns = ['labels', 'input_ids', 'token_type_ids', 'attention_mask']
    tok_test_dataset = raw_test_dataset.map(tokenize_fn, batched=True)
    if "label" in tok_test_dataset.column_names:
        tok_test_dataset = tok_test_dataset.rename_column("label", "labels")
    columns_to_remove = [
        column for column in tok_test_dataset.column_names
        if column not in expected_columns
    ]
    tok_test_dataset = tok_test_dataset.remove_columns(columns_to_remove)
    tok_test_dataset.set_format("torch")

    test_ds = DataLoader(
        tok_test_dataset,
        batch_size=batch_size,
        collate_fn=data_collator
    )
    return test_ds


def create_client_dataloaders(
    tok_client_datasets: List[Dataset],
    batch_size: int,
    collate_fn: Optional[Any],
    num_workers: int = 0,
    pin_memory: bool = False
) -> List[DataLoader]:
    """
    Instantiate PyTorch DataLoader objects for all tokenized client datasets.

    Args:
        tok_client_datasets (list of Dataset): Preprocessed client datasets.
        batch_size (int): Mini-batch size.
        collate_fn (Optional): Collate function for batch padding.
        num_workers (int): Number of background subprocess workers.
        pin_memory (bool): If True, pin memory for faster GPU tensor transfer.

    Returns:
        list of DataLoader: Client data loaders.
    """
    client_dataloaders = []
    for tok_client_dataset in tok_client_datasets:
        client_dataloader = DataLoader(
            tok_client_dataset,
            shuffle=True,
            batch_size=batch_size,
            collate_fn=collate_fn,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=True if num_workers > 0 else False
        )
        client_dataloaders.append(client_dataloader)
    return client_dataloaders


def tokenize_function(ds_path: str, ds_name: str, tokenizer: AutoTokenizer):
    """
    Construct dataset-specific text tokenization lambda for GLUE NLP tasks.

    Args:
        ds_path (str): Dataset repository path (e.g. 'glue').
        ds_name (str): GLUE benchmark task (e.g. 'mrpc', 'sst2', 'mnli', 'qnli', 'rte').
        tokenizer (AutoTokenizer): Pretrained Hugging Face tokenizer.

    Returns:
        Callable: Tokenization mapping function.
    """
    if ds_path in ("glue", "nyu-mll/glue"):
        if ds_name in ("mrpc", "rte", "stsb"):
            return lambda example: tokenizer(example["sentence1"], example["sentence2"], truncation=True)
        if ds_name in ("cola", "sst2"):
            return lambda example: tokenizer(example["sentence"], truncation=True)
        if ds_name == "qnli":
            return lambda example: tokenizer(example["question"], example["sentence"], truncation=True)
        if ds_name == "mnli":
            return lambda example: tokenizer(example["premise"], example["hypothesis"], truncation=True)
    return None


def test_dataset_split(raw_datasets: DatasetDict, ds_path: str, ds_name: str) -> Dict[str, Dataset]:
    """
    Extract standard validation / test splits matching NLP federated literature benchmarks.

    Args:
        raw_datasets (DatasetDict): Loaded raw dataset dictionary.
        ds_path (str): Dataset path.
        ds_name (str): Dataset task name.

    Returns:
        dict: Map of split names to Dataset objects.
    """
    if ds_path in ('nyu-mll/glue', 'glue'):
        if ds_name == 'mnli':
            return {
                'mnli_matched': raw_datasets['validation_matched'],
                'mnli_mismatched': raw_datasets['validation_mismatched']
            }
        return {ds_name: raw_datasets['validation']}
    return {ds_name: raw_datasets.get('test', raw_datasets.get('validation'))}


def extreme_non_iid_datasets(raw_train_dataset: Dataset, num_clients: int, classes_per_client: int) -> List[Dataset]:
    """
    Partition dataset under strict class-exclusive allocation (e.g. exactly 1 or 2 classes per client).

    Args:
        raw_train_dataset (Dataset): Training dataset.
        num_clients (int): Total number of clients (must match partition mapping).
        classes_per_client (int): Number of distinct classes assigned per client.

    Returns:
        list of Dataset: Client partitions.

    Raises:
        ValueError: If partition configuration is unsupported.
    """
    df = raw_train_dataset.to_pandas()
    num_classes = len(df['label'].unique())
    client_datasets = []

    if num_clients == 10 and classes_per_client == 1:
        classes_assigned = [[i] for i in range(10)]
    elif num_clients == 5 and classes_per_client == 2:
        classes_assigned = [[2 * i, 2 * i + 1] for i in range(5)]
    else:
        raise ValueError(f"Unsupported extreme non-IID config: {num_clients} clients with {classes_per_client} classes/client.")

    samples_per_client = len(raw_train_dataset) // num_clients
    samples_per_class_per_client = samples_per_client // classes_per_client
    label_indices = {label: df.index[df['label'] == label].tolist() for label in range(num_classes)}

    for client_classes in classes_assigned:
        client_indices = []
        for c in client_classes:
            idx_to_take = label_indices[c][:samples_per_class_per_client]
            label_indices[c] = label_indices[c][samples_per_class_per_client:]
            client_indices.extend(idx_to_take)
        client_datasets.append(raw_train_dataset.select(client_indices))

    return client_datasets


def prepare_federated_datasets(
    ds_path: str,
    ds_name: str,
    checkpoint: str,
    num_clients: int,
    alpha: float,
    batch_size: int,
    num_workers: int = 0,
    pin_memory: bool = False
) -> Tuple['FederatedDataset', Dict[str, DataLoader]]:
    """
    End-to-end dataset pipeline: loading, Dirichlet/extreme partitioning, tokenization, and loader construction.

    Args:
        ds_path (str): Dataset repository path (e.g. 'glue', 'mnist').
        ds_name (str): Task name (e.g. 'mrpc', 'sst2', 'mnist').
        checkpoint (str): Pretrained backbone model name or path.
        num_clients (int): Number of federated clients.
        alpha (float): Dirichlet distribution alpha.
        batch_size (int): Local mini-batch size.
        num_workers (int): Background DataLoader workers.
        pin_memory (bool): Enable pinned CUDA host memory.

    Returns:
        tuple: (FederatedDataset, test_dataloader_dict)
    """
    raw_datasets = load_dataset(path=ds_path, name=ds_name)
    raw_train_dataset = raw_datasets['train']

    MAX_SAMPLES = 100000000
    train_limit = min(len(raw_train_dataset), MAX_SAMPLES)
    raw_train_dataset = raw_train_dataset.select(range(train_limit))

    # Fast-path handling for MNIST image datasets
    if ds_name == 'mnist':
        classes_per_client = 1 if num_clients == 10 else 2
        client_datasets = extreme_non_iid_datasets(raw_train_dataset, num_clients, classes_per_client)
        tok_client_datasets = []
        for client_ds in client_datasets:
            tok_ds = client_ds.rename_column("label", "labels") if "label" in client_ds.column_names else client_ds
            tok_ds.set_format("torch", columns=["image", "labels"])
            tok_client_datasets.append(tok_ds)

        test_ds_formatted = raw_datasets['test']
        if "label" in test_ds_formatted.column_names:
            test_ds_formatted = test_ds_formatted.rename_column("label", "labels")
        test_ds_formatted.set_format("torch", columns=["image", "labels"])

        client_dataloaders = create_client_dataloaders(tok_client_datasets, batch_size, None, num_workers, pin_memory)
        fed_ds = FederatedDataset(client_dataloaders)
        test_ds = {'mnist': DataLoader(test_ds_formatted, batch_size=batch_size)}
        return fed_ds, test_ds

    # Standard NLP pipeline with Hugging Face Tokenizers
    raw_test_datasets = test_dataset_split(raw_datasets, ds_path, ds_name)
    for split_name in raw_test_datasets.keys():
        test_limit = min(len(raw_test_datasets[split_name]), MAX_SAMPLES)
        raw_test_datasets[split_name] = raw_test_datasets[split_name].select(range(test_limit))

    prior_distribution, _ = calculate_prior_distribution(raw_train_dataset)

    if checkpoint in ['microsoft/deberta-v3-base', 'prajjwal1/bert-tiny']:
        tokenizer = AutoTokenizer.from_pretrained(checkpoint, use_fast=False)
    else:
        tokenizer = AutoTokenizer.from_pretrained(checkpoint)

    tokenize_fn = tokenize_function(ds_path, ds_name, tokenizer)
    tok_raw_train_dataset = raw_train_dataset.map(tokenize_fn, batched=True)
    client_datasets = federated_dirichlet_datasets(tok_raw_train_dataset, prior_distribution, num_clients, alpha)

    expected_columns = ['labels', 'input_ids', 'token_type_ids', 'attention_mask']
    tok_client_datasets = []
    for client_ds in client_datasets:
        tok_ds = client_ds.rename_column("label", "labels") if "label" in client_ds.column_names else client_ds
        columns_to_remove = [col for col in tok_ds.column_names if col not in expected_columns]
        tok_ds = tok_ds.remove_columns(columns_to_remove)
        tok_ds.set_format("torch")
        tok_client_datasets.append(tok_ds)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    client_dataloaders = create_client_dataloaders(
        tok_client_datasets, batch_size, data_collator, num_workers=num_workers, pin_memory=pin_memory
    )
    fed_ds = FederatedDataset(client_dataloaders)
    test_ds = {k: preprocess_test_dataset(v, tokenize_fn, data_collator, batch_size) for k, v in raw_test_datasets.items()}

    return fed_ds, test_ds


class FederatedDataset:
    """
    Encapsulates collection of client DataLoaders with cyclic batch generation.

    Attributes:
        client_dataloaders (list of DataLoader): Per-client PyTorch DataLoader instances.
        client_batch_generators (list of Generator): Infinite generator streams for each client.
    """

    def __init__(self, client_dataloaders: List[DataLoader]):
        """
        Initialize the federated dataset wrapper.

        Args:
            client_dataloaders (list of DataLoader): DataLoader instances for each partitioned client.
        """
        self.client_dataloaders = client_dataloaders
        self.client_batch_generators = [
            self.dataloader_batch_generator(client_dl)
            for client_dl in client_dataloaders
        ]

    def epoch_steps(self, client_ids: List[int]) -> int:
        """
        Calculate maximum mini-batch steps per local epoch across a group of clients.

        Args:
            client_ids (list of int): Client IDs participating in the round.

        Returns:
            int: Maximum step count across the provided clients.
        """
        return max(len(self.client_dataloaders[client_id]) for client_id in client_ids)

    def next_client_batch(self, client_id: int) -> Dict[str, torch.Tensor]:
        """
        Fetch the next mini-batch of tensors for a specific client.

        Args:
            client_id (int): Client index.

        Returns:
            dict: Batch dictionary of input and label tensors.
        """
        return next(self.client_batch_generators[client_id])

    @staticmethod
    def dataloader_batch_generator(dataloader: DataLoader) -> Generator[Dict[str, torch.Tensor], None, None]:
        """
        Create an infinite cyclic generator yielding batches from a DataLoader.

        Args:
            dataloader (DataLoader): PyTorch DataLoader.

        Yields:
            dict: Next mini-batch dictionary.
        """
        while True:
            for batch in dataloader:
                yield batch


class ClientSampler:
    """
    Uniform client sampler ensuring equal round participation without replacement.

    Attributes:
        client_ids (list of int): Pool of available client indices.
        sample_size (int): Number of clients sampled per round.
        num_clients (int): Total size of the client pool.
        client_sample_generator (Generator): Infinite generator yielding client ID batches.
    """

    def __init__(self, client_ids: List[int], sample_size: int):
        """
        Initialize the client sampler.

        Args:
            client_ids (list of int): Complete list of client IDs.
            sample_size (int): Number of clients sampled per communication round.

        Raises:
            ValueError: If total client count is not evenly divisible by sample size.
        """
        self.client_ids = client_ids
        self.sample_size = sample_size
        self.num_clients = len(client_ids)
        if self.num_clients % self.sample_size != 0:
            raise ValueError(f"Number of clients ({self.num_clients}) must be divisible by sample size ({self.sample_size}).")
        self.client_sample_generator = self.sample_generator()

    def sample(self) -> List[int]:
        """
        Retrieve the next randomly sampled group of client IDs for a communication round.

        Returns:
            list of int: Sampled client indices of length `sample_size`.
        """
        return next(self.client_sample_generator)

    def sample_generator(self) -> Generator[List[int], None, None]:
        """
        Infinite generator yielding shuffled, non-overlapping client partitions.

        Yields:
            list of int: Next group of sampled client IDs.
        """
        while True:
            shuffled_client_ids = random.sample(self.client_ids, self.num_clients)
            groups = [
                shuffled_client_ids[i:i + self.sample_size]
                for i in range(0, self.num_clients, self.sample_size)
            ]
            for sampled_client_ids in groups:
                yield sampled_client_ids
