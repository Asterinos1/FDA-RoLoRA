from fdaopt.utils import np, random, DataLoader, AutoTokenizer, DataCollatorWithPadding, load_dataset
def calculate_prior_distribution(raw_train_dataset):
    """
    Calculate the prior distribution of labels in the training dataset.
    This function calculates the proportion of each label in the training dataset,
    which is used as the prior distribution for Dirichlet sampling.
    Args:
        raw_train_dataset (datasets.arrow_dataset.Dataset): The training dataset.
    Returns:
        tuple: A tuple containing:
            - prior_distribution (numpy.ndarray): An array of label proportions.
            - num_labels (int): The number of unique labels in the dataset.
    """
    label_count_dict = dict(raw_train_dataset.to_pandas()['label'].value_counts())
    n = len(raw_train_dataset)
    label_count_tuple = sorted(label_count_dict.items(), key=lambda x: x[0])
    prior_distribution = np.array([c / n for _, c in label_count_tuple])
    num_labels = len(label_count_dict)
    return prior_distribution, num_labels
def sample_to_client(client_sample_counts, sample_label):
    """
    Assign a sample to a client based on the sample's label and the remaining counts for each client.
    This function assigns a sample to the client that still needs more samples of the given label.
    It decrements the count for that label for the chosen client. If no client has a remaining count
    of at least 1 for the label, the function assigns the sample to the client with the largest leftover count
    for that label.
    Args:
        client_sample_counts (numpy.ndarray): A 2D array where each row corresponds to a client and each column corresponds
            to the count (float) of samples needed for each label. The element at (i, j) represents the decimal number of
            samples of label j that client i still needs.
        sample_label (int): The label of the sample to be assigned to a client.
    Returns:
        int: The index of the client to which the sample has been assigned.
    """
    num_clients = len(client_sample_counts)
    client_indices = np.random.permutation(num_clients)
    for client_idx in client_indices:
        client_sample_count = client_sample_counts[client_idx]
        if client_sample_count[sample_label] >= 1:
            client_sample_count[sample_label] -= 1
            return client_idx
    client_idx = np.argmax(client_sample_counts[:, sample_label])
    client_sample_counts[client_idx][sample_label] -= 1
    return client_idx
def federated_dirichlet_datasets(raw_train_dataset, prior_distribution, num_clients, alpha):
    """
    Create federated datasets using Dirichlet-distributed label partitions.
    This function partitions the training dataset into multiple subsets, each corresponding
    to a client. The label distribution for each client is drawn from a Dirichlet distribution
    parameterized by the given prior distribution and concentration parameter alpha.
    The function also aims to keep the datasets as equal in size as possible.
    Args:
        raw_train_dataset (datasets.arrow_dataset.Dataset): The training dataset.
        prior_distribution (numpy.ndarray): An array representing the prior distribution of labels.
        num_clients (int): The number of clients.
        alpha (float): The concentration parameter for the Dirichlet distribution.
    Returns:
        list: A list of `datasets.arrow_dataset.Dataset` objects, each representing a client's dataset.
    """
    n = len(raw_train_dataset)
    client_num_samples = n / num_clients
    client_distributions = np.random.dirichlet(alpha * prior_distribution, num_clients)
    client_sample_indices = [[] for i in range(num_clients)]
    client_sample_counts = np.array([
        client_distributions[client_idx, :] * client_num_samples
        for client_idx in range(num_clients)
    ])
    for sample_idx, sample in enumerate(raw_train_dataset):
        client_idx = sample_to_client(client_sample_counts, sample['label'])
        client_sample_indices[client_idx].append(sample_idx)
    client_datasets = []
    for client_indices in client_sample_indices:
        client_dataset = raw_train_dataset.select(client_indices)
        client_datasets.append(client_dataset)
    return client_datasets
def tokenize_client_datasets(client_datasets, tokenize_fn):
    """
    Tokenize and preprocess a list of client datasets.
    This function tokenizes and preprocesses each dataset in the provided list of client datasets.
    It applies the specified tokenization function, renames the "label" column to "labels",
    removes unnecessary columns, and sets the format to PyTorch tensors.
    Args:
        client_datasets (list of datasets.arrow_dataset.Dataset): A list of client datasets to be tokenized and preprocessed.
        tokenize_fn (function): A function that takes an example and returns its tokenized form.
    Returns:
        list of datasets.arrow_dataset.Dataset: A list of tokenized and preprocessed client datasets.
    """
    expected_columns = ['labels', 'input_ids', 'token_type_ids', 'attention_mask']
    tok_client_datasets = []
    for client_dataset in client_datasets:
        tok_client_dataset = client_dataset.map(tokenize_fn, batched=True)
        tok_client_dataset = tok_client_dataset.rename_column("label", "labels")
        columns_to_remove = [
            column for column in tok_client_dataset.column_names
            if column not in expected_columns
        ]
        tok_client_dataset = tok_client_dataset.remove_columns(columns_to_remove)
        tok_client_dataset.set_format("torch")
        tok_client_datasets.append(tok_client_dataset)
    return tok_client_datasets
def preprocess_test_dataset(raw_test_dataset, tokenize_fn, data_collator, batch_size):
    """
    Preprocess and tokenize the test dataset, then create a DataLoader for it.
    This function tokenizes and preprocesses the provided test dataset using the specified
    tokenization function. It renames the "label" column to "labels", removes unnecessary columns,
    sets the format to PyTorch tensors, and then creates a DataLoader for the test dataset.
    Args:
        raw_test_dataset (datasets.arrow_dataset.Dataset): The raw test dataset to be tokenized and preprocessed.
        tokenize_fn (function): A function that takes an example and returns its tokenized form.
        data_collator (transformers.DataCollator): A data collator to be used for padding and batching.
        batch_size (int): The batch size to be used by the DataLoader.
    Returns:
        DataLoader: A DataLoader for the tokenized and preprocessed test dataset.
    """
    expected_columns = ['labels', 'input_ids', 'token_type_ids', 'attention_mask']
    tok_test_dataset = raw_test_dataset.map(tokenize_fn, batched=True)
    tok_test_dataset = tok_test_dataset.rename_column("label", "labels")
    columns_to_remove = [
        column for column in tok_test_dataset.column_names
        if column not in expected_columns
    ]
    tok_test_dataset = tok_test_dataset.remove_columns(columns_to_remove)
    tok_test_dataset.set_format("torch")
    test_ds = DataLoader(
        tok_test_dataset, batch_size=batch_size, collate_fn=data_collator
    )
    return test_ds
def create_client_dataloaders(tok_client_datasets, batch_size, collate_fn, num_workers=0, pin_memory=False):
    """
    Create data loaders for a list of tokenized client datasets.
    This function takes a list of tokenized client datasets and creates a DataLoader for each dataset.
    The resulting data loaders are stored in a list and returned.
    Args:
        tok_client_datasets (list of datasets.arrow_dataset.Dataset): A list of tokenized client datasets.
        batch_size (int): The batch size to be used by the data loaders.
        collate_fn (function): A collate function to be used by the data loaders.
    Returns:
        list of DataLoader: A list of data loaders, each corresponding to a tokenized client dataset.
    """
    client_dataloaders = []
    for tok_client_dataset in tok_client_datasets:
        client_dataloader = DataLoader(
            tok_client_dataset, shuffle=True, batch_size=batch_size, collate_fn=collate_fn,
            num_workers=num_workers, pin_memory=pin_memory,
            persistent_workers=True if num_workers > 0 else False
        )
        client_dataloaders.append(client_dataloader)
    return client_dataloaders
def tokenize_function(ds_path, ds_name, tokenizer):
    """
    Return a tokenization function based on the dataset path and name.
    This function returns the appropriate tokenization function for the specified dataset.
    Currently, it supports the GLUE MRPC dataset.
    Args:
        ds_path (str): The path or identifier of the dataset.
        ds_name (str): The name of the dataset.
        tokenizer (AutoTokenizer): The tokenizer to be used.
    Returns:
        function: A tokenization function.
    """
    if ds_path == "glue" or ds_path == "nyu-mll/glue":
        if ds_name == "mrpc":
            return lambda example: tokenizer(example["sentence1"], example["sentence2"], truncation=True)
        if ds_name == "rte":
            return lambda example: tokenizer(example["sentence1"], example["sentence2"], truncation=True)
        if ds_name == "stsb":
            return lambda example: tokenizer(example["sentence1"], example["sentence2"], truncation=True)
        if ds_name == "cola":
            return lambda example: tokenizer(example["sentence"], truncation=True)
        if ds_name == "sst2":
            return lambda example: tokenizer(example["sentence"], truncation=True)
        if ds_name == "qnli":
            return lambda example: tokenizer(example["question"], example["sentence"], truncation=True)
        if ds_name == "mnli":
            return lambda example: tokenizer(example["premise"], example["hypothesis"], truncation=True)
    return None
def test_dataset_split(raw_datasets, ds_path, ds_name):
    """
    Split the raw dataset into the appropriate test set.
    This function extracts the test set from the provided raw dataset based on the dataset path.
    For the GLUE dataset, it returns the validation split as the test set following prior work.
    Args:
        raw_datasets (datasets.DatasetDict): The raw dataset containing multiple splits.
        ds_path (str): The path or identifier of the dataset.
    Returns:
        Dictionary with key the name of the test ds and value a datasets.Dataset from the extracted test dataset split.
    """
    if ds_path == 'nyu-mll/glue' or ds_path == 'glue':
        if ds_name == 'mnli':
            return {'mnli_matched': raw_datasets['validation_matched'],
                    'mnli_mismatched': raw_datasets['validation_mismatched']}
        else:
            return {ds_name: raw_datasets['validation']}
def prepare_federated_datasets(ds_path, ds_name, checkpoint, num_clients, alpha, batch_size, num_workers=0, pin_memory=False):
    raw_datasets = load_dataset(path=ds_path, name=ds_name)
    raw_train_dataset = raw_datasets['train']
    MAX_SAMPLES = 100000000
    train_limit = min(len(raw_train_dataset), MAX_SAMPLES)
    raw_train_dataset = raw_train_dataset.select(range(train_limit))
    if ds_name == 'mnist':
        classes_per_client = 1 if num_clients == 10 else 2
        client_datasets = extreme_non_iid_datasets(raw_train_dataset, num_clients, classes_per_client)
        tok_client_datasets = []
        for client_ds in client_datasets:
            tok_ds = client_ds.rename_column("label", "labels")
            tok_ds.set_format("torch", columns=["image", "labels"])
            tok_client_datasets.append(tok_ds)
        test_ds_formatted = raw_datasets['test'].rename_column("label", "labels")
        test_ds_formatted.set_format("torch", columns=["image", "labels"])
        client_dataloaders = create_client_dataloaders(tok_client_datasets, batch_size, None, num_workers, pin_memory)
        fed_ds = FederatedDataset(client_dataloaders)
        test_ds = {'mnist': DataLoader(test_ds_formatted, batch_size=batch_size)}
        return fed_ds, test_ds
    raw_test_datasets = test_dataset_split(raw_datasets, ds_path, ds_name)
    for split_name in raw_test_datasets.keys():
        test_limit = min(len(raw_test_datasets[split_name]), MAX_SAMPLES)
        raw_test_datasets[split_name] = raw_test_datasets[split_name].select(range(test_limit))
    prior_distribution, num_labels = calculate_prior_distribution(raw_train_dataset)
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
        tok_ds = client_ds.rename_column("label", "labels")
        columns_to_remove = [col for col in tok_ds.column_names if col not in expected_columns]
        tok_ds = tok_ds.remove_columns(columns_to_remove)
        tok_ds.set_format("torch")
        tok_client_datasets.append(tok_ds)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    client_dataloaders = create_client_dataloaders(tok_client_datasets, batch_size, data_collator, num_workers=num_workers, pin_memory=pin_memory)
    fed_ds = FederatedDataset(client_dataloaders)
    test_ds = {k: preprocess_test_dataset(v, tokenize_fn, data_collator, batch_size) for k, v in raw_test_datasets.items()}
    return fed_ds, test_ds
def extreme_non_iid_datasets(raw_train_dataset, num_clients, classes_per_client):
    import numpy as np
    df = raw_train_dataset.to_pandas()
    num_classes = len(df['label'].unique())
    client_datasets = []
    if num_clients == 10 and classes_per_client == 1:
        classes_assigned = [[i] for i in range(10)]
    elif num_clients == 5 and classes_per_client == 2:
        classes_assigned = [[2*i, 2*i+1] for i in range(5)]
    else:
        raise ValueError("Unsupported non-IID configuration.")
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
class FederatedDataset:
    """
    A class to handle federated datasets for training in a federated learning setup.
    This class encapsulates the logic for managing multiple client dataloaders and providing
    batched data for federated learning training loops.
    Args:
        client_dataloaders (list of DataLoader): A list of DataLoader objects, each corresponding to a client's dataset.
    """
    def __init__(self, client_dataloaders):
        self.client_dataloaders = client_dataloaders
        self.client_batch_generators = [
            self.dataloader_batch_generator(client_dl)
            for client_dl in client_dataloaders
        ]
    def epoch_steps(self, client_ids):
        """
        Determine the number of steps (batches) in an epoch for the given clients.
        Args:
            client_ids (list of int): A list of client IDs.
        Returns:
            int: The maximum number of steps (batches) for the given clients.
        """
        return max(
            len(self.client_dataloaders[client_id])
            for client_id in client_ids
        )
    def next_client_batch(self, client_id):
        """
        Retrieve the next batch of data for the specified client.
        Args:
            client_id (int): The ID of the client.
        Returns:
            dict: A batch of data from the client's DataLoader.
        """
        return next(
            self.client_batch_generators[client_id]
        )
    @staticmethod
    def dataloader_batch_generator(dataloader):
        """
        A generator that yields (unending) batches of data from a DataLoader.
        Args:
            dataloader (DataLoader): A DataLoader object.
        Yields:
            dict: A batch of data from the DataLoader.
        """
        while True:
            for batch in dataloader:
                yield batch
class ClientSampler:
    """
    A class that helps random sampling of clients in each round.
    Attributes:
        client_ids (list): A list of client IDs to be sampled.
        sample_size (int): The size of each group of sampled client IDs.
        num_clients (int): The total number of clients.
        client_sample_generator (generator): A generator that yields groups of client IDs.
    """
    def __init__(self, client_ids, sample_size):
        self.client_ids = client_ids
        self.sample_size = sample_size
        self.num_clients = len(client_ids)
        if self.num_clients % self.sample_size != 0:
            raise ValueError("The number of clients must be divisible with the sample size.")
        self.client_sample_generator = self.sample_generator()
    def sample(self):
        """
        Returns the next group of sampled client IDs.
        Returns:
            list: A list of sampled client IDs of length `sample_size`.
        """
        return next(self.client_sample_generator)
    def sample_generator(self):
        """
        A generator function to continuously generate groups of sampled client IDs.
        This function shuffles the client IDs and partitions them into groups of the specified sample size.
        It yields each group of sampled client IDs.
        Yields:
            list: A list of sampled client IDs of length `sample_size`.
        """
        while True:
            shuffled_client_ids = random.sample(self.client_ids, self.num_clients)
            groups = [shuffled_client_ids[i:i + self.sample_size] for i in range(0, self.num_clients, self.sample_size)]
            for sampled_client_ids in groups:
                yield sampled_client_ids

