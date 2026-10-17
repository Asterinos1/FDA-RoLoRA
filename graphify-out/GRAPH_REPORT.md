# Graph Report - .  (2026-08-14)

## Corpus Check
- 37 files · ~31,012 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 268 nodes · 537 edges · 22 communities (15 shown, 7 thin omitted)
- Extraction: 99% EXTRACTED · 1% INFERRED · 0% AMBIGUOUS · INFERRED: 7 edges (avg confidence: 0.91)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- [[_COMMUNITY_Federated Training & Client Sampling|Federated Training & Client Sampling]]
- [[_COMMUNITY_Federated Dataset Unit Tests|Federated Dataset Unit Tests]]
- [[_COMMUNITY_Federated Data Preparation Pipeline|Federated Data Preparation Pipeline]]
- [[_COMMUNITY_FDA Optimization & Tensor Operations|FDA Optimization & Tensor Operations]]
- [[_COMMUNITY_AMS Sketch & Training Utilities|AMS Sketch & Training Utilities]]
- [[_COMMUNITY_Core Documentation & Dependencies|Core Documentation & Dependencies]]
- [[_COMMUNITY_Metrics Visualization & Plotting|Metrics Visualization & Plotting]]
- [[_COMMUNITY_Dual-Track Variance Simulation|Dual-Track Variance Simulation]]
- [[_COMMUNITY_Multi-Seed Benchmarking & Grid Search|Multi-Seed Benchmarking & Grid Search]]
- [[_COMMUNITY_Client Sampler Test Suite|Client Sampler Test Suite]]
- [[_COMMUNITY_Dynamic Variance Simulation|Dynamic Variance Simulation]]
- [[_COMMUNITY_Multi-Process Simulation Engine|Multi-Process Simulation Engine]]
- [[_COMMUNITY_Hyperparameter Generation & Config|Hyperparameter Generation & Config]]
- [[_COMMUNITY_Client Sampler Generator|Client Sampler Generator]]
- [[_COMMUNITY_Main Entrypoint & Parameter Tools|Main Entrypoint & Parameter Tools]]
- [[_COMMUNITY_LLM Run Assessment|LLM Run Assessment]]
- [[_COMMUNITY_Latest Run Evaluator|Latest Run Evaluator]]
- [[_COMMUNITY_RoLoRA Comparison Test|RoLoRA Comparison Test]]

## God Nodes (most connected - your core abstractions)
1. `prepare_federated_datasets()` - 22 edges
2. `fda_rolora_opt()` - 20 edges
3. `fedavg_rolora()` - 20 edges
4. `copy_parameters()` - 19 edges
5. `ClientSampler` - 18 edges
6. `fda_opt()` - 18 edges
7. `fda_rolora_opt2()` - 18 edges
8. `MetricsHandler` - 17 edges
9. `fda_rolora_dual_track_opt()` - 17 edges
10. `fda_rolora_dual_track_ema_standard_opt()` - 17 edges

## Surprising Connections (you probably didn't know these)
- `Federated LoRA Architectures` --conceptually_related_to--> `Flower Federated Learning Framework (flwr)`  [INFERRED]
  README.md → requirements_old.txt
- `Federated LoRA Architectures` --conceptually_related_to--> `PEFT (Parameter-Efficient Fine-Tuning)`  [INFERRED]
  README.md → requirements.txt
- `run_fda_rolora_test()` --calls--> `fda_rolora_opt()`  [EXTRACTED]
  fdaopt/fda_rolora_test.py → fdaopt/training/fda_opt.py
- `run_fedavg_rolora_test()` --calls--> `fedavg_rolora()`  [EXTRACTED]
  fdaopt/rolora_test.py → fdaopt/training/fed_opt.py
- `fda_opt()` --calls--> `prepare_federated_datasets()`  [EXTRACTED]
  fdaopt/training/fda_opt.py → fdaopt/datasets/fed_data_prep.py

## Import Cycles
- None detected.

## Communities (22 total, 7 thin omitted)

### Community 0 - "Federated Training & Client Sampling"
Cohesion: 0.13
Nodes (48): ClientSampler, A class that helps random sampling of clients in each round.     Attributes:, run_fda_rolora_test(), run_fedavg_rolora_test(), MetricsHandler, check_dynamic_early_stopping(), compute_client_drifts(), compute_metrics() (+40 more)

### Community 1 - "Federated Dataset Unit Tests"
Cohesion: 0.10
Nodes (23): calculate_prior_distribution(), create_client_dataloaders(), extreme_non_iid_datasets(), federated_dirichlet_datasets(), FederatedDataset, prepare_federated_datasets(), preprocess_test_dataset(), Preprocess and tokenize the test dataset, then create a DataLoader for it. (+15 more)

### Community 2 - "Federated Data Preparation Pipeline"
Cohesion: 0.10
Nodes (23): calculate_prior_distribution(), create_client_dataloaders(), federated_dirichlet_datasets(), FederatedDataset, prepare_federated_datasets(), preprocess_test_dataset(), Tokenize and preprocess a list of client datasets.      This function tokenize, Preprocess and tokenize the test dataset, then create a DataLoader for it. (+15 more)

### Community 3 - "FDA Optimization & Tensor Operations"
Cohesion: 0.11
Nodes (22): average_client_parameters(), average_client_parameters2(), compute_drifts(), fda_sketch_estimation(), fda_variance_approx(), get_ksi_unit(), get_ksi_unit2(), get_save_device() (+14 more)

### Community 4 - "AMS Sketch & Training Utilities"
Cohesion: 0.12
Nodes (10): Set the seed for reproducibility.     Args:         seed (int): The seed value, set_seed(), get_opt_hyperparams(), Extracts optimizer hyperparameters from the given hyperparameters dictionary., AmsSketch, Assumed that x is tensor shaped (d,) , i.e., a vector (for example, indices, i.e, Efficient computation of sketch using PyTorch tensors.         Args:         -, PyTorch translation of the TensorFlow function using a simple for loop. (+2 more)

### Community 5 - "Core Documentation & Dependencies"
Cohesion: 0.15
Nodes (13): Fourier Domain Adaptation (FDA), FDA-LoRA Project, Federated LoRA Architectures, Hugging Face Accelerate, Hugging Face Datasets, Hugging Face Evaluate, Flower Federated Learning Framework (flwr), PEFT (Legacy) (+5 more)

### Community 6 - "Metrics Visualization & Plotting"
Cohesion: 0.22
Nodes (15): DataFrame, ndarray, Series, _convergence_speed(), evaluate_methods(), extract_dynamic_metrics(), format_plot_label(), get_available_csv_files() (+7 more)

### Community 7 - "Dual-Track Variance Simulation"
Cohesion: 0.18
Nodes (8): AdvancedDualTrackVarianceSimulator, configure_simulation(), plot_simulation(), Generates diagnostic plots mapping variance thresholds and halting dynamics., Interactive prompt to define standard federated learning environment parameters,, Calculates effective variance incorporating heavy-tails, stragglers, and shocks., Simulates intra-round variance and early-stopping dynamics in a dual-track, Executes the federated training simulation across all rounds.

### Community 8 - "Multi-Seed Benchmarking & Grid Search"
Cohesion: 0.39
Nodes (3): AdvancedDualTrackVarianceSimulator, evaluate_multi_seed(), run_grid_search()

### Community 9 - "Client Sampler Test Suite"
Cohesion: 0.38
Nodes (4): ClientSampler, A class that helps random sampling of clients in each round.      Attributes:, Returns the next group of sampled client IDs.          Returns:             l, A generator function to continuously generate groups of sampled client IDs.

### Community 12 - "Hyperparameter Generation & Config"
Cohesion: 0.47
Nodes (5): create_hyperparameters(), get_next_filename(), get_opt_args(), Get the next filename for the hyperparameters.     Returns:         str: The n, Get the optional arguments for the optimizer.     Args:         opt_args (list

## Knowledge Gaps
- **6 isolated node(s):** `Fourier Domain Adaptation (FDA)`, `Hugging Face Datasets`, `Hugging Face Evaluate`, `PEFT (Legacy)`, `PEFT (Server)` (+1 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **7 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `prepare_federated_datasets()` connect `Federated Dataset Unit Tests` to `Federated Training & Client Sampling`?**
  _High betweenness centrality (0.105) - this node is a cross-community bridge._
- **Why does `AmsSketch` connect `AMS Sketch & Training Utilities` to `Federated Training & Client Sampling`?**
  _High betweenness centrality (0.067) - this node is a cross-community bridge._
- **Why does `ClientSampler` connect `Federated Training & Client Sampling` to `Federated Dataset Unit Tests`, `Client Sampler Generator`?**
  _High betweenness centrality (0.064) - this node is a cross-community bridge._
- **What connects `Calculate the prior distribution of labels in the training dataset.      This`, `Assign a sample to a client based on the sample's label and the remaining counts`, `Create federated datasets using Dirichlet-distributed label partitions.      T` to the rest of the system?**
  _79 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `Federated Training & Client Sampling` be split into smaller, more focused modules?**
  _Cohesion score 0.12896174863387977 - nodes in this community are weakly interconnected._
- **Should `Federated Dataset Unit Tests` be split into smaller, more focused modules?**
  _Cohesion score 0.09788359788359788 - nodes in this community are weakly interconnected._
- **Should `Federated Data Preparation Pipeline` be split into smaller, more focused modules?**
  _Cohesion score 0.09788359788359788 - nodes in this community are weakly interconnected._