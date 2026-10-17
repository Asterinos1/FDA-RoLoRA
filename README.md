<div align="center">

# FDA-RoLoRA: Federated Dynamic Averaging for Robust Low-Rank Adaptation

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE) [![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)](https://www.python.org/) [![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/) [![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Transformers%20%7C%20PEFT-ffcc00)](https://huggingface.co/) [![Topic](https://img.shields.io/badge/Domain-Federated%20Learning%20%7C%20LoRA%20%7C%20Fine%20Tuning-8A2BE2)](#abstract)

<p align="center">
  <a href="#abstract"><b>Abstract</b></a> •
  <a href="#acknowledgments--prior-work"><b>Acknowledgments</b></a> •
  <a href="#releases"><b>Releases</b></a> •
  <a href="#repository-structure"><b>Repo Structure</b></a> •
  <a href="#environment-setup"><b>Environment Setup</b></a> •
  <a href="#citation"><b>Citation</b></a> •
  <a href="#license"><b>License</b></a>
</p>

</div>

This repository contains the official codebase and experimental evaluation for my diploma thesis at the School of ECE, Technical University of Crete: **"Investigating the applicability of Federated Dynamic Averaging in Federated Low-Rank Adaptation Architectures"**.

---

## Abstract

Federated Learning (FL) has established itself as a foundational paradigm for privacy-preserving collaborative model training across decentralized data sources. However, scaling modern foundation models to billions of parameters exposes a critical dual bottleneck in federated optimization: an excessive *spatial* communication overhead from transmitting high-dimensional parameter updates, and an inefficient *temporal* communication pattern imposed by rigid, fixed-interval synchronization schedules. Parameter-efficient fine-tuning methods such as Low-Rank Adaptation (LoRA) address the spatial dimension by decomposing updates into compact low-rank factors. Yet, naively applying LoRA in federated environments introduces a fundamental structural inconsistency: independent averaging of constituent low-rank matrices fails to reconstruct the true global update due to the non-commutativity of matrix multiplication. Furthermore, existing federated LoRA variants remain temporally inefficient by relying on static synchronization frequencies agnostic to local training dynamics.

To address both dimensions simultaneously, this thesis presents the **FDA-RoLoRA Dual Track** framework, a principled integration of Federated Dynamic Averaging (FDA) within a Robust Low-Rank Adaptation (RoLoRA) protocol. The proposed architecture enforces coordinate-space invariance through isolated alternating aggregation of down-projection ($\mathbf{A}$) and up-projection ($\mathbf{B}$) matrices, eliminating geometric cross-contamination while maintaining strict aggregation linearity. Concurrently, the framework introduces an adaptive, variance-driven synchronization mechanism governed by independent per-matrix thresholds ($\Theta_A, \Theta_B$), combined with a calibrated setpoint ratio ($\kappa = 1.0$) and Matrix $\mathbf{B}$ variance discounting ($\gamma = 3.0$) to counteract low-rank curvature penalties and premature braking. The framework is extensively evaluated on standard GLUE benchmarks using RoBERTa-base architectures under non-IID data partitioning against FedAvg+LoRA, FedOpt+LoRA, and fixed-schedule RoLoRA baselines to quantify the trade-offs between communication efficiency, convergence stability, and task performance.

---

## Acknowledgments & Prior Work

**FDA-RoLoRA** represents the principled combination and extension of two brilliant works in federated optimization and parameter-efficient fine-tuning:
- **[RoLoRA](https://arxiv.org/abs/2502.01755)** (*Sun et al., 2025*): Resolves low-rank aggregation inconsistency by alternating updates between down-projection ($\mathbf{A}$) and up-projection ($\mathbf{B}$) matrices.
- **[FDA-Opt](https://arxiv.org/abs/2505.04535)** (*Theologitis et al., 2025*): Replaces fixed synchronization rounds with adaptive, variance-driven communication thresholds.

This repository directly expands upon the open-source implementation from the [FDA-Opt repository](https://github.com/michaeltheologitis/FDA-Opt) by [Michail Theologitis](https://github.com/michaeltheologitis).

---

## Releases

* **`v1.0.0` — Thesis & Experimental Benchmark Release (Snapshot Template)**
  * Official release snapshot capturing the exact state of the codebase used to execute all experimental evaluations and GLUE benchmark suites reported in the diploma thesis.
  * Contains the complete FDA-RoLoRA Dual-Track framework (calibrated $\kappa$ controller, Matrix $\mathbf{B}$ variance discounting), all baseline architectures (FedAvg-LoRA, FedOpt-LoRA, RoLoRA, FFA-LoRA), cluster orchestration scripts, and raw per-round telemetry logs.

---

## Repository Structure

```text
FDA-LoRA/
├── fdaopt/                                # Core algorithmic framework
│   ├── datasets/fed_data_prep.py          # Federated non-IID partitioning & client samplers
│   ├── metrics/mentrics_handler.py        # Per-round telemetry logging & metric tracking
│   ├── models/                            # Model wrappers (PEFT LoRA, toy models, tensor ops)
│   └── training/                          # Federated trainers (FedAvg, RoLoRA, FDA-Opt variants)
├── fixed_calibration/                     # Authoritative calibrated dual-track controller
│   └── fda_rolora_dual_track_calibrated.py# Calibrated trainer with configurable ratio (kappa) & B-discounting
├── threshold_controller_experiments/      # Standalone HPC & cluster orchestration module
│   ├── run_threshold_experiments.py       # Staged execution runner (RoLoRA -> FDA k=0.5 -> FDA k=1.0)
│   ├── run_threshold_experiments.sh       # Multi-GPU asynchronous worker script
│   └── README.md                          # Dedicated execution & cluster guide
├── figures/                               # Publication-ready plots, loss curves & threshold telemetry
├── results/round_metrics/                 # Complete raw CSV per-round telemetry logs across all runs
├── sources/                               # Reference papers & technical notes (FDA-Opt, RoLoRA, FFA-LoRA)
├── FDA_vs_Baselines_v2.md                 # Master head-to-head evaluation scorecard & mathematical analysis
├── FDA_vs_Baselines_Evaluation_Assessment.md # Consolidated multi-task experimental evaluation
├── key_contrast_summary.txt               # Plain-text paradigm contrast summary (FDA-Opt vs RoLoRA vs FDA)
├── thesis_experiments.py                  # Master experimental configuration & dispatch engine
├── export_individual_plots.py             # High-resolution standalone and matrix figure exporter
├── requirements.txt                       # Local environment dependencies
└── requirements_server.txt                # Cluster / HPC dependencies (PyTorch, Transformers, PEFT)
```

---

## Environment Setup

To install the required dependencies and set up the environment, ensure you are using Python 3.10+ and run:

```bash
# Clone the repository and navigate into it
git clone https://github.com/Asterinos1/FDA-RoLoRA.git
cd FDA-RoLoRA

# Install core dependencies for local execution
pip install -r requirements.txt

# (Optional) For cluster execution (ARIS / Polytechnix)
pip install -r requirements_server.txt
```

---

## Citation

If you find this work or codebase helpful in your research, please cite:

```bibtex
@mastersthesis{karalis2026fdarolora,
  title  = {Investigating the applicability of Federated Dynamic Averaging in Federated Low-Rank Adaptation Architectures},
  author = {Karalis, Asterinos},
  school = {School of Electrical and Computer Engineering, Technical University of Crete},
  year   = {2026},
  type   = {Diploma Thesis},
  url    = {https://github.com/Asterinos1/FDA-RoLoRA}
}
```

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
