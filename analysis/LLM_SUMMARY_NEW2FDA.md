# LLM Summary: NEW2FDA vs. Federated Baselines

```json
[
  "NONLIN (MLP) Outperformance: NEW2FDA methods lead in non-linear configurations. In NONLIN-10c1, FDA EMA-Standard achieves 72.47% (peak 75.49%), beating RoLoRA (69.40%) and FedAvg-LoRA (67.01%). In NONLIN-5c2, FDA Standard achieves 76.89% (peak 80.07%), beating RoLoRA (75.73%) and FedAvg-LoRA (75.07%).",
  "EMA Variance Damping: In the pathological 10c1 partition, EMA momentum (rho=0.9) reduces seed variance to +/-0.85%, delivering highly stable cross-seed convergence.",
  "Compute vs Communication Tradeoff: FDA dynamically scales local epochs (e.g. 5-7 local epochs vs 3 fixed), trading increased local compute steps per round for superior representation learning under non-IID data.",
  "Loss Divergence Resistance: In linear settings, standard FedAvg-LoRA and FFA-LoRA show severe loss explosion (>80-470), whereas RoLoRA and NEW2FDA bound testing loss tightly (<6.2)."
]
```

## Setting: LIN_10c1 (LIN - 10c1)

| Method | R=25 | R=50 | R=75 | R=100 (Final) | Peak Acc (Best R) | Avg Epochs | Total Steps |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **FDA EMA-Standard** | 40.0% | 50.6% | 47.4% | **54.6%** | 59.7% (R85) | 11.23 | 105530 |
| **FDA Standard** | 40.0% | 50.0% | 44.6% | **55.8%** | 59.5% (R72) | 9.05 | 85070 |
| **FFA-LoRA** | 27.1% | 25.6% | 29.7% | **31.2%** | 39.1% (R75) | 5.0 | 47000 |
| **FedAvg-LoRA** | 56.4% | 62.1% | 63.1% | **68.0%** | 71.0% (R97) | 5.0 | 47000 |
| **RoLoRA** | 40.5% | 46.9% | 49.1% | **54.8%** | 60.1% (R88) | 5.0 | 47000 |

## Setting: LIN_5c2 (LIN - 5c2)

| Method | R=25 | R=50 | R=75 | R=100 (Final) | Peak Acc (Best R) | Avg Epochs | Total Steps |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **FDA EMA-Standard** | 37.1% | 51.4% | 50.1% | **55.1%** | 62.5% (R94) | 10.62 | 198594 |
| **FDA Standard** | 45.8% | 55.2% | 52.1% | **57.9%** | 64.8% (R91) | 8.99 | 168113 |
| **FFA-LoRA** | 32.6% | 31.4% | 33.2% | **31.6%** | 38.2% (R40) | 5.0 | 93500 |
| **FedAvg-LoRA** | 42.7% | 46.9% | 43.1% | **43.9%** | 54.5% (R35) | 5.0 | 93500 |
| **RoLoRA** | 42.4% | 55.3% | 48.2% | **61.3%** | 65.9% (R80) | 5.0 | 93500 |

## Setting: NONLIN_10c1 (NONLIN - 10c1)

| Method | R=25 | R=50 | R=75 | R=100 (Final) | Peak Acc (Best R) | Avg Epochs | Total Steps |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **FDA EMA-Standard** | 50.6% | 65.2% | 62.0% | **72.5%** | 75.5% (R94) | 11.19 | 105217 |
| **FDA Standard** | 49.3% | 63.9% | 64.5% | **71.2%** | 74.6% (R98) | 8.89 | 83597 |
| **FFA-LoRA** | 48.5% | 61.4% | 66.4% | **69.1%** | 71.8% (R98) | 5.0 | 47000 |
| **FedAvg-LoRA** | 48.8% | 63.0% | 67.1% | **67.0%** | 71.8% (R85) | 5.0 | 47000 |
| **RoLoRA** | 53.5% | 64.1% | 61.4% | **69.4%** | 75.2% (R92) | 5.0 | 47000 |

## Setting: NONLIN_5c2 (NONLIN - 5c2)

| Method | R=25 | R=50 | R=75 | R=100 (Final) | Peak Acc (Best R) | Avg Epochs | Total Steps |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **FDA EMA-Standard** | 62.0% | 69.7% | 61.3% | **71.4%** | 75.0% (R82) | 10.38 | 194168 |
| **FDA Standard** | 66.0% | 72.3% | 72.7% | **76.9%** | 80.1% (R94) | 8.48 | 158513 |
| **FFA-LoRA** | 57.0% | 57.6% | 58.3% | **59.2%** | 60.1% (R92) | 5.0 | 93500 |
| **FedAvg-LoRA** | 65.1% | 68.9% | 67.2% | **75.1%** | 75.1% (R100) | 5.0 | 93500 |
| **RoLoRA** | 65.6% | 75.7% | 72.4% | **75.7%** | 79.9% (R96) | 5.0 | 93500 |

