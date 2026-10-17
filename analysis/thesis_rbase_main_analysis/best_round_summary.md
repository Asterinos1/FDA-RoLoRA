# Peak and Final Test Accuracy Summary

## Scenario: `RTE-C20-R30-RK4`

| Method                 | Peak Accuracy   | Peak Round     | Final Accuracy   | Actual Rounds   |   Min Test Loss |
|:-----------------------|:----------------|:---------------|:-----------------|:----------------|----------------:|
| FedAvg-LoRA            | 59.21% ± 1.08%  | R30.0 [30, 30] | 59.21% ± 1.08%   | 30.0 [30, 30]   |        0.68847  |
| FedAvg-RoLoRA          | 54.63% ± 1.82%  | R14.0 [1, 29]  | 53.19% ± 1.63%   | 30.0 [30, 30]   |        0.691773 |
| FDA-RoLoRA (DualTrack) | 54.39% ± 2.61%  | R10.7 [1, 20]  | 52.71% ± 1.44%   | 30.0 [30, 30]   |        0.691769 |

## Scenario: `RTE-C50-R12-RK4`

| Method                 | Peak Accuracy   | Peak Round   | Final Accuracy   | Actual Rounds   |   Min Test Loss |
|:-----------------------|:----------------|:-------------|:-----------------|:----------------|----------------:|
| FedAvg-RoLoRA          | 54.51% ± 3.13%  | R7.0 [1, 10] | 52.59% ± 0.21%   | 12.0 [12, 12]   |        0.691793 |
| FedAvg-LoRA            | 53.91% ± 2.08%  | R5.3 [1, 10] | 52.47% ± 0.42%   | 12.0 [12, 12]   |        0.691767 |
| FDA-RoLoRA (DualTrack) | 50.90% ± 3.13%  | R1.0 [1, 1]  | 50.90% ± 3.13%   | 12.0 [12, 12]   |        0.692922 |

## Scenario: `RTE-C50-R12-RK8`

| Method                 | Peak Accuracy   | Peak Round   | Final Accuracy   | Actual Rounds   |   Min Test Loss |
|:-----------------------|:----------------|:-------------|:-----------------|:----------------|----------------:|
| FedAvg-LoRA            | 52.95% ± 0.42%  | R3.3 [1, 5]  | 52.71% ± 0.00%   | 12.0 [12, 12]   |        0.691824 |
| FedAvg-RoLoRA          | 52.95% ± 0.42%  | R6.7 [1, 11] | 52.71% ± 0.00%   | 12.0 [12, 12]   |        0.691776 |
| FDA-RoLoRA (DualTrack) | 50.90% ± 3.13%  | R1.0 [1, 1]  | 50.78% ± 3.33%   | 12.0 [12, 12]   |        0.692311 |

## Scenario: `SST2-C20-R75-RK4`

| Method                 | Peak Accuracy   | Peak Round     | Final Accuracy   | Actual Rounds   |   Min Test Loss |
|:-----------------------|:----------------|:---------------|:-----------------|:----------------|----------------:|
| FDA-RoLoRA (DualTrack) | 92.43% ± 0.20%  | R16.0 [14, 18] | 92.39% ± 0.24%   | 16.7 [14, 20]   |        0.222674 |

## Scenario: `SST2-C50-R30-RK4`

| Method                 | Peak Accuracy   | Peak Round     | Final Accuracy   | Actual Rounds   |   Min Test Loss |
|:-----------------------|:----------------|:---------------|:-----------------|:----------------|----------------:|
| FDA-RoLoRA (DualTrack) | 91.55% ± 0.76%  | R18.0 [14, 21] | 91.02% ± 0.35%   | 20.0 [16, 22]   |        0.227471 |

## Scenario: `SST2-C50-R30-RK8`

| Method                 | Peak Accuracy   | Peak Round     | Final Accuracy   | Actual Rounds   |   Min Test Loss |
|:-----------------------|:----------------|:---------------|:-----------------|:----------------|----------------:|
| FDA-RoLoRA (DualTrack) | 91.17% ± 0.11%  | R15.0 [13, 17] | 91.09% ± 0.13%   | 17.0 [15, 20]   |        0.230862 |

