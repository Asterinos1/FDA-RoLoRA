# Computation & Wall-Clock Efficiency Summary

## Scenario: `RTE-C20-R30-RK4`

| Method                 |   Total Local Steps |   Steps to Peak |   Avg Epochs/Rnd |   Total Time (min) |   Time to Peak (min) |   Avg Round Time (s) |
|:-----------------------|--------------------:|----------------:|-----------------:|-------------------:|---------------------:|---------------------:|
| FDA-RoLoRA (DualTrack) |             1533.33 |         413.333 |          12.7778 |            165.836 |              44.8642 |              331.673 |
| FedAvg-LoRA            |             2400    |        2400     |          20      |            249.456 |             249.456  |              498.913 |
| FedAvg-RoLoRA          |             2400    |        1120     |          20      |            248.306 |             116.269  |              496.612 |

## Scenario: `RTE-C50-R12-RK4`

| Method                 |   Total Local Steps |   Steps to Peak |   Avg Epochs/Rnd |   Total Time (min) |   Time to Peak (min) |   Avg Round Time (s) |
|:-----------------------|--------------------:|----------------:|-----------------:|-------------------:|---------------------:|---------------------:|
| FDA-RoLoRA (DualTrack) |             197.333 |           6     |          8.22222 |            42.7682 |              1.28888 |              213.841 |
| FedAvg-LoRA            |             480     |         213.333 |         20       |           103.235  |             45.9869  |              516.174 |
| FedAvg-RoLoRA          |             480     |         280     |         20       |           103.688  |             60.7249  |              518.441 |

## Scenario: `RTE-C50-R12-RK8`

| Method                 |   Total Local Steps |   Steps to Peak |   Avg Epochs/Rnd |   Total Time (min) |   Time to Peak (min) |   Avg Round Time (s) |
|:-----------------------|--------------------:|----------------:|-----------------:|-------------------:|---------------------:|---------------------:|
| FDA-RoLoRA (DualTrack) |                 206 |           6     |          8.58333 |            44.3064 |              1.30321 |              221.532 |
| FedAvg-LoRA            |                 480 |         133.333 |         20       |           100.991  |             27.7285  |              504.954 |
| FedAvg-RoLoRA          |                 480 |         266.667 |         20       |            99.385  |             55.2947  |              496.925 |

## Scenario: `SST2-C20-R75-RK4`

| Method                 |   Total Local Steps |   Steps to Peak |   Avg Epochs/Rnd |   Total Time (min) |   Time to Peak (min) |   Avg Round Time (s) |
|:-----------------------|--------------------:|----------------:|-----------------:|-------------------:|---------------------:|---------------------:|
| FDA-RoLoRA (DualTrack) |                8957 |            8427 |          10.0554 |            439.948 |              413.976 |              1571.21 |

## Scenario: `SST2-C50-R30-RK4`

| Method                 |   Total Local Steps |   Steps to Peak |   Avg Epochs/Rnd |   Total Time (min) |   Time to Peak (min) |   Avg Round Time (s) |
|:-----------------------|--------------------:|----------------:|-----------------:|-------------------:|---------------------:|---------------------:|
| FDA-RoLoRA (DualTrack) |             4370.67 |         3754.67 |          9.83712 |            548.712 |              471.208 |              1630.32 |

## Scenario: `SST2-C50-R30-RK8`

| Method                 |   Total Local Steps |   Steps to Peak |   Avg Epochs/Rnd |   Total Time (min) |   Time to Peak (min) |   Avg Round Time (s) |
|:-----------------------|--------------------:|----------------:|-----------------:|-------------------:|---------------------:|---------------------:|
| FDA-RoLoRA (DualTrack) |             3637.33 |         2977.33 |          9.71528 |            438.067 |              359.689 |              1546.99 |

