# Computation Efficiency Summary

## LIN - 10c1

| Method           |   Cumulative local steps per client |   Local steps per client to peak |   Avg Epochs / Round |
|:-----------------|------------------------------------:|---------------------------------:|---------------------:|
| FDA EMA-Standard |                              105531 |                          89582   |              11.2267 |
| FDA Standard     |                               85070 |                          68933.3 |               9.05   |
| FFA-LoRA         |                               47000 |                          23970   |               5      |
| LoRA             |                               47000 |                          43083.3 |               5      |
| RoLoRA           |                               47000 |                          41360   |               5      |

## LIN - 5c2

| Method           |   Cumulative local steps per client |   Local steps per client to peak |   Avg Epochs / Round |
|:-----------------|------------------------------------:|---------------------------------:|---------------------:|
| FDA EMA-Standard |                              198594 |                         161568   |                10.62 |
| FDA Standard     |                              168113 |                         151470   |                 8.99 |
| FFA-LoRA         |                               93500 |                          45815   |                 5    |
| LoRA             |                               93500 |                          42386.7 |                 5    |
| RoLoRA           |                               93500 |                          70436.7 |                 5    |

## NONLIN - 10c1

| Method           |   Cumulative local steps per client |   Local steps per client to peak |   Avg Epochs / Round |
|:-----------------|------------------------------------:|---------------------------------:|---------------------:|
| FDA EMA-Standard |                            105217   |                         100173   |             11.1933  |
| FDA SMA          |                             94689.3 |                          84976   |             10.0733  |
| FDA Standard     |                             83597.3 |                          79242   |              8.89333 |
| FFA-LoRA         |                             47000   |                          45433.3 |              5       |
| LoRA             |                             47000   |                          40263.3 |              5       |
| RoLoRA           |                             47000   |                          43866.7 |              5       |

## NONLIN - 5c2

| Method           |   Cumulative local steps per client |   Local steps per client to peak |   Avg Epochs / Round |
|:-----------------|------------------------------------:|---------------------------------:|---------------------:|
| FDA EMA-Standard |                              194168 |                         150722   |             10.3833  |
| FDA SMA          |                              169671 |                         132334   |              9.07333 |
| FDA Standard     |                              158514 |                         148166   |              8.47667 |
| FFA-LoRA         |                               93500 |                          86955   |              5       |
| LoRA             |                               93500 |                          90695   |              5       |
| RoLoRA           |                               93500 |                          87266.7 |              5       |

