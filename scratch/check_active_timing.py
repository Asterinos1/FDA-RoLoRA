# Let's inspect the exact timing and round progression of each active run

# Active runs from status output:
# 1. [2b987091da] QNLI Non-IID C50 R30 FDA-Pure k=0.5 s43
#    Elapsed at status = 1439.9m
#    Status recorded: [13:22:05] Round 27/30 finished.
#    Rounds left: 3 rounds (28, 29, 30).
#    Average time per round = 1439.9m / 27 ≈ 53.3 minutes/round.
#    Or if last round was Epochs=1 (Matrix B, Ratio=0.5):
#    Notice: controller epoch was 1! Epochs: 1.
#    When epochs=1, a round is much faster (~10-15m), but average across rounds is ~50m.
#    At 53m/round, 3 rounds = ~160 mins (2.6 hours) from 13:22 -> ~16:00!
#    Wait! If Round 27 finished at 13:22, 3 rounds would complete by ~16:00.
#    Since it is now 19:07, [2b987091da] has ALMOST CERTAINLY FINISHED or is in its final evaluation!

# 2. [547a1903e9] MNLI Non-IID C20 R50 FDA-Pure k=1.0 s42
#    Elapsed at status = 848.8m
#    Status recorded: [14:11:26] Round 14/50 finished.
#    Average time per round = 848.8m / 14 ≈ 60.6 minutes/round.
#    Target rounds: 50.
#    Min rounds for early stopping (mr) = 30 rounds!
#    Gate ES0.6, MR30.
#    Rounds remaining to MR30 = 30 - 14 = 16 rounds.
#    16 rounds * ~60 mins = 960 minutes (~16 hours!).
#    Even to complete 3 rounds = ~3 hours.
#    Will it finish in the next 3 hours? NO, definitely not (needs at least 16 more rounds to MR30 early stopping, or 36 rounds to R50).

# 3. [19d39ebea4] MNLI Non-IID C20 R50 RoLoRA E10 s43
#    Elapsed at status = 827.6m
#    RoLoRA fixed E10 on MNLI C20 takes ~35-40 min per round.
#    827.6m / 35m ≈ ~23 rounds completed by ~14:00.
#    Total rounds = 50, Early stopping MR = 30.
#    To reach Round 30 from ~Round 23 = 7 rounds * 35m = ~245m (~4 hours).
#    To reach Round 50 = ~16 hours.
#    Could it finish in next 3 hours if it triggers early stopping right at round 30?
#    Let's check if it could hit round 30 in 3 hours.

# 4. [ee14faf072] MNLI Non-IID C20 R50 FDA-Pure k=1.0 s43
#    Elapsed at status = 562.5m
#    Status recorded: [13:51:13] Round 9/50 finished.
#    Average time per round = 562.5m / 9 ≈ 62.5 minutes/round.
#    Only 9 of 50 rounds done. MR is 30.
#    Needs 21 more rounds just to reach MR30 (~21 hours).
#    Will NOT finish in the next 3 hours.

# 5. [9f4fe5a89e] MNLI Non-IID C20 R50 FDA-Pure k=0.5 s43
#    Elapsed at status = 507.0m
#    Status recorded: [13:36:11] Round 11/50 finished.
#    Average time per round = 507.0m / 11 ≈ 46.1 minutes/round.
#    Only 11 of 50 rounds done. MR is 30.
#    Needs 19 more rounds just to reach MR30 (~14.5 hours).
#    Will NOT finish in the next 3 hours.

# 6. [ff4aea2a67] MNLI Non-IID C50 R30 RoLoRA E10 s42
#    Elapsed at status = 480.2m
#    MNLI C50 R30: Total rounds = 30. Early stopping MR = 18!
#    For RoLoRA C50, average round time is ~17-20 mins.
#    At 480.2m elapsed: 480.2 / ~18m ≈ ~26 rounds!
#    Wait! If it was already at ~24-26 rounds when status was checked, could it finish soon?
#    Let's check how long RoLoRA C50 R30 took in previous completed runs!
