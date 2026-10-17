---
type: "query"
date: "2026-09-22T09:10:57.804333+00:00"
question: "how our FDA dual track standard method works"
contributor: "graphify"
outcome: "useful"
source_nodes: ["training_fda_opt_fda_rolora_dual_track_opt", "training_fda_opt_fda_rolora_dual_track_ema_standard_opt", "models_ops_variance"]
---

# Q: how our FDA dual track standard method works

## Answer

FDA Dual-Track Standard method alternates optimization between LoRA Matrix A and Matrix B with independent threshold tracks theta_A and theta_B. Local client training continues while variance <= theta_active up to E_max. At round end, next theta is updated proportional to target epoch ratio ((E_max/2)/round_epochs)*variance.

## Outcome

- Signal: useful

## Source Nodes

- training_fda_opt_fda_rolora_dual_track_opt
- training_fda_opt_fda_rolora_dual_track_ema_standard_opt
- models_ops_variance