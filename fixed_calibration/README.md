# Fixed Calibration Experiment Suite

This directory contains the isolated, non-invasive calibration suite for **FDA-RoLoRA-DualTrack**, designed to test:
1. **Configurable Controller Ratios ($\kappa$):** Testing different denominator setpoints in $\theta_{\text{next}} = \left(\frac{\kappa \cdot E_{\max}}{E_{\text{actual}}}\right) \cdot \sigma^2$ (e.g. $\kappa = 1.0$ to target $E_{\max}$, $\kappa = 0.75$, and $\kappa = 0.5$ standard).
2. **Gated Matrix B Variance Discount:** Testing a variance discount factor on Matrix B (`enable_b_discount=True`, `b_discount_factor=3.0`), dormant (`False`) by default.
3. **Exact Production Early Stopping:** Incorporating the 6-round minimum, 3-round patience, 0.5% min-delta, and task-specific accuracy gates.
4. **Multi-GPU Parallel Execution:** Compatible with the production A10 GPU scheduler (`3` jobs/GPU, batch-64 capped at `2`/GPU).

---

## File Contents

| File | Description |
| :--- | :--- |
| `fda_rolora_dual_track_calibrated.py` | Contains the implementation of `fda_rolora_dual_track_calibrated_opt`. Ready to drop into `fdaopt/training/fda_opt.py`. |
| `thesis_calibration_experiments.py` | Master orchestrator script supporting planning, parallel GPU execution, active run registry, and process control. |
| `run_calibration_experiments.sh` | Production bash launcher for the Ubuntu GPU server with CLI command dispatch. |

---

## Deployment Instructions on the Ubuntu GPU Server

### Step 1: Add the New Method to Server's `fda_opt.py`
Append the function from `fixed_calibration/fda_rolora_dual_track_calibrated.py` to the end of `ARIS-FDA-LoRA/fdaopt/training/fda_opt.py`:
```bash
cat fixed_calibration/fda_rolora_dual_track_calibrated.py >> fdaopt/training/fda_opt.py
```
*(No existing lines or methods in `fda_opt.py` are modified or touched!)*

### Step 2: Make the Bash Script Executable
```bash
chmod +x fixed_calibration/run_calibration_experiments.sh
```

### Step 3: View the Execution Plan
```bash
# View the exact 96-run calibration plan across both seeds (42 and 43)
./run_calibration_experiments.sh plan
```

### Step 4: Run the Calibration Experiments
```bash
# Launch the 96-run calibration suite in background via nohup
nohup ./run_calibration_experiments.sh start > calib_suite.log 2>&1 &
```

### Step 5: Monitor Live Status & Process Control
```bash
# Check active runs, GPU assignment, elapsed time, and latest accuracy
./fixed_calibration/run_calibration_experiments.sh status

# Gracefully stop an active run by its short Control ID
./fixed_calibration/run_calibration_experiments.sh stop <CONTROL_ID>

# Permanently drop a run so it will not resume
./fixed_calibration/run_calibration_experiments.sh drop <CONTROL_ID>
```

---

## Distinct Run Identifier (`EXTRA`) Specification

Every calibration run generates an isolated, unambiguous identifier:
```text
THESIS-RBASE-CALIB-DS-{dataset}-C{clients}-R{rounds}-M-fda-calib-LR{lr}-RK4-E{emin}TO{emax}-CRATIO{ratio}-{NODISC|BDISC{factor}}-S{seed}-ES{gate}-MR6-P3-D0.005-W3
```
* **Metrics CSV:** Saved to `results/round_metrics/{EXTRA}_YYYYMMDD_HHMMSS.csv`
* **Worker Logs:** Saved to `results/logs/thesis_fixed_calibration/{EXTRA}.stdout.log`
* **Run Control:** Managed inside `results/run_control_fixed_calibration/`

This guarantees that **zero baseline results, logs, or metrics are overwritten**.
