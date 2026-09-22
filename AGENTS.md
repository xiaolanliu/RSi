# Instructions for an agent starting without prior context

Read `README.md`, `docs/USAGE.md`, `docs/ARCHITECTURE.md`, then `docs/REPRODUCE.md`.
For the integrated agent loop, also read `docs/AGENT_LOOP.md`. Run it in the
`rsi` Python 3.11 conda environment. All worker processes use `sys.executable`.
No GPT calls are authorized during integration verification. Default to mock;
never read Codex/IDE credentials. Only the dedicated `RSI_SIM_OPENAI_API_KEY`
may be used when the user subsequently requests live evaluation.
Work from the repository root unless explicitly passing absolute paths. Do not assume the original author's filesystem exists.

## Current release

- Production bundle: `models/v11/fold_all.pt`, `risk_kind=three_signal_v11`, `stage_kind=ordered_duration_v7`.
- `models/encoder_2000/` contains the online encoder, while `models/offline_2000/` contains the distinct offline teacher. Both are original epoch-2000 checkpoints.
- Core online API: `agent_closed_loop.monitor.OnlineMonitor`. Portable entry points: `rsi_tools/replay.py` and `rsi_tools/verify.py`.
- Runtime uses only the V11 bundle, installed code, NumPy/PyTorch, and caller-provided causal visual features. Original COMPILE/SIEVE folders and `runs/` are not runtime dependencies.
- Required research/training modules remain under `agent_closed_loop/` and `compile/`. Obsolete report builders and historical-data tests were removed; recover them from commit `51b1da3` if researching old experiments.

## Required semantics

1. Observations are consecutive at 30 FPS. Reset for each episode. Do not subsample state while retaining fixed frame windows.
2. Twelve joint values are radians; two gripper widths are meters. `left7_right7` means L6,gL,R6,gR. `joints12_grippers2` means L6,R6,gL,gR. Pass the layout explicitly at hardware boundaries.
3. `step` accepts raw state/visual. `step_normalized` accepts already normalized 76-D features. Never mix the two within an episode or normalize twice.
4. Visual48 must use the released preprocessing and matching Wan VAE. Whole-video temporal interpolation, future-frame windows, arbitrary visual embeddings and zero visual placeholders are not equivalent inputs.
5. Offline forward action deltas and online observed backward deltas are different. The deployed pi0.5 postprocessed command is an absolute target; never add the state to it again without checking the producer's transforms.
6. Keep the three final OOD signals and normal-only fit/calibration split. A new weight/window changes the score distribution and requires recalibration. Do not tune thresholds using failures.
7. Treat `confirmed_phase` as the monotone stage memory; `accepted_phase=0` means suspended by OOD. Probability argmax is not the execution state. `rsi_loop` owns simulation control; the head itself emits evidence only.
8. Do not claim action-conditioned dynamics are deployed: command diagnostics are not part of the released alarm.

## Before and after model-affecting changes

```bash
rsi-verify --hashes-only
python tools/run_unit_tests.py
rsi-verify --device cpu --output outputs/verification_cpu.json
# If CUDA is available:
rsi-verify --device cuda:0 --output outputs/verification_cuda.json
```

All 9 supplied episodes must match the independent historical outputs; alarms, confirmed phases and offline boundaries must match exactly, float arrays use documented tolerances. Test changed training/preparation code with a small isolated fixture; a smoke test is not a full 2000-epoch retraining claim.

The offline fixture includes independently computed original-source CPU probabilities in addition to the original CUDA TF32 report. `check(..., device=...)` selects the appropriate reference. This documented backend difference does not affect the online reference or permit weakening online alarm checks.

Preserve release weights and golden outputs. Save new training runs to a new ignored output directory. If intentionally releasing a new model, document changes, establish new independent evaluation, update artifacts and expected outputs deliberately. Do not silently rewrite expected outputs to make a failing test pass.

Environment: Python 3.10, torch 2.8.0, numpy 1.26.4. CPU is enough for replay. See `docs/ENVIRONMENT.md` for installation, GPU training and optional VAE requirements. No Git LFS is required for these checkpoints.

The integrated environment uses Python 3.11, torch 2.7.0 and numpy 1.26.0 to
match IsaacSim 5.1. Re-run golden parity when changing these versions.
Preserve native RoboDojo physics/control rate and success checks. A 25→30 Hz
causal clock adapter does not establish simulation OOD calibration. Mock holds
test ownership transfer, not recovery skill. Only complete native VLA-only
successes can be promoted automatically to demonstration videos.
