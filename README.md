# ViZDoom visual PPO demo

Milestone 1: a reproducible visual-only PPO baseline for ViZDoom Basic. There is intentionally no Transformer, memory, or surprise gate yet.

## Run the supplied trained demo now

This checkout includes a project-local Python runtime and trained weights. From PowerShell:

```powershell
.\.python\python.exe evaluate.py --model artifacts\demo\best\best_model.zip --episodes 5 --render
```

Visible demos are deliberately paced at Doom's native 35-tic rate. Without this pacing, synchronous ViZDoom completes successful episodes almost instantly and appears to jump directly to the `FINISHED` screen.

For a non-visual 20-episode test, omit `--render` and use `--episodes 20`.

The selected checkpoint was trained for 20,000 environment decisions. On a separate fixed-seed 20-episode test (seeds 2000-2019), it achieved mean `0.10`, standard deviation `1.55`, and median `0.95` scaled reward; 16/20 episodes had positive reward. Four episodes timed out, so this is a working demonstration rather than a solved-policy claim.

## Model contract

- Observation: grayscale frame, `84 x 84 x 1`, uint8; no privileged variables.
- Actions: `MOVE_LEFT`, `MOVE_RIGHT`, `ATTACK` (0, 1, 2).
- Reward: ViZDoom `basic.cfg` reward multiplied by `0.01` for PPO stability. The exact unscaled value is retained as `info["raw_reward"]`; scaling does not change the optimal policy.
- Policy: Stable-Baselines3 PPO `CnnPolicy`, NatureCNN, 256 visual features.
- Runtime: one PyTorch CPU thread by default; this avoids severe oversubscription for the small CNN.

## Windows setup

Install 64-bit Python 3.11 or 3.12, then run:

```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python smoke_test.py
```

## Train and TensorBoard

```powershell
python train.py --timesteps 100000 --seed 0 --output artifacts\demo
tensorboard --logdir artifacts\tensorboard
```

Checkpoints are saved under `artifacts`. Resume the supplied run with:

```powershell
python train.py --timesteps 50000 --resume artifacts\demo\checkpoints\ppo_vizdoom_20000_steps.zip --output artifacts\continued
```

## Evaluate / live demo

```powershell
python evaluate.py --model artifacts\demo\best\best_model.zip --episodes 20
python evaluate.py --model artifacts\demo\best\best_model.zip --episodes 5 --render
```

Evaluation freezes the policy and uses deterministic actions by default. Report multiple episodes, not one lucky run.

### Windows note

ViZDoom's native Windows engine can hang while its child process shuts down. The entry-point scripts therefore flush/save all results and use immediate process termination after completion on Windows. This workaround is not used on Linux or macOS and does not alter training or inference.
