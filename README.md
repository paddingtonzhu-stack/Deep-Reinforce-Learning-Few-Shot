# ViZDoom pretrained agents

## Deadly Corridor — recommended demo

This project now includes the pretrained Sample Factory APPO Deadly Corridor
agent. It fights enemies, moves through the corridor, collects supplies, and is
much more suitable for a presentation than the small Basic scenario below.

On Windows, double-click `demo_deadly_corridor.cmd`, or run:

```powershell
.\demo_deadly_corridor.cmd
```

This displays five episodes at 20 FPS. To benchmark twenty episodes without
rendering:

```powershell
.\evaluate_deadly_corridor.cmd
```

The launcher selects the upstream `best` checkpoint, policy 0, and deterministic
evaluation. The model snapshot and isolated Windows runtime are ignored by Git
because they are downloaded/generated dependencies rather than source files.

For Linux training or fine-tuning, install Sample Factory and download the same
published checkpoint:

```bash
pip install "sample-factory[vizdoom]==2.1.1"
python -m sample_factory.huggingface.load_from_hub \
  -r MattStammers/vizdoom_deadly_corridor -d ./artifacts
python -m sf_examples.vizdoom.enjoy_vizdoom \
  --algo=APPO --env=doom_deadly_corridor --train_dir=./artifacts \
  --experiment=vizdoom_deadly_corridor --load_checkpoint_kind=best
```

The remaining sections document the earlier compact PPO baseline.

Milestone 1: a reproducible visual-only PPO baseline for ViZDoom Basic. There is intentionally no Transformer, memory, or surprise gate yet.

### OOD test suite

Run the frozen PPO baseline against the in-distribution control and controlled
observation/dynamics shifts:

```bash
python test.py
```

The suite covers missing frames, darker observations, sensor noise, a changed
frame skip, and a combined visual shift. Every invocation creates a new UTC
timestamped directory under `results/ood/`; previous runs are never overwritten.
Each run saves its resolved `config.json`, full `report.json`, and
`episodes.csv`. For a quick pipeline check, use `python test.py --episodes 2`.

## Run the supplied trained demo now

This checkout includes a project-local Python runtime and trained weights. From PowerShell:

```powershell
.\.python\python.exe evaluate.py --model artifacts\demo\best\best_model.zip --episodes 5 --render
```

Visible demos are deliberately slowed to `0.75` seconds per Doom tick. Since one policy decision repeats for four ticks, even the fastest two-decision win takes about six seconds and its movement/shooting remains visible. The terminal also prints every chosen action and reward. Change the speed with `--render-delay`, for example `--render-delay 0.4` for a faster demo.

For a non-visual 20-episode test, omit `--render` and use `--episodes 20`.

The selected checkpoint was trained for 30,000 environment decisions. In a controlled comparison over 50 unseen seeds (5000-5049), it was the best saved checkpoint: 37/50 positive-reward episodes, mean `-0.14`, standard deviation `1.76`, and median `0.95` scaled reward. Failures incur a large timeout penalty, which explains why the mean is below zero despite a 74% positive-result rate. This is a working demonstration rather than a solved-policy claim.

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
python train.py --timesteps 50000 --resume artifacts\demo\checkpoints\ppo_vizdoom_30000_steps.zip --output artifacts\continued
```

## Evaluate / live demo

```powershell
python evaluate.py --model artifacts\demo\best\best_model.zip --episodes 20
python evaluate.py --model artifacts\demo\best\best_model.zip --episodes 5 --render
```

Evaluation freezes the policy and uses deterministic actions by default. Report multiple episodes, not one lucky run.

### Windows note

ViZDoom's native Windows engine can hang while its child process shuts down. The entry-point scripts therefore flush/save all results and use immediate process termination after completion on Windows. This workaround is not used on Linux or macOS and does not alter training or inference.
