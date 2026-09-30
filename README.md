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
`episodes.csv`. Progress is shown in the terminal and simultaneously saved as
`run.log` in the same versioned directory. For a quick pipeline check, use
`python test.py --episodes 2`.

Device selection defaults to `auto`: CUDA is used when the installed PyTorch
build can access it, otherwise the test falls back to CPU. Select a GPU
explicitly when validating a multi-GPU Linux workstation:

```bash
python test.py --episodes 2 --device cuda:0
python test.py --episodes 2 --device cuda:1
```

Each report records CUDA availability, all detected GPU names, the requested
device, and the device actually used by the PPO model. Evaluation uses one GPU
per process; multi-GPU training will be configured separately.

Analyze all saved runs in Jupyter:

```bash
pip install -r requirements-analysis.txt
jupyter lab analyze_ood.ipynb
```

The notebook automatically selects the newest run with the largest episode
count, uses matched seeds for OOD-versus-control comparisons, and plots reward,
success rate, confidence intervals, and per-seed degradation.

## Deadly Corridor research baseline

The next experimental baseline uses the longer `deadly_corridor` scenario with
visual-only 84x84 grayscale observations. The policy receives no health value,
coordinates, labels, object locations, or automap. Its eight discrete actions
cover forward/backward movement, strafing, turning, attack, and forward attack.

First verify the environment and PPO checkpoint path:

```bash
python smoke_test_corridor.py
```

Before a long Linux run, verify every CUDA device independently:

```bash
python cuda_check.py
```

This performs a real matrix multiplication on each detected GPU and writes
`artifacts/cuda_check.json`. Both configured devices must report `pass`.

Run a short end-to-end training validation across the configured seeds/GPUs:

```bash
python run_multiseed.py --timesteps 4096
```

After that succeeds, start the full three-seed baseline:

```bash
python run_multiseed.py
```

The configuration is `configs/deadly_corridor_baseline.json`. Seeds 0 and 1
start concurrently on `cuda:0` and `cuda:1`; seed 2 starts on the first free
GPU. Each seed has an independent directory and `training.log` under
`artifacts/deadly_corridor_baseline/`. The default budget is one million
environment decisions per seed. TensorBoard data, checkpoints, evaluation
records, the best model, and the final model are kept separately for each seed.

Do not add Transformer memory until all three baseline runs finish and can be
evaluated with a shared seed set. This freezes the CNN/PPO control needed for a
fair ablation.

After all three training seeds finish, evaluate their best checkpoints on the
same 100 held-out episode seeds:

```bash
python test_corridor.py --device cuda:1
```

This performs 300 frozen-policy episodes: 100 test seeds for each of the three
independently trained models. Results are saved without overwriting prior runs
under `results/deadly_corridor_baseline/<UTC timestamp>/` as `config.json`,
`episodes.csv`, `report.json`, and `run.log`. The report includes per-model and
aggregate reward, episode length, death rate, and corridor-completion rate.

If a run is interrupted, resume from its latest periodic checkpoint while
specifying only the remaining number of decisions. For example, to continue a
seed-2 run from 300,000 to one million decisions:

```bash
python train.py --config configs/deadly_corridor_baseline.json --seed 2 \
  --device cuda:0 --output artifacts/deadly_corridor_baseline/seed_2 \
  --resume artifacts/deadly_corridor_baseline/seed_2/checkpoints/ppo_vizdoom_300000_steps.zip \
  --timesteps 700000
```

Resume mode preserves the checkpoint's existing timestep counter, so callback
checkpoint names and TensorBoard steps continue from the restored run.

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
