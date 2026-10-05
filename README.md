# ViZDoom pretrained agents

## Current Sample Factory experiment map

The active Deadly Corridor research now has canonical method names for
CNN+APPO, GRU+APPO, long-context GRU+APPO, GRU+optional-attention+APPO,
Transformer+APPO, GTrXL+APPO, and the evaluation-only Hugging Face GRU
reference. See [`SAMPLE_FACTORY_METHODS.md`](SAMPLE_FACTORY_METHODS.md) for the
architecture table, current results, and exact commands. New training commands
should use `sf_train_corridor.py --method=...`; the legacy `--memory` form is
kept for compatibility.

```bash
python sf_train_corridor.py --list-methods
```

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

### Preserved v1 result

`configs/deadly_corridor_baseline_v1.json` preserves the original native-reward
experiment exactly. Its artifacts remain under `artifacts/deadly_corridor_baseline/`
and its held-out results remain under `results/deadly_corridor_baseline/`. Do not
overwrite or resume those directories. The completed v1 experiment achieved
positive combat reward but 100% death and 0% corridor completion across 300
held-out episodes.

### Corrected v2 baseline

`configs/deadly_corridor_baseline_v2.json` is the new default. Native combat
reward is still logged and scaled by 0.01, but death adds -10 and genuine
corridor completion adds +10. A timeout is recorded as truncation and never
misclassified as completion. Entropy coefficient is increased from 0.01 to
0.02 to reduce the premature policy collapse observed in v1. V2 writes only to
`artifacts/deadly_corridor_baseline_v2/` and
`results/deadly_corridor_baseline_v2/`.

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

The default configuration is `configs/deadly_corridor_baseline_v2.json`. Seeds 0 and 1
start concurrently on `cuda:0` and `cuda:1`; seed 2 starts on the first free
GPU. Each seed has an independent directory and `training.log` under
`artifacts/deadly_corridor_baseline_v2/`. The default budget is one million
environment decisions per seed. TensorBoard data, checkpoints, evaluation
records, the best model, and the final model are kept separately for each seed.
The launcher also writes `launcher.log`, emits a heartbeat every minute, and on
failure prints the process exit signal plus the last 60 lines of that seed's
`training.log`. A Linux out-of-memory kill is therefore reported explicitly
rather than leaving the launcher apparently idle.

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

After copying the versioned test result to the analysis machine, inspect it with:

```bash
jupyter lab analyze_corridor.ipynb
```

The notebook reports bootstrap reward intervals, terminal outcomes, episode
length, and matched-test agreement between independently trained seeds. Reward
must not be interpreted as task success without checking death and completion.

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

## Matched Sample Factory memory ablation

The downloaded `deadly-corridor-upstream` checkpoint is not a memoryless PPO
baseline. It is a Sample Factory APPO policy with a 512-unit GRU,
`use_rnn=True`, and recurrence 32. It is preserved unchanged under
`artifacts/deadly-corridor-upstream/` and serves as the recurrent reference.

The controlled baseline in `sf_train_corridor.py` uses the same Sample Factory
environment, RGB 128x72 input, convolutional encoder, APPO hyperparameters,
8 workers x 4 environments, two policies, frame skip 4, and 10-million-step
budget. The CNN-only ablation changes only `use_rnn=False` and recurrence 1.
This makes the comparison meaningful; the earlier Stable-Baselines3 model is
not used as the no-memory control because its observations, rewards, network,
optimizer, and sampling system are different.

Use a separate Linux environment so the existing `.drl` environment and SB3
results stay intact:

```bash
python3 -m venv .sf
source .sf/bin/activate
python -m pip install --upgrade pip
pip install -r requirements-sample-factory.txt
```

Verify the two configurations without starting training:

```bash
python sf_train_corridor.py --memory cnn --check-config \
  --experiment check_cnn --train_dir artifacts/config_checks
python sf_train_corridor.py --memory gru --check-config \
  --experiment check_gru --train_dir artifacts/config_checks
```

Run a short end-to-end CNN-only smoke test first:

```bash
python run_sf_nomemory_multiseed.py --steps 4096
```

After the smoke test succeeds, remove or rename only those smoke-test output
directories, then run the full three-seed experiment:

```bash
python run_sf_nomemory_multiseed.py
```

Seeds 0 and 1 run concurrently on physical GPUs 0 and 1; seed 2 starts on the
first GPU that becomes free. Each run is named `sf_corridor_cnn_seed_N` under
`artifacts/sample_factory/`. The launcher writes both terminal output and
`artifacts/sample_factory/sf_corridor_cnn_runs/launcher.log`; each seed has its
own log. If a worker fails, the launcher prints its exit code or signal and the
last 60 log lines. Before starting any worker, it also verifies CUDA visibility,
the requested physical indices, and a real matrix multiplication on every GPU.

Evaluate the preserved GRU reference on 100 reproducible held-out seeds:

```bash
python sf_evaluate_corridor.py \
  --experiment deadly-corridor-upstream --train-dir artifacts \
  --policy-index 0 --checkpoint best --episodes 100 --seed-start 10000 \
  --device gpu --results-root results/sample_factory_gru_reference
```

Evaluate each trained CNN-only seed on exactly the same episode seeds by
changing only `--experiment` and the results directory, for example:

```bash
python sf_evaluate_corridor.py \
  --experiment sf_corridor_cnn_seed_0 --train-dir artifacts/sample_factory \
  --policy-index 0 --checkpoint best --episodes 100 --seed-start 10000 \
  --device gpu --results-root results/sample_factory_cnn_seed_0
```

Every evaluation creates a timestamped directory containing `config.json`,
`episodes.csv`, `report.json`, and `run.log`. It reports native scaled reward,
completion, death, timeout, and episode frames. Evaluation is deterministic,
the policy is frozen, and no test episode performs a training update.

### Transformer temporal-memory experiment

`sf_transformer_core.py` replaces only the GRU/identity temporal core. The
visual encoder, policy and value heads, APPO settings, rollout length, reward,
workers, frame skip, and training budget remain matched. Its default core is:

```text
CNN feature 512 -> Linear 256 -> 32-token memory
                -> 2 Transformer encoder layers, 4 heads, FFN 512
                -> Linear 512 -> unchanged policy/value heads
```

The token memory is reset at episode boundaries. During rollout inference it
stores the previous 32 projected visual features; during learning it follows
Sample Factory's packed episode sequences, so attention and gradients never
cross a death, completion, or reset boundary. The temporal core has a similar
parameter scale to the 512-unit GRU and uses zero dropout.

After pulling the latest code in the existing `.sf` environment, validate the
configuration without training:

```bash
python sf_train_corridor.py --memory transformer --check-config \
  --experiment check_transformer --train_dir artifacts/config_checks
```

Run a short three-seed GPU smoke test first:

```bash
python run_sf_transformer_multiseed.py --steps 4096
```

If all seeds exit with code zero, preserve the smoke output and start the full
matched experiment:

```bash
mv artifacts/sample_factory_transformer artifacts/sample_factory_transformer_smoke
python run_sf_transformer_multiseed.py
```

The launcher checks both GPUs before starting, runs seeds 0 and 1 concurrently,
starts seed 2 on the first available GPU, writes one log per seed plus a launcher
log, and prints the final 60 lines after any failure. Transformer attention will
usually make this experiment slower than the CNN and GRU runs.

Evaluate both policies from every trained seed on the same held-out seeds:

```bash
for seed in 0 1 2; do
  for policy in 0 1; do
    python sf_evaluate_corridor.py \
      --experiment sf_corridor_transformer_seed_${seed} \
      --train-dir artifacts/sample_factory_transformer \
      --policy-index ${policy} --checkpoint best \
      --episodes 100 --seed-start 10000 --device gpu \
      --results-root results/sample_factory_transformer_seed_${seed}_policy_${policy}
  done
done
```

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
