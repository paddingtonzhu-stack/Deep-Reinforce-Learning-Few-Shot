# ViZDoom Temporal-Memory Experiment Handoff

> Historical handoff from 2026-10-01. It records the state at that time and is
> not the current experiment guide. Use `SAMPLE_FACTORY_METHODS.md` for the
> current method taxonomy, results, and commands.

Updated: 2026-10-01 (Europe/Berlin)

## Repository

- GitHub: `git@github.com:paddingtonzhu-stack/Deep-Reinforce-Learning-Few-Shot.git`
- Branch: `main`
- Latest pushed commit: `7181f22` (`Preserve multiseed logs across resumes`)
- Linux checkout: `~/research/drl/Deep-Reinforce-Learning-Few-Shot`
- Linux virtual environment: `.sf`

## Research question

Does temporal memory improve generalization and task completion in ViZDoom
`doom_deadly_corridor` when the visual encoder, APPO settings, training budget,
and held-out evaluation seeds are controlled?

The three compared architectures are:

1. CNN-only Sample Factory APPO control.
2. CNN + GRU pretrained Sample Factory reference.
3. CNN + Transformer temporal-memory experiment.

## Shared visual encoder and training setup

Input is RGB `3x72x128`, with no health, coordinates, automap, labels, or other
privileged variables.

```text
Conv2D 3->32, kernel 8, stride 4, ELU
Conv2D 32->64, kernel 4, stride 2, ELU
Conv2D 64->128, kernel 3, stride 2, ELU
Flatten 2304
Linear 2304->512, ELU
```

Matched Sample Factory settings include APPO, frame skip 4, rollout/recurrence
32 for memory models, batch size 1024, one optimization epoch, 8 workers x 4
environments, two policies, and 10 million game frames per policy. A process
therefore reports approximately 20 million total frames for two policies.

## Completed results

All evaluations use deterministic frozen policies on seeds `10000..10099`.

### CNN-only

Six policies (3 training seeds x 2 policies), 100 episodes each:

```text
completion rates: 0%, 0%, 0%, 48%, 0%, 0%
aggregate: 48/600 = 8%
death: 552/600 = 92%
```

Only one of six CNN policies learned a partially successful strategy.

### GRU reference

Two pretrained policies, 100 episodes each:

```text
policy 0: 84% completion, 16% death, mean reward 19.476
policy 1: 68% completion, 32% death, mean reward 16.825
aggregate: 152/200 = 76%
```

Conclusion: memory is not strictly necessary because one CNN policy reached
48%, but GRU memory greatly improves average completion and training
reliability. More GRU training seeds would be needed for a strong formal
reliability claim.

## Pretrained GRU model

- Hugging Face: https://huggingface.co/MattStammers/vizdoom_deadly_corridor
- Local experiment name: `deadly-corridor-upstream`
- Download destination: `artifacts/deadly-corridor-upstream`

## Transformer implementation

Files:

- `sf_transformer_core.py`
- `sf_train_corridor.py`
- `run_sf_transformer_multiseed.py`
- `sf_evaluate_corridor.py`

Architecture:

```text
CNN feature 512
-> Linear 512->256
-> right-aligned memory of previous/current 32 feature tokens
-> 2 Transformer encoder layers
   - 4 attention heads
   - FFN dimension 512
   - GELU
   - dropout 0
-> Linear 256->512
-> unchanged policy and value heads
```

The Transformer is causal in operation because it processes tokens
sequentially and reads only the final current token. Its memory resets on every
episode boundary. Packed Sample Factory training sequences were tested with
nonzero gradients. Transformer temporal-core parameters: 1,325,824. GRU core
parameters: 1,575,936.

Completed tests:

- Python compilation.
- CNN, GRU, and Transformer configuration parsing.
- Online Transformer inference and state propagation.
- Packed-sequence forward/backward with nonzero gradients.
- End-to-end Sample Factory learner update and checkpoint save.
- Transformer checkpoint reconstruction and evaluation.
- Regression evaluation of the existing GRU checkpoint.

## Linux Transformer commands

```bash
cd ~/research/drl/Deep-Reinforce-Learning-Few-Shot
git pull
source .sf/bin/activate
python run_sf_transformer_multiseed.py
```

The launcher runs seeds 0 and 1 concurrently, then seed 2 on the first free
GPU. Logs are under:

```text
artifacts/sample_factory_transformer/sf_corridor_transformer_runs/
```

The launcher now appends timestamped resume markers instead of overwriting
per-seed logs.

## Critical current Linux state

An original Transformer run was interrupted at 11:45. An attempted resume
found its checkpoints but Sample Factory failed to load them after three
attempts and started from scratch in the same directories. Those processes
were stopped, and the interrupted artifact directory was archived under a name
like:

```text
artifacts/sample_factory_transformer_interrupted_20261001T...
```

A clean run was started at 21:34 through `nohup`:

```text
launcher PID: 241343
seed 0 parent: 241381
seed 1 parent: 241382
```

However, orphaned GPU workers from the failed 21:28 resume were discovered.
They must be terminated without touching the clean workers.

Old orphan PIDs to kill:

```text
239688 239754 239841 239842
239720 239790 240178 240179
```

Clean-run PIDs observed in `nvidia-smi` and not to kill:

```text
241343
241545 241627 241715 241716
242823 242857 242907 242909
```

Recommended immediate commands:

```bash
kill -TERM 239688 239754 239841 239842 239720 239790 240178 240179
sleep 5
ps -p 239688,239754,239841,239842,239720,239790,240178,240179 \
  -o pid,ppid,stat,cmd
```

If any old PIDs remain:

```bash
kill -KILL 239688 239754 239841 239842 239720 239790 240178 240179 2>/dev/null
```

Then verify the clean run:

```bash
nvidia-smi
ps -p 241343 -o pid,etime,%cpu,%mem,stat,cmd
tail -20 artifacts/sample_factory_transformer/sf_corridor_transformer_runs/launcher.log
```

The clean run uses `nohup`, so SSH can disconnect safely. Expected runtime is
roughly 60-70 minutes total. Transformer FPS was approximately 10,000 when not
contending with duplicate workers.

## Transformer evaluation after successful training

Wait for:

```text
All matched Transformer-memory Sample Factory runs completed successfully
```

Then evaluate all six policies:

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

Final comparison target:

```text
CNN-only:   8% aggregate completion
GRU:       76% aggregate completion
Transformer: pending
```

## Known issues

- Sample Factory 2.1.1 on Windows can attempt a duplicate checkpoint rename
  during shutdown of a tiny serial smoke test. Linux asynchronous training is
  the target path.
- Linux resume checkpoint loading failed for the interrupted Transformer run.
  Do not mix restarted models with old checkpoints; use the clean directory.
- `torch.jit.script` and Transformer nested-tensor messages are warnings, not
  training failures.
- `nvidia-smi` numbering can differ from PyTorch CUDA ordinal display. Trust
  the launcher's PyTorch CUDA preflight for the configured device mapping.
