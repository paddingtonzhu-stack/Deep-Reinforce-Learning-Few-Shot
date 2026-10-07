# Sample Factory Deadly Corridor methods

This is the canonical map of the Sample Factory experiments. Sample Factory
uses **APPO**, its asynchronous PPO implementation. The shorthand “+ PPO” in
discussion therefore means “+ Sample Factory APPO” in the code.

## Method catalog

| Method flag | Architecture | Role | Current evidence |
|---|---|---|---|
| `cnn` | CNN + APPO | No-memory control | 8% aggregate completion; one policy reached 48% |
| `gru` | CNN + 512-unit GRU + APPO, recurrence 32 | Standard trainable GRU | Architecture used by the upstream reference |
| `gru_long` | Same GRU + APPO, recurrence 64 | Current controlled improvement experiment | Pending GRU-32 vs GRU-64 study |
| `gru_residual_ln` | CNN + residual LayerNorm GRU + APPO, recurrence 64 | Cross-seed stability candidate | Next controlled training hypothesis after GRU-64 and action-vote ensemble failed upstream gates |
| `gru_completion_bonus` | CNN + GRU-64 + APPO with +10 successful-terminal reward | Completion-aligned training candidate | Next isolated hypothesis after residual LayerNorm GRU failed 0/3 seeds |
| `gru_skill_curriculum` | CNN + GRU-64 + APPO, pretraining at Doom skill 1 | Exploration-stability curriculum candidate | Next isolated hypothesis after low-LR continuation failed its multi-seed gate |
| `gru_orthogonal` | CNN + orthogonally initialized GRU-64 + APPO | Rejected initialization control | Failed the 2M gate: 1/3 paired wins, mean 15% vs 22%, worst 0% vs 7% |
| `gru_lag1` | CNN + GRU-64 + lag-controlled APPO | Stale-trajectory candidate | Caps asynchronous policy lag at one version while preserving the matched GRU-64 architecture and optimizer |
| `gru_sync` | CNN + GRU-64 + synchronous PPO | On-policy collection candidate | Sets `async_rl=False`; two 1024-sample batches consume each 2048-sample synchronous collection exactly once |
| `gru_explore` | CNN + GRU-64 + stronger-exploration APPO | Exploration-stability candidate | Raises only `exploration_loss_coeff` from 0.001 to 0.003 to test early policy collapse |
| `gru_state_refresh` | CNN + GRU + loss-preserving state-refresh APPO | Stale recurrent-state candidate | 128 learner steps with a current-policy hidden state detached at step 64; all samples retain policy/value loss |
| GRU-64 late-checkpoint average | Parameter average of the two late snapshots from each GRU-64 run | Optimization-noise robustness candidate | Next isolated hypothesis after completion bonus reduced matched 2M mean completion from 22% to 13% |
| GRU-64 ensemble | Per-action majority vote across three independently trained GRU-64 policies | Initialization-robust inference candidate | Motivated by low pairwise death-set overlap (Jaccard 0.35–0.36) after GRU-64 failed the upstream gate |
| `gru_attention` | CNN + GRU + gated optional attention + APPO | Our experimental architecture | 88.4% enabled vs 88.0% disabled on 500 matched episodes; no demonstrated attention benefit |
| `transformer` | CNN + causal finite-memory Transformer + APPO | Experimental architecture | Failed: 0.3% aggregate completion |
| `gtrxl` | CNN + gated Transformer-XL-style core + APPO | Experimental architecture | Failed to learn beyond approximately reward 5 |
| `upstream_gru` | Published CNN + GRU + APPO checkpoint | Evaluation-only Hugging Face reference | 84.4% on 500 episodes, seeds 20000–20499 |

The locally trained `gru_attention` seed-0 checkpoint reached 88.4%, but
disabling its attention branch retained 88.0%. It should therefore be described
as a strong locally trained **GRU-based checkpoint**, not evidence that
attention improves GRU.

The completion-bonus screen failed its predefined gate: shaped seeds achieved
8%, 15%, and 16% completion, versus 38%, 9%, and 19% for their budget-matched
GRU-64 controls on seeds 70000--70099. It won only one of three comparisons and
reduced both mean and worst-seed completion, so it must not advance.

The next isolated hypothesis averages the two late checkpoints from each
already-completed 10M GRU-64 run. This is evaluated per training seed and does
not select or combine favorable seeds. Use `average_sf_checkpoints.py`; it
averages learned floating tensors while retaining the newest checkpoint's
normalization statistics and metadata.

Run the paired three-seed screen on a fresh evaluation range with:

```bash
python run_sf_gru_swa_screen.py \
  --episodes=100 --seed-start=80000 --device=gpu
```

The launcher evaluates three averaged policies, their three exact latest-
checkpoint parents, and upstream. It writes paired failures, `swa_report.json`,
and refreshes the global research catalog. Advancement requires wins on at
least two matched training seeds, improved mean, non-degraded worst seed, and
a mean within five completion points of upstream.

The late-checkpoint average screen improved two of three paired seeds but did
not advance: averaged policies scored 71%, 74%, and 80% versus 70%, 73%, and
81% for their exact latest-checkpoint parents on seeds 80000--80099. Mean and
worst-seed completion improved slightly (75.0%/71.0% versus 74.7%/70.0%), but
the mean remained nine points behind upstream's 84.0%.

The next isolated hypothesis is a two-stage optimization schedule. Each 10M
GRU-64 run is cloned into a fresh experiment, including an exact source hash,
then continued to 12M steps at learning rate `3e-5`. This tests whether smaller
late updates improve cross-seed reliability without selecting training seeds,
altering the architecture, or modifying the preserved source artifacts:

```bash
python run_sf_gru_lr_finetune.py \
  --seeds=0,1,2 --gpu=0 --learning-rate=3e-5 \
  --target-env-steps=12000000 \
  --train-dir=artifacts/sample_factory_gru_lr_finetune_12m
```

After all three continuations finish, evaluate them and their exact 10M parents
on the fresh `90000--90099` range:

```bash
python run_sf_gru_lr_finetune_screen.py \
  --episodes=100 --seed-start=90000 --device=gpu
```

The stage advances only with at least two paired seed wins, improved mean,
non-degraded worst-seed completion, and a mean within five completion points
of upstream.

The low-learning-rate continuation did not advance on seeds 90000--90099. It
scored 65%, 70%, and 73% versus 62%, 73%, and 73% for the exact 10M parents:
one win, one loss, one tie, unchanged mean completion (69.3%), and 82% for
upstream. This rejects simple late-stage learning-rate reduction.

The next screen isolates a difficulty curriculum: 1M steps at Doom skill 1,
then the same checkpoint continues to 2M total steps in the unchanged standard
skill-5 environment. Architecture, optimizer, total budget, and evaluation are
matched to the existing 2M GRU-64 controls:

```bash
python run_sf_gru_skill_curriculum.py \
  --seeds=0,1,2 --gpu=0 --easy-steps=1000000 --total-steps=2000000 \
  --train-dir=artifacts/sample_factory_gru_skill_curriculum_2m
```

Evaluate against the budget-matched 2M GRU-64 parents and upstream on fresh
seeds `100000--100099`:

```bash
python run_sf_gru_skill_curriculum_screen.py \
  --episodes=100 --seed-start=100000 --device=gpu
```

The skill curriculum failed decisively: all three curriculum policies scored
0% completion versus 42%, 5%, and 16% for their budget-matched controls on
seeds 100000--100099. The next isolated hypothesis addresses the observed
cross-seed optimization variance directly with a fixed GRU initialization
recipe. It preserves the GRU-64 architecture, APPO settings, and 2M budget,
but initializes each recurrent gate matrix orthogonally, each input matrix
with Xavier uniform weights, and all GRU biases to zero.

Train three seeds with:

```bash
python run_sf_transformer_multiseed.py \
  --memory=gru_orthogonal --seeds=0,1,2 --gpus=0,1 \
  --num-policies=1 --steps=2000000 --context=64 \
  --train-dir=artifacts/sample_factory_gru_orthogonal_2m
```

Evaluate on the fresh 110000--110099 range with:

```bash
python run_sf_gru_orthogonal_screen.py \
  --episodes=100 --seed-start=110000 --device=gpu
```

The unchanged gate requires at least two paired wins, improved mean,
non-degraded worst seed, and candidate mean within five completion points of
upstream.

The lag-one screen failed decisively on seeds 120000--120099: all three
lag-controlled candidates scored 0% completion, versus 39%, 6%, and 18% for
their budget-matched GRU-64 controls; upstream scored 85%. This rejects
discarding asynchronous trajectories after one policy version.

The next isolated hypothesis removes collection-time staleness without
discarding rollouts by using Sample Factory's synchronous mode. Its eight
workers, four environments per worker, and rollout 64 collect 2048 samples per
iteration, so two 1024-sample batches consume every sample exactly once. Train
with:

```bash
python run_sf_transformer_multiseed.py \
  --memory=gru_sync --seeds=0,1,2 --gpus=0,1 \
  --num-policies=1 --steps=2000000 --context=64 \
  --train-dir=artifacts/sample_factory_gru_sync_2m
```

Evaluate on fresh seeds 130000--130099 with:

```bash
python run_sf_gru_sync_screen.py \
  --episodes=100 --seed-start=130000 --device=gpu
```

The same four-part stage gate applies.

The synchronous screen failed on seeds 130000--130099. Candidates scored 0%,
0%, and 35% versus 40%, 8%, and 25% for matched asynchronous controls; upstream
scored 87%. One paired win, lower mean, and a zero worst seed reject synchronous
collection as a reliability fix.

The next isolated hypothesis targets the repeated early reward~4 collapse by
increasing only the symmetric-KL exploration coefficient from 0.001 to 0.003:

```bash
python run_sf_transformer_multiseed.py \
  --memory=gru_explore --seeds=0,1,2 --gpus=0,1 \
  --num-policies=1 --steps=2000000 --context=64 \
  --train-dir=artifacts/sample_factory_gru_explore_2m

python run_sf_gru_explore_screen.py \
  --episodes=100 --seed-start=140000 --device=gpu
```

The same four-part stage gate applies; do not tune the coefficient on this seed
range if it fails.

The stronger-exploration screen failed on seeds `140000--140099`. Candidates
scored 10%, 22%, and 7% versus 35%, 6%, and 22% for their matched controls;
upstream scored 89%. One paired win and a lower mean (13% versus 21%) reject
the larger exploration coefficient.

The next isolated hypothesis tests whether Sample Factory's reward-based
checkpoint selection is misaligned with corridor completion. It does not
retrain or modify any policy. For each preserved 10M GRU-64 training seed, it
evaluates the reward-best checkpoint and two late snapshots on development
seeds `150000--150099`, selecting by completion, then mean reward, then a fixed
filename tie-break. It then compares those frozen selections with the original
reward-best checkpoints on disjoint holdout seeds `160000--160099`:

```bash
python run_sf_gru_checkpoint_selection_screen.py \
  --episodes=100 \
  --selection-seed-start=150000 \
  --holdout-seed-start=160000 \
  --device=gpu
```

The selection manifest records exact checkpoint hashes and development
results. Holdout outcomes never influence selection. The unchanged four-part
gate is applied only to holdout results.

The completion-aligned checkpoint-selection screen improved the preserved
10M GRU-64 runs on the disjoint `160000--160099` holdout. Selected late
checkpoints scored 66%, 78%, and 79% completion versus 59%, 70%, and 80% for
the original reward-best checkpoints: two paired wins, mean 74.3% versus
69.7%, and worst seed 66% versus 59%. The method nevertheless failed the
predefined upstream-proximity gate because the upstream checkpoint scored
86.0%, leaving an 11.7-point mean gap. The selection rule is therefore a
useful operational correction, but not the requested upstream-beating method.

The next isolated training hypothesis is loss-preserving recurrent-state
refresh. Sample Factory stores recurrent states generated by the behavior
policy; with asynchronous APPO these states can be stale when the learner
starts each recurrence. Classic loss-free burn-in would discard samples and
change the effective update budget. Instead, `gru_state_refresh` consumes 128
learner steps, recomputes the midpoint hidden state with current parameters,
and detaches it at step 64. This preserves the GRU-64 gradient horizon, the
1024-sample minibatch, and policy/value loss on every sample while halving the
number of behavior-policy state boundaries.

Train three 2M-step seeds and evaluate on fresh seeds `170000--170099`:

```bash
python run_sf_transformer_multiseed.py \
  --memory=gru_state_refresh --seeds=0,1,2 --gpus=0,1 \
  --num-policies=1 --steps=2000000 \
  --train-dir=artifacts/sample_factory_gru_state_refresh_2m

python run_sf_gru_state_refresh_screen.py \
  --episodes=100 --seed-start=170000 --device=gpu
```

List these definitions from the code:

```bash
python sf_train_corridor.py --list-methods
```

## Canonical training entry point

Use `--method` for new work. The older `--memory` argument remains supported so
existing commands and saved configurations are not broken.

```bash
# CNN + APPO
python sf_train_corridor.py --method=cnn --experiment=example_cnn \
  --train_dir=artifacts/examples

# Standard GRU + APPO (recurrence 32)
python sf_train_corridor.py --method=gru --experiment=example_gru \
  --train_dir=artifacts/examples

# Long-context GRU + APPO (recurrence 64)
python sf_train_corridor.py --method=gru_long --experiment=example_gru_long \
  --train_dir=artifacts/examples

# GRU + optional attention + APPO
python sf_train_corridor.py --method=gru_attention \
  --experiment=example_gru_attention --train_dir=artifacts/examples

# Transformer + APPO
python sf_train_corridor.py --method=transformer \
  --experiment=example_transformer --train_dir=artifacts/examples
```

All matched defaults—visual encoder, optimizer, workers, reward, frame skip,
and batch settings—remain in `sf_train_corridor.py`. Architecture names and
recurrent settings are centralized in `sf_methods.py`. Optional custom temporal
cores are implemented together in `sf_temporal_cores.py`. The old
`sf_transformer_core.py` module remains as a compatibility shim.

## Controlled GRU recurrence study

The current study tests a method change rather than selecting a lucky seed:

```bash
python run_sf_gru_recurrence_study.py \
  --seeds=0,1,2 --recurrences=32,64 --gpu=0 --steps=2000000 \
  --train-dir=artifacts/sample_factory_gru_recurrence_2m
```

It runs sequentially to avoid the VizDoom lock/shutdown issue observed with
multiple independent training processes. GRU-64 advances to a 10M-step study
only if it improves average and worst-seed completion in the 2M screen.

## Upstream Hugging Face reference

The upstream checkpoint is not trained by `sf_train_corridor.py`:

```bash
python -m sample_factory.huggingface.load_from_hub \
  -r MattStammers/vizdoom_deadly_corridor -d ./artifacts
```

In this repository it is stored as experiment `deadly-corridor-upstream` and
evaluated with:

```bash
python sf_evaluate_corridor.py \
  --experiment=deadly-corridor-upstream --train-dir=artifacts \
  --policy-index=0 --checkpoint=best --episodes=500 --seed-start=20000 \
  --device=gpu --results-root=results/upstream_gru_policy_0_best_500
```

## Evaluation rule

Architecture claims require multiple training seeds and identical unseen
evaluation seeds. Report mean completion, worst-seed completion, death rate,
and paired failure counts. A single best checkpoint is useful operationally but
does not establish a reliable method improvement.

## Reproducible research harness

`sf_research_harness.py` is the canonical orchestration entry point for the
current controlled GRU study. It wraps the existing trainers and evaluator; it
does not duplicate model implementations.

The overall research objective is not restricted to GRU. A candidate may use
another architecture or training method if it is introduced as one controlled
hypothesis and evaluated through the same multi-seed stage gates. GRU-64 is the
current candidate because the earlier Transformer and GTrXL candidates failed.

```bash
python sf_research_harness.py status \
  --train-dir=artifacts/sample_factory_gru_recurrence_2m

python sf_research_harness.py evaluate \
  --train-dir=artifacts/sample_factory_gru_recurrence_2m \
  --results-root=results/sample_factory_gru_recurrence_2m \
  --episodes=100 --seed-start=30000

python sf_research_harness.py summarize \
  --results-root=results/sample_factory_gru_recurrence_2m \
  --output=results/sample_factory_gru_recurrence_2m/harness_report.json

python sf_research_harness.py catalog \
  --artifacts-root=artifacts --results-root=results \
  --output=results/research_catalog.json
```

The versioned project skill under `skills/vizdoom-gru-research/` teaches Codex
to use this harness and enforce the same stage gates in later sessions.
The catalog indexes every saved evaluation, training configuration, checkpoint,
and log while leaving the underlying artifacts untouched.

## Multi-seed GRU ensemble screen

The 10M-step GRU-64 confirmation improved over GRU-32 on all three seeds, but
its mean completion was 74.0% versus 87.4% for upstream on matched seeds
40000–40499. The three GRU-64 policies nevertheless had substantially
different death sets. The next isolated hypothesis is therefore an inference
ensemble, not another training change. Each policy keeps its own recurrent
state and deterministic actions are combined branch-by-branch by majority
vote. Three-way ties use a fixed member selected before evaluation.

Repeat `--ensemble-experiment` for all members:

```bash
python sf_evaluate_corridor.py \
  --train-dir=artifacts/sample_factory_gru_recurrence_10m_confirm \
  --ensemble-experiment=sf_corridor_gru_r64_seed_0 \
  --ensemble-experiment=sf_corridor_gru_r64_seed_1 \
  --ensemble-experiment=sf_corridor_gru_r64_seed_2 \
  --ensemble-tie-break-index=2 --checkpoint=best \
  --episodes=100 --seed-start=50000 --device=gpu \
  --results-root=results/sample_factory_gru64_ensemble_screen
```

The `50000` range is a fresh development screen. Do not use the already
inspected `40000` range to claim ensemble performance.
