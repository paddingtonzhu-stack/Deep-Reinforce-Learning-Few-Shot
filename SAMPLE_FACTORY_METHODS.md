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
