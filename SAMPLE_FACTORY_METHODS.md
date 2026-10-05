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
| `gru_attention` | CNN + GRU + gated optional attention + APPO | Our experimental architecture | 88.4% enabled vs 88.0% disabled on 500 matched episodes; no demonstrated attention benefit |
| `transformer` | CNN + causal finite-memory Transformer + APPO | Experimental architecture | Failed: 0.3% aggregate completion |
| `gtrxl` | CNN + gated Transformer-XL-style core + APPO | Experimental architecture | Failed to learn beyond approximately reward 5 |
| `upstream_gru` | Published CNN + GRU + APPO checkpoint | Evaluation-only Hugging Face reference | 84.4% on 500 episodes, seeds 20000–20499 |

The locally trained `gru_attention` seed-0 checkpoint reached 88.4%, but
disabling its attention branch retained 88.0%. It should therefore be described
as a strong locally trained **GRU-based checkpoint**, not evidence that
attention improves GRU.

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
```

The versioned project skill under `skills/vizdoom-gru-research/` teaches Codex
to use this harness and enforce the same stage gates in later sessions.
