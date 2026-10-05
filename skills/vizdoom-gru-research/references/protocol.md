# Controlled GRU protocol

## Fixed references

- Repository: `~/research/drl/Deep-Reinforce-Learning-Few-Shot`
- Linux environment: `.sf`
- Upstream experiment: `deadly-corridor-upstream` under `artifacts/`
- Upstream result: 84.4% completion over seeds 20000-20499
- Local champion: `sf_corridor_gru_attention_seed_0`, best policy 0
- Local champion result: 88.4% over seeds 20000-20499
- Attention-disabled result: 88.0%; attention has no demonstrated benefit

Do not describe the local champion as proof that attention beats GRU. Its gain
over upstream is primarily a better set of GRU weights and checkpoint selection.

## Current screen

Training directory:

```text
artifacts/sample_factory_gru_recurrence_2m
```

Status:

```bash
python sf_research_harness.py status \
  --train-dir=artifacts/sample_factory_gru_recurrence_2m
```

Training, when no run already exists:

```bash
python sf_research_harness.py train \
  --seeds=0,1,2 --recurrences=32,64 --gpu=0 --steps=2000000 \
  --train-dir=artifacts/sample_factory_gru_recurrence_2m
```

Matched screening evaluation uses development seeds starting at 30000, keeping
the earlier 20000-20499 comparison range separate:

```bash
python sf_research_harness.py evaluate \
  --train-dir=artifacts/sample_factory_gru_recurrence_2m \
  --results-root=results/sample_factory_gru_recurrence_2m \
  --episodes=100 --seed-start=30000 --checkpoint=best --device=gpu

python sf_research_harness.py summarize \
  --results-root=results/sample_factory_gru_recurrence_2m \
  --output=results/sample_factory_gru_recurrence_2m/harness_report.json

python sf_research_harness.py catalog \
  --artifacts-root=artifacts --results-root=results \
  --output=results/research_catalog.json
```

## Stage gates

Advance GRU-64 from 2M screening only if all hold:

1. It beats GRU-32 in at least two of three training seeds.
2. Its mean completion is higher.
3. Its worst-seed completion is no lower.

At the full budget, require the same multi-seed comparison with at least 500
episodes per policy. Then compare the selected candidate and upstream checkpoint
on a second disjoint 500-seed range. Report paired discordant outcomes rather
than only aggregate percentages.

If GRU-64 fails the screen, do not rationalize or extend it. Record the negative
result and choose one new hypothesis. Suitable next hypotheses should target a
known GRU failure mechanism while preserving the controlled baseline, such as
burn-in for recurrent minibatches or completion-aligned training. Do not stack
multiple changes in one experiment.

## Artifact safety

- Never overwrite `artifacts/deadly-corridor-upstream`.
- Never overwrite `artifacts/sample_factory_gru_attention_single_policy`.
- New protocols receive a new training directory and results root.
- `best` and `latest` are different checkpoint-selection rules; do not mix them
  within one comparison.
- Evaluation is deterministic and must not update policy weights.
