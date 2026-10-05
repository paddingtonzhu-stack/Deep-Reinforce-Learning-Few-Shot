---
name: vizdoom-gru-research
description: Run, evaluate, and interpret controlled Sample Factory GRU research in this ViZDoom Deadly Corridor project when the goal is to outperform the upstream Hugging Face GRU with a reproducible method rather than a lucky checkpoint.
---

# ViZDoom GRU research

Use the repository harness as the source of execution behavior; do not recreate
training loops inside the skill. Read `SAMPLE_FACTORY_METHODS.md` before
changing architectures or interpreting old artifacts.

## Research invariant

A method improvement must reproduce across training seeds. Never attribute a
gain to an architecture based only on its best checkpoint. Compare matched
training budgets and identical unseen evaluation seeds, and report mean plus
worst-seed completion. Treat the upstream Hugging Face result (84.4% on seeds
20000-20499) as an operational target, not as a multi-seed training baseline.

## Workflow

1. Inspect the active job and existing artifacts before launching anything.
   Use `python sf_research_harness.py status ...` locally or through an already
   authorized SSH session. Do not terminate unrelated processes.
2. Screen one controlled change at a smaller budget. For the current study this
   is GRU recurrence 32 versus 64, seeds 0-2, 2M steps each.
3. Evaluate every seed with the same checkpoint rule, episode count, and unseen
   seed range using `sf_research_harness.py evaluate`.
4. Run `sf_research_harness.py summarize`. Advance only when the candidate wins
   at least two seeds, improves mean completion, and does not reduce worst-seed
   completion.
5. Confirm an advancing method at the full budget, then evaluate on a second
   disjoint seed range. A final claim must include paired failure counts against
   the upstream checkpoint.
6. After every training or evaluation stage, run
   `python sf_research_harness.py catalog`. Preserve successes and failures;
   the catalog is evidence for choosing the next hypothesis.

Use fresh artifact/result directories for new protocols. Never reuse evaluation
seeds for training or tune directly on the final confirmation range. Preserve
the 88.4% locally trained checkpoint and upstream artifacts as read-only
references.

For exact stage commands, acceptance rules, and current fixed paths, read
[`references/protocol.md`](references/protocol.md).

## Remote execution

Credentials are never part of this skill or repository. Use an existing
user-authenticated SSH session or key-based SSH. Password files must not be
printed, committed, copied into command arguments, or embedded in helpers.
Starting a requested experiment and reading its logs are in scope; deleting
artifacts, killing ambiguous processes, or replacing champion checkpoints
requires explicit confirmation.
