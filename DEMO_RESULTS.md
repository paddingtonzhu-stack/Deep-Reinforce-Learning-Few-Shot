# Demo checkpoint evaluation

- Checkpoint: `artifacts/demo/best/best_model.zip`
- Training decisions: 30,000
- Evaluation seeds: 5000 through 5049
- Episodes: 50
- Deterministic policy: yes
- Mean reward: -0.14 (scaled)
- Standard deviation: 1.76
- Median reward: 0.95
- Positive-reward episodes: 37/50 (74%)
- Mean policy decisions per episode: 21.6

Rewards are ViZDoom Basic rewards multiplied by 0.01. A timeout is roughly -3.10; a fast successful episode typically returns 0.95.

## Saved-checkpoint comparison

All models were evaluated deterministically on the same seeds (5000-5049):

| Training decisions | Positive episodes | Mean reward | Median reward |
|---:|---:|---:|---:|
| 10,000 | 33/50 | -0.62 | 0.95 |
| 20,000 | 34/50 | -0.33 | 0.95 |
| **30,000** | **37/50** | **-0.14** | **0.95** |
