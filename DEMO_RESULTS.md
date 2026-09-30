# Demo checkpoint evaluation

- Checkpoint: `artifacts/demo/best/best_model.zip`
- Training decisions: 20,000
- Evaluation seeds: 2000 through 2019
- Episodes: 20
- Deterministic policy: yes
- Mean reward: 0.10 (scaled)
- Standard deviation: 1.55
- Median reward: 0.95
- Positive-reward episodes: 16/20
- Mean policy decisions per episode: 17.9

Rewards are ViZDoom Basic rewards multiplied by 0.01. A timeout returns -3.00; a fast successful episode typically returns 0.95.
