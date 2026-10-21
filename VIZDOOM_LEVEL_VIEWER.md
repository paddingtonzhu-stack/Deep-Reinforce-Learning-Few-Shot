# ViZDoom level viewer

This viewer opens the exact built-in scenario assets used by the project. It
does not train or alter any model.

## Levels

- `deadly_corridor`: move through a narrow defended corridor and reach the armor
  at its far end. This is the source task for the Hugging Face CNN+GRU policy.
- `battle1`: fight and survive in a more open arena. This is the few-shot target.
- `battle2`: fight and survive in a different arena. This is the held-out
  zero-shot transfer test.

Battle is not merely a different texture or lighting condition. Its geometry,
objective, useful movement, and enemy directions differ from Deadly Corridor.

## Run on Windows PowerShell

One-time setup (installs only the small ViZDoom player environment):

```powershell
.\setup_vizdoom_player.ps1
```

Then play either Battle arena:

```powershell
.\play_battle.ps1 -Map 1
.\play_battle.ps1 -Map 2
```

Or inspect any project level directly:

```powershell
.\inspect_vizdoom_level.ps1 -Level deadly_corridor
.\inspect_vizdoom_level.ps1 -Level battle1
.\inspect_vizdoom_level.ps1 -Level battle2
```

The default `spectator` mode lets you control the game through the Doom window.
Click that window and use its normal keyboard/mouse controls. To watch automatic
rollouts instead:

```powershell
.\inspect_vizdoom_level.ps1 -Level battle1 -Mode random -Episodes 2
```

## Run on Linux

```bash
.sf/bin/python inspect_vizdoom_level.py --list
.sf/bin/python inspect_vizdoom_level.py --level deadly_corridor --mode spectator
.sf/bin/python inspect_vizdoom_level.py --level battle1 --mode spectator
.sf/bin/python inspect_vizdoom_level.py --level battle2 --mode spectator
```

Use `--mute` if audio is unavailable or `--headless --mode random` for a quick
non-visual configuration check.
