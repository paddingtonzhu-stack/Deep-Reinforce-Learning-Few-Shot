@echo off
setlocal
cd /d "%~dp0"
".python-corridor\python.exe" run_deadly_corridor.py --algo=APPO --env=doom_deadly_corridor --train_dir=artifacts --experiment=deadly-corridor-upstream --device=cpu --max_num_episodes=20 --policy_index=0 --load_checkpoint_kind=best --eval_deterministic=True --no_render
endlocal
