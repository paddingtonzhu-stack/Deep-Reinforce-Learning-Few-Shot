"""Run the existing audited trainer on the held-out Corridor spawn variant."""
import sys

import sf_train_corridor as trainer
from sf_corridor_spawn_variant import ENV_NAME, register_spawn_variant


def main():
    original = trainer.register_vizdoom_components

    def register_all():
        original()
        register_spawn_variant()

    trainer.register_vizdoom_components = register_all
    if not any(arg == "--env" or arg.startswith("--env=") for arg in sys.argv[1:]):
        sys.argv.append(f"--env={ENV_NAME}")
    return trainer.main()


if __name__ == "__main__":
    raise SystemExit(main())
