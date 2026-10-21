"""Use the audited fixed-seed evaluator on the held-out spawn variant."""
import sf_evaluate_corridor as evaluator
from sf_corridor_spawn_variant import ENV_NAME, register_spawn_variant


def register_components():
    evaluator.register_vizdoom_components()
    evaluator.register_transformer_core()
    evaluator.register_corridor_actor_critic()
    register_spawn_variant(evaluator.OutcomeInfoWrapper)


def main():
    evaluator.EVAL_ENV = ENV_NAME
    evaluator.register_components = register_components
    return evaluator.main()


if __name__ == "__main__":
    raise SystemExit(main())
