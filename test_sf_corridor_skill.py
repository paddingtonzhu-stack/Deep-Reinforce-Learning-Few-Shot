import unittest

from sf_corridor_objective import configure_doom_skill


class FakeGame:
    def __init__(self):
        self.skill = None

    def set_doom_skill(self, skill):
        self.skill = skill


class FakeBase:
    def __init__(self):
        self.game = None

    def _create_doom_game(self, mode):
        self.game = FakeGame()


class FakeEnv:
    def __init__(self):
        self.unwrapped = FakeBase()


class DoomSkillTests(unittest.TestCase):
    def test_skill_is_applied_after_game_creation(self):
        env = configure_doom_skill(FakeEnv(), 1)
        env.unwrapped._create_doom_game("algo")
        self.assertEqual(env.unwrapped.game.skill, 1)


if __name__ == "__main__":
    unittest.main()
