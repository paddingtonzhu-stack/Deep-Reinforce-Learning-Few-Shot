import tempfile
import unittest
from pathlib import Path

from run_sf_gru_swa_screen import completion_percent


class CompletionPercentTests(unittest.TestCase):
    def test_reads_current_boolean_schema(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "episodes.csv"
            path.write_text("episode,completed\n0,True\n1,False\n2,True\n", encoding="utf-8")
            self.assertAlmostEqual(completion_percent(path), 200.0 / 3.0)

    def test_reads_legacy_outcome_schema(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "episodes.csv"
            path.write_text("episode,outcome\n0,completed\n1,death\n", encoding="utf-8")
            self.assertEqual(completion_percent(path), 50.0)


if __name__ == "__main__":
    unittest.main()
