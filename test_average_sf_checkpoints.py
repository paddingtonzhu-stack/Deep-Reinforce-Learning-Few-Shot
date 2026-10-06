from pathlib import Path
import tempfile
import unittest

import torch

from average_sf_checkpoints import average_checkpoints


def _save(path: Path, value: float, normalizer: float, *, shape=(2,)) -> None:
    torch.save(
        {
            "train_step": int(value),
            "model": {
                "core.weight": torch.full(shape, value, dtype=torch.float32),
                "obs_normalizer.running_mean": torch.tensor([normalizer], dtype=torch.float64),
                "integer_buffer": torch.tensor([int(value)], dtype=torch.int64),
            },
            "optimizer": {"from": value},
        },
        path,
    )


class AverageCheckpointTests(unittest.TestCase):
    def test_average_uses_latest_metadata_and_normalizers(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first, latest, output = (root / name for name in ("first.pth", "latest.pth", "out.pth"))
            _save(first, 2.0, 20.0)
            _save(latest, 4.0, 40.0)

            average_checkpoints([first, latest], output)
            result = torch.load(output, map_location="cpu", weights_only=False)

            self.assertEqual(result["train_step"], 4)
            self.assertEqual(result["optimizer"], {"from": 4.0})
            self.assertTrue(
                torch.equal(result["model"]["core.weight"], torch.tensor([3.0, 3.0]))
            )
            self.assertTrue(
                torch.equal(
                    result["model"]["obs_normalizer.running_mean"],
                    torch.tensor([40.0], dtype=torch.float64),
                )
            )
            self.assertTrue(torch.equal(result["model"]["integer_buffer"], torch.tensor([4])))

    def test_average_rejects_incompatible_shapes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first, latest, output = (root / name for name in ("first.pth", "latest.pth", "out.pth"))
            _save(first, 2.0, 20.0, shape=(2,))
            _save(latest, 4.0, 40.0, shape=(3,))

            with self.assertRaisesRegex(ValueError, "incompatible"):
                average_checkpoints([first, latest], output)


if __name__ == "__main__":
    unittest.main()
