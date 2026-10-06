import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from prepare_sf_finetune import prepare_finetune


class PrepareFinetuneTests(unittest.TestCase):
    def test_clones_latest_checkpoint_and_rewrites_lr(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            (source / "checkpoint_p0").mkdir(parents=True)
            (source / "config.json").write_text(
                json.dumps({"experiment": "old", "train_dir": "old", "learning_rate": 1e-4}),
                encoding="utf-8",
            )
            torch.save(
                {
                    "env_steps": 10_000_000,
                    "best_performance": np.float64(19.5),
                    "curr_lr": 1e-4,
                    "model": {"weight": torch.tensor([1.0])},
                    "optimizer": {"param_groups": [{"lr": 1e-4, "params": []}], "state": {}},
                },
                source / "checkpoint_p0" / "checkpoint_0001_10000000.pth",
            )
            destination_root = root / "destinations"
            destination = destination_root / "new"
            output = prepare_finetune(source, destination, destination_root, 3e-5, 12_000_000)

            checkpoint = torch.load(output, map_location="cpu", weights_only=False)
            torch.load(output, map_location="cpu", weights_only=True)
            config = json.loads((destination / "config.json").read_text(encoding="utf-8"))
            provenance = json.loads(
                (destination / "finetune_provenance.json").read_text(encoding="utf-8")
            )
            self.assertEqual(checkpoint["curr_lr"], 3e-5)
            self.assertIsInstance(checkpoint["best_performance"], float)
            self.assertEqual(checkpoint["optimizer"]["param_groups"][0]["lr"], 3e-5)
            self.assertEqual(config["experiment"], "new")
            self.assertEqual(config["train_for_env_steps"], 12_000_000)
            self.assertEqual(provenance["source_env_steps"], 10_000_000)


if __name__ == "__main__":
    unittest.main()
