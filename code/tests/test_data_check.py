from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from peca.data.check import DatasetError, check_dataset


def _make_frame(root: Path, frame_id: str) -> None:
    for relative in ("line", "gt"):
        directory = root / relative
        directory.mkdir(parents=True, exist_ok=True)
        (directory / f"{frame_id}.png").touch()
    segment = root / "seg"
    segment.mkdir(parents=True, exist_ok=True)
    (segment / f"{frame_id}.png").touch()
    (segment / f"{frame_id}.json").write_text("{}", encoding="utf-8")


class DatasetCheckTests(unittest.TestCase):
    def test_design_sheet_layout(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            sequence = root / "character"
            _make_frame(sequence, "0000")
            _make_frame(sequence, "0001")
            _make_frame(sequence / "ref", "0000")
            config = {
                "dataset": {
                    "name": "synthetic",
                    "root": str(root),
                    "expected_sequences": 1,
                    "expected_frames": 2,
                },
                "protocol": {"name": "design_sheet"},
            }
            report = check_dataset(config)
            self.assertTrue(report["valid"])
            self.assertEqual(report["references"], 1)

    def test_mismatched_segment_sidecar_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            sequence = root / "character"
            _make_frame(sequence, "0000")
            (sequence / "seg/0000.json").unlink()
            config = {
                "dataset": {"name": "synthetic", "root": str(root)},
                "protocol": {"name": "first_frame"},
            }
            with self.assertRaises(DatasetError):
                check_dataset(config)


if __name__ == "__main__":
    unittest.main()
