"""Regression tests for script helpers (Colab pilot, 2026-11-23)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)          # main() is guarded by __name__ == "__main__"
    return mod


def test_display_path_inside_and_outside_repo(tmp_path):
    """The smoke test crashed when --out-dir was on Google Drive (outside the repo)."""
    mod = _load("test_event_pipeline")
    assert mod.display_path(ROOT / "results" / "x.png") == str(Path("results") / "x.png")
    outside = tmp_path / "drive" / "smoke" / "x.png"
    assert mod.display_path(outside) == str(outside.resolve())
