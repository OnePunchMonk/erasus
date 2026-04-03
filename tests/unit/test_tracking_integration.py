"""
Tests for deeper experiment tracking helpers.
"""

from __future__ import annotations

from erasus.experiments.experiment_tracker import ExperimentTracker


def test_local_tracker_logs_curve_and_model_version(tmp_path):
    tracker = ExperimentTracker("tracking_test", backend="local", output_dir=str(tmp_path))
    tracker.log_curve("forget_loss", [0, 1, 2], [1.5, 1.0, 0.5])
    tracker.log_model_version(
        name="model",
        path="checkpoint.pt",
        aliases=["latest"],
        metadata={"strategy": "npo"},
    )

    run_info = tracker.finish()
    run_dir = tmp_path / next(tmp_path.iterdir()).name

    assert (run_dir / "forget_loss_curve.json").exists()
    assert (run_dir / "model_version.json").exists()
    assert run_info["backend"] == "local"


def test_log_unlearning_run_bundle(tmp_path):
    tracker = ExperimentTracker("bundle_test", backend="local", output_dir=str(tmp_path))
    tracker.log_unlearning_run(
        strategy="altpo",
        selector="random",
        metrics={"forget_acc": 0.1, "retain_acc": 0.9},
        curves={"retain_acc": {"x": [0, 1], "y": [0.7, 0.9]}},
        model_path="model.pt",
        metadata={"issue": 79},
    )
    tracker.finish()

    run_dir = tmp_path / next(tmp_path.iterdir()).name
    assert (run_dir / "summary.json").exists()
    assert (run_dir / "retain_acc_curve.json").exists()
    assert (run_dir / "unlearned_model_version.json").exists()
