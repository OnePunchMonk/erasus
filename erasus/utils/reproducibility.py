"""
erasus.utils.reproducibility — Reproducibility utilities.

Provides comprehensive reproducibility setup including deterministic
algorithms, experiment snapshots, and environment logging.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn as nn


def make_reproducible(
    seed: int = 42,
    deterministic_algorithms: bool = True,
    warn_only: bool = True,
) -> None:
    """
    Set up full reproducibility across all random sources.

    Parameters
    ----------
    seed : int
        Global random seed.
    deterministic_algorithms : bool
        If ``True``, enable PyTorch deterministic algorithms.
        May reduce performance.
    warn_only : bool
        If ``True``, only warn (don't raise) when a non-deterministic
        operation is encountered.
    """
    import random
    import numpy as np

    # Python
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    # NumPy
    np.random.seed(seed)

    # PyTorch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # cuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Deterministic algorithms
    if deterministic_algorithms:
        try:
            torch.use_deterministic_algorithms(True, warn_only=warn_only)
        except TypeError:
            # Older PyTorch versions don't have warn_only
            try:
                torch.use_deterministic_algorithms(True)
            except AttributeError:
                pass

    # Set CUBLAS workspace config for reproducibility
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


class ExperimentSnapshot:
    """
    Captures a reproducibility snapshot of the current experiment state.

    Records:
    - Python / PyTorch / CUDA versions
    - Random state checksums
    - Model architecture hash
    - Hyperparameters
    - Git state (if available)

    Parameters
    ----------
    experiment_name : str
        Human-readable experiment identifier.
    output_dir : str
        Directory to save snapshot files.
    """

    def __init__(
        self,
        experiment_name: str = "experiment",
        output_dir: str = "./experiments",
    ):
        self.experiment_name = experiment_name
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.snapshot: Dict[str, Any] = {}

    def capture(
        self,
        model: Optional[nn.Module] = None,
        config: Optional[Dict[str, Any]] = None,
        seed: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Capture a full reproducibility snapshot.

        Parameters
        ----------
        model : nn.Module, optional
            Model to hash architecture of.
        config : dict, optional
            Experiment hyperparameters.
        seed : int, optional
            Random seed used.

        Returns
        -------
        dict — The complete snapshot.
        """
        self.snapshot = {
            "experiment_name": self.experiment_name,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "environment": self._capture_environment(),
        }

        if seed is not None:
            self.snapshot["seed"] = seed

        if config is not None:
            self.snapshot["config"] = config

        if model is not None:
            self.snapshot["model"] = self._capture_model_info(model)

        self.snapshot["random_states"] = self._capture_random_states()
        self.snapshot["git"] = self._capture_git_info()

        return self.snapshot

    def save(self, filename: Optional[str] = None) -> Path:
        """
        Save the snapshot to a JSON file.

        Returns the path to the saved file.
        """
        if not self.snapshot:
            self.capture()

        if filename is None:
            ts = time.strftime("%Y%m%d_%H%M%S")
            filename = f"{self.experiment_name}_{ts}.json"

        path = self.output_dir / filename
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.snapshot, f, indent=2, default=str)

        return path

    @staticmethod
    def load(path: str) -> Dict[str, Any]:
        """Load a snapshot from file."""
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    # ------------------------------------------------------------------
    # Capture helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _capture_environment() -> Dict[str, str]:
        """Capture system and library versions."""
        env = {
            "python": sys.version,
            "platform": platform.platform(),
            "pytorch": torch.__version__,
            "cuda_available": str(torch.cuda.is_available()),
        }

        if torch.cuda.is_available():
            env["cuda_version"] = torch.version.cuda or "N/A"
            env["cudnn_version"] = str(torch.backends.cudnn.version())
            env["gpu_name"] = torch.cuda.get_device_name(0)
            env["gpu_count"] = str(torch.cuda.device_count())

        try:
            import numpy as np
            env["numpy"] = np.__version__
        except ImportError:
            pass

        try:
            import transformers
            env["transformers"] = transformers.__version__
        except ImportError:
            pass

        return env

    @staticmethod
    def _capture_model_info(model: nn.Module) -> Dict[str, Any]:
        """Capture model architecture hash and parameter counts."""
        # Architecture string hash
        arch_str = str(model)
        arch_hash = hashlib.md5(arch_str.encode()).hexdigest()

        # Parameter counts
        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

        # State dict hash
        state_hash = hashlib.md5(
            str(sorted(model.state_dict().keys())).encode()
        ).hexdigest()[:12]

        return {
            "architecture_hash": arch_hash,
            "state_dict_keys_hash": state_hash,
            "total_parameters": total,
            "trainable_parameters": trainable,
            "frozen_parameters": total - trainable,
        }

    @staticmethod
    def _capture_random_states() -> Dict[str, str]:
        """Capture checksums of current random states."""
        states = {}

        # Python random
        import random
        states["python_random_state_hash"] = hashlib.md5(
            str(random.getstate()).encode()
        ).hexdigest()[:12]

        # NumPy
        try:
            import numpy as np
            np_state = np.random.get_state()
            states["numpy_random_state_hash"] = hashlib.md5(
                str(np_state[1][:10]).encode()
            ).hexdigest()[:12]
        except ImportError:
            pass

        # PyTorch
        torch_state = torch.random.get_rng_state()
        states["torch_random_state_hash"] = hashlib.md5(
            torch_state.numpy().tobytes()[:100]
        ).hexdigest()[:12]

        return states

    @staticmethod
    def _capture_git_info() -> Dict[str, str]:
        """Capture git commit and branch (if in a git repo)."""
        info: Dict[str, str] = {}
        try:
            import subprocess

            info["commit"] = subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                stderr=subprocess.DEVNULL,
            ).decode().strip()

            info["branch"] = subprocess.check_output(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                stderr=subprocess.DEVNULL,
            ).decode().strip()

            # Check for uncommitted changes
            status = subprocess.check_output(
                ["git", "status", "--porcelain"],
                stderr=subprocess.DEVNULL,
            ).decode().strip()
            info["dirty"] = str(len(status) > 0)

        except Exception:
            info["available"] = "false"

        return info


# ======================================================================
# Convenience functions
# ======================================================================


def set_seed(seed: int = 42) -> None:
    """
    Set global random seed (alias for ``make_reproducible``
    without deterministic algorithm enforcement).
    """
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_reproducibility_info() -> Dict[str, Any]:
    """
    Quick reproducibility environment report.
    """
    snapshot = ExperimentSnapshot()
    return snapshot.capture()
