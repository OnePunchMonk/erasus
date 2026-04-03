"""erasus.utils — Utility functions and helpers."""

from erasus.utils.decorators import experimental
from erasus.utils.helpers import (
    count_parameters,
    model_size_mb,
    freeze_model,
    unfreeze_model,
    freeze_except,
    to_device,
    gradient_norm,
    Timer,
    ensure_dir,
    hash_config,
    save_json,
    load_json,
)

from erasus.utils.distributed import (
    is_distributed,
    get_rank,
    get_world_size,
    is_main_process,
    setup_distributed,
    cleanup_distributed,
    wrap_model_ddp,
    all_reduce_mean,
    broadcast_object,
)

from erasus.utils.batch import unpack_batch
from erasus.utils.profiling import UnlearningProfiler, profile_section, profile_model_memory
from erasus.utils.reproducibility import make_reproducible, ExperimentSnapshot

__all__ = [
    # decorators
    "experimental",
    # helpers
    "count_parameters",
    "model_size_mb",
    "freeze_model",
    "unfreeze_model",
    "freeze_except",
    "to_device",
    "gradient_norm",
    "Timer",
    "ensure_dir",
    "hash_config",
    "save_json",
    "load_json",
    # distributed
    "is_distributed",
    "get_rank",
    "get_world_size",
    "is_main_process",
    "setup_distributed",
    "cleanup_distributed",
    "wrap_model_ddp",
    "all_reduce_mean",
    "broadcast_object",
    # profiling
    "UnlearningProfiler",
    "profile_section",
    "profile_model_memory",
    # batch
    "unpack_batch",
    # reproducibility
    "make_reproducible",
    "ExperimentSnapshot",
]
