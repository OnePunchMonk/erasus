"""erasus.utils — Utility functions and helpers."""

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

__all__ = [
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
]
