"""
erasus.utils.decorators — Shared utility decorators.
"""

from __future__ import annotations

import functools
import warnings
from typing import Any, Callable, Optional, TypeVar, Union, cast


F = TypeVar("F", bound=Callable[..., Any])


def experimental(
    obj: Optional[F] = None,
    *,
    message: Optional[str] = None,
    category: type[Warning] = UserWarning,
) -> Union[F, Callable[[F], F]]:
    """
    Mark a function or class as experimental.

    The decorated target emits a warning when invoked/instantiated and
    is annotated with ``__erasus_experimental__ = True``.
    """

    def decorator(target: F) -> F:
        note = message or f"{target.__name__} is experimental and may change without notice."

        if isinstance(target, type):
            original_init = target.__init__

            @functools.wraps(original_init)
            def wrapped_init(self: Any, *args: Any, **kwargs: Any) -> None:
                warnings.warn(note, category=category, stacklevel=2)
                original_init(self, *args, **kwargs)

            target.__init__ = wrapped_init  # type: ignore[method-assign]
            setattr(target, "__erasus_experimental__", True)
            setattr(target, "__erasus_experimental_message__", note)
            return target

        @functools.wraps(target)
        def wrapped(*args: Any, **kwargs: Any) -> Any:
            warnings.warn(note, category=category, stacklevel=2)
            return target(*args, **kwargs)

        setattr(wrapped, "__erasus_experimental__", True)
        setattr(wrapped, "__erasus_experimental_message__", note)
        return cast(F, wrapped)

    if obj is not None:
        return decorator(obj)
    return decorator
