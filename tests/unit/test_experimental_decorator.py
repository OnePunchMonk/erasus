"""
Tests for the @experimental decorator.
"""

from __future__ import annotations

import pytest


class TestExperimentalDecorator:
    def test_function_warns_and_preserves_result(self):
        from erasus.utils import experimental

        @experimental
        def add(a: int, b: int) -> int:
            return a + b

        with pytest.warns(UserWarning, match="add is experimental"):
            result = add(2, 3)

        assert result == 5
        assert getattr(add, "__erasus_experimental__", False) is True

    def test_class_warns_on_instantiation(self):
        from erasus.utils import experimental

        @experimental(message="Toy is experimental.")
        class Toy:
            def __init__(self, value: int) -> None:
                self.value = value

        with pytest.warns(UserWarning, match="Toy is experimental."):
            toy = Toy(7)

        assert toy.value == 7
        assert getattr(Toy, "__erasus_experimental__", False) is True
