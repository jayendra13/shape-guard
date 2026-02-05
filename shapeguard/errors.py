"""
ShapeGuard error types with rich diagnostic information.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from shapeguard.core import Dim


class ShapeGuardError(Exception):
    """
    Base exception for shape contract violations.

    Provides detailed diagnostic information to help users fix shape errors
    without re-running code.
    """

    def __init__(
        self,
        message: str,
        *,
        function: str | None = None,
        argument: str | None = None,
        expected: Any = None,
        actual: Any = None,
        reason: str | None = None,
        bindings: str | None = None,
    ) -> None:
        self.function = function
        self.argument = argument
        self.expected = expected
        self.actual = actual
        self.reason = reason
        self.bindings = bindings
        super().__init__(message)

    def __str__(self) -> str:
        lines = ["ShapeGuardError:"]
        if self.function:
            lines.append(f"  function: {self.function}")
        if self.argument:
            lines.append(f"  argument: {self.argument}")
        if self.expected is not None:
            lines.append(f"  expected: {self._format_shape(self.expected)}")
        if self.actual is not None:
            lines.append(f"  actual:   {self._format_shape(self.actual)}")
        if self.reason:
            lines.append(f"  reason:   {self.reason}")
        if self.bindings:
            lines.append(f"  bindings: {self.bindings}")
        return "\n".join(lines)

    @staticmethod
    def _format_shape(shape: Any) -> str:
        """Format a shape spec or tuple for display."""
        if isinstance(shape, tuple):
            return "(" + ", ".join(str(d) for d in shape) + ")"
        return str(shape)


class UnificationError(ShapeGuardError):
    """
    Raised when a symbolic dimension cannot unify with a concrete value.

    This happens when the same Dim object is constrained to different
    integer values across arguments.
    """

    def __init__(
        self,
        dim: Dim,
        expected_value: int,
        expected_source: str,
        actual_value: int,
        actual_source: str,
    ) -> None:
        self.dim = dim
        self.expected_value = expected_value
        self.expected_source = expected_source
        self.actual_value = actual_value
        self.actual_source = actual_source

        reason = (
            f"dimension '{dim.name}' bound to {expected_value} from {expected_source}, "
            f"but got {actual_value} from {actual_source}"
        )
        super().__init__(
            reason,
            reason=reason,
        )


class RankMismatchError(ShapeGuardError):
    """Raised when array rank (number of dimensions) doesn't match spec."""

    def __init__(
        self,
        *,
        function: str | None = None,
        argument: str | None = None,
        expected_rank: int | str,
        actual_rank: int,
        expected_shape: Any,
        actual_shape: tuple[int, ...],
        bindings: str | None = None,
    ) -> None:
        reason = f"expected rank {expected_rank}, got rank {actual_rank}"
        super().__init__(
            reason,
            function=function,
            argument=argument,
            expected=expected_shape,
            actual=actual_shape,
            reason=reason,
            bindings=bindings,
        )


class DimensionMismatchError(ShapeGuardError):
    """Raised when a specific dimension doesn't match the expected value."""

    def __init__(
        self,
        *,
        function: str | None = None,
        argument: str | None = None,
        dim_index: int,
        expected_value: int,
        actual_value: int,
        expected_shape: Any,
        actual_shape: tuple[int, ...],
        bindings: str | None = None,
    ) -> None:
        reason = f"dim[{dim_index}] expected {expected_value}, got {actual_value}"
        super().__init__(
            reason,
            function=function,
            argument=argument,
            expected=expected_shape,
            actual=actual_shape,
            reason=reason,
            bindings=bindings,
        )
