"""
Shape specification parsing and matching.
"""

from __future__ import annotations

from typing import Any

from shapeguard._compat import get_shape
from shapeguard.core import Dim, UnificationContext, _EllipsisType
from shapeguard.errors import (
    DimensionMismatchError,
    RankMismatchError,
    ShapeGuardError,
)

# Type alias for shape specifications
# Each element can be: int (exact match), Dim (symbolic), None (wildcard),
# or ... / ELLIPSIS (variable leading dims)
ShapeSpec = tuple[int | Dim | None | _EllipsisType, ...]


def _has_ellipsis(spec: ShapeSpec) -> bool:
    """Check if spec contains an ellipsis."""
    return any(s is ... or isinstance(s, _EllipsisType) for s in spec)


# Type alias for spec parts without ellipsis
SpecPart = tuple[int | Dim | None, ...]


def _filter_ellipsis(items: ShapeSpec) -> SpecPart:
    """Filter out ellipsis from spec items."""
    return tuple(s for s in items if not (s is ... or isinstance(s, _EllipsisType)))


def _split_ellipsis_spec(spec: ShapeSpec) -> tuple[SpecPart, SpecPart]:
    """
    Split a spec at the ellipsis into (before, after) parts.

    The returned tuples do not contain ellipsis elements.

    Raises ValueError if more than one ellipsis.
    """
    ellipsis_indices = [i for i, s in enumerate(spec) if s is ... or isinstance(s, _EllipsisType)]

    if len(ellipsis_indices) > 1:
        raise ValueError("Shape spec cannot contain more than one ellipsis")

    if not ellipsis_indices:
        return _filter_ellipsis(spec), ()

    idx = ellipsis_indices[0]
    return _filter_ellipsis(spec[:idx]), _filter_ellipsis(spec[idx + 1 :])


def match_shape(
    actual: tuple[int, ...],
    spec: ShapeSpec,
    ctx: UnificationContext,
    source: str,
) -> None:
    """
    Match an actual shape against a specification.

    Args:
        actual: The actual shape to check
        spec: The shape specification (can include ... for variable dims)
        ctx: Unification context for tracking dimension bindings
        source: Description of where this shape came from (for error messages)

    Raises:
        RankMismatchError: If the number of dimensions doesn't match
        DimensionMismatchError: If a concrete dimension doesn't match
        UnificationError: If a symbolic dimension conflicts with prior binding

    Examples:
        match_shape((3, 4), (n, m), ctx, "x")       # exact rank match
        match_shape((2, 3, 4), (..., n, m), ctx, "x")  # ellipsis matches (2,)
        match_shape((3, 4), (..., n, m), ctx, "x")     # ellipsis matches ()
    """
    # Handle ellipsis in spec
    if _has_ellipsis(spec):
        before, after = _split_ellipsis_spec(spec)
        required_dims = len(before) + len(after)

        if len(actual) < required_dims:
            raise RankMismatchError(
                expected_rank=f"{required_dims}+",  # "2+" means at least 2
                actual_rank=len(actual),
                expected_shape=spec,
                actual_shape=actual,
                bindings=ctx.format_bindings(),
            )

        # Match the 'before' part (leading fixed dims)
        for i, spec_dim in enumerate(before):
            _match_dim(actual[i], spec_dim, i, actual, spec, ctx, source)

        # Match the 'after' part (trailing fixed dims)
        # These align from the end
        offset = len(actual) - len(after)
        for i, spec_dim in enumerate(after):
            actual_idx = offset + i
            _match_dim(actual[actual_idx], spec_dim, actual_idx, actual, spec, ctx, source)

        return

    # No ellipsis: require exact rank match
    if len(actual) != len(spec):
        raise RankMismatchError(
            expected_rank=len(spec),
            actual_rank=len(actual),
            expected_shape=spec,
            actual_shape=actual,
            bindings=ctx.format_bindings(),
        )

    # Check each dimension (filter for type narrowing, no ellipsis present)
    spec_dims = _filter_ellipsis(spec)
    for i, spec_dim in enumerate(spec_dims):
        _match_dim(actual[i], spec_dim, i, actual, spec, ctx, source)


def _match_dim(
    actual_dim: int,
    spec_dim: int | Dim | None,
    index: int,
    actual_shape: tuple[int, ...],
    spec: ShapeSpec,
    ctx: UnificationContext,
    source: str,
) -> None:
    """Match a single dimension against its spec."""
    dim_source = f"{source}[{index}]"

    if spec_dim is None:
        # Wildcard: accept any value
        return

    elif isinstance(spec_dim, Dim):
        # Symbolic dimension: unify with context
        ctx.bind(spec_dim, actual_dim, dim_source)

    elif isinstance(spec_dim, int):
        # Concrete dimension: must match exactly
        if actual_dim != spec_dim:
            raise DimensionMismatchError(
                dim_index=index,
                expected_value=spec_dim,
                actual_value=actual_dim,
                expected_shape=spec,
                actual_shape=actual_shape,
                bindings=ctx.format_bindings(),
            )

    else:
        raise TypeError(
            f"Invalid spec element at position {index}: {spec_dim!r} "
            f"(expected int, Dim, None, or ...)"
        )


def check_shape(
    x: Any,
    spec: ShapeSpec,
    name: str = "array",
    *,
    ctx: UnificationContext | None = None,
) -> UnificationContext:
    """
    Check that an array's shape matches a specification.

    This is the standalone shape checking function for use outside decorators.

    Args:
        x: Array-like object to check
        spec: Shape specification to match against
        name: Name to use in error messages
        ctx: Optional unification context (created if not provided)

    Returns:
        The unification context (useful for chaining checks)

    Raises:
        ShapeGuardError: If shape doesn't match specification

    Example:
        from shapeguard import check_shape, Dim

        n = Dim("n")
        check_shape(x, (n, 128), name="input")
        check_shape(y, (n, 64), name="output")  # n must match
    """
    if ctx is None:
        ctx = UnificationContext()

    actual = get_shape(x)

    try:
        match_shape(actual, spec, ctx, name)
    except ShapeGuardError as e:
        # Add name context to error
        e.argument = name
        raise

    return ctx


def format_spec(spec: ShapeSpec) -> str:
    """Format a shape spec for display in error messages."""

    def fmt_dim(d: int | Dim | None | _EllipsisType) -> str:
        if d is None:
            return "*"
        elif d is ... or isinstance(d, _EllipsisType):
            return "..."
        elif isinstance(d, Dim):
            return d.name
        else:
            return str(d)

    return "(" + ", ".join(fmt_dim(d) for d in spec) + ")"
