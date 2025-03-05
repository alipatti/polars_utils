import polars as pl
from polars import _typing as pt
from typing import Iterable, Optional

from polars_utils.stats import mean
from polars_utils.weights import Weight


def absorb(
    x: pl.Expr,
    fixed_effects: Iterable[pt.IntoExpr],
    *,
    w: Weight = None,
    by: Optional[Iterable[pt.IntoExpr]] = None,
    add_back_mean=True,
):
    """
    Absorbs (categorical) fixed effects by demeaning.
    """
    # if by isn't passed, do everything together
    by = by or [pl.lit(1)]

    return (
        x
        # subtract mean
        - x.pipe(mean, w=w).over(*by, *fixed_effects)
        # add back mean within cell
        + (x.pipe(mean, w=w).over(*by) if add_back_mean else 0)
    )
