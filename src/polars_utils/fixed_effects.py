import polars as pl
from typing import Iterable, Optional

from polars_utils import IntoExpr
from polars_utils.stats import mean


def absorb(
    x: pl.Expr,
    fixed_effects: Iterable[IntoExpr],
    *,
    w: Optional[IntoExpr] = None,
    by: Optional[Iterable[IntoExpr]] = None,
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
