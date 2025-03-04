from typing import Literal

import polars as pl

from polars_utils import match_name
from polars_utils.weights import Weight, into_normalized_weight


def quantile(
    x: pl.Expr,
    *,
    w: Weight = None,
    endpoint: Literal["left", "right", "midpoint"] = "midpoint",
) -> pl.Expr:
    """
    Computes (weighted) quantiles of a column.
    """
    right_quantiles = into_normalized_weight(w).sort_by(x).cum_sum()  # ends with a 1
    left_quantiles = right_quantiles.shift(n=+1, fill_value=0)  # starts with a zero

    match endpoint:
        case "midpoint":
            quantiles = [right_quantiles, left_quantiles]
        case "right":
            quantiles = [right_quantiles]
        case "left":
            quantiles = [left_quantiles]
        case _:
            raise ValueError

    return (
        pl.mean_horizontal(*quantiles)
        .gather(x.rank(method="ordinal") - 1)
        # set name of output
        .pipe(match_name, x)
    )


def xtile(x: pl.Expr, *, n: int, w: Weight = None, label="{i}"):
    """
    Splits data into bins of roughly equal weight a la `xtile` in stata.
    """
    quantiles = x.pipe(quantile, w=w, endpoint="midpoint")

    return quantiles.cut(
        breaks=pl.linear_space(0, 1, n - 1, closed="none", eager=True).to_list(),
        labels=[label.format(i=i + 1, n=n) for i in range(n)],
    )
