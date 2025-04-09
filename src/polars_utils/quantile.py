from typing import Iterable, Literal, Optional

import polars as pl

from polars_utils import match_name, DF
from polars_utils.weights import Weight, into_normalized_weight
from polars_utils.stats import mean
from polars import _typing as pt


def expr_quantile(
    x: pl.Expr,
    *,
    w: Optional[pt.IntoExprColumn] = None,
    endpoint: Optional[Literal["left", "right", "mid"]] = "mid",
) -> pl.Expr:
    """
    Computes (weighted) quantiles of a column.
    """

    d = 0 if endpoint == "right" else 1.0 if endpoint == "left" else 0.5

    if w is None:
        return x.rank("ordinal").sub(d).truediv(x.len())

    w = into_normalized_weight(w, null_mask=x)
    right_endpoints = w.sort_by(x).cum_sum().gather(x.rank(method="ordinal") - 1)

    return right_endpoints.sub(d * w).pipe(match_name, x)


def df_quantile(
    df: DF,
    cols: Iterable[str],
    *,
    w: Optional[pt.IntoExprColumn] = None,
    ties: Optional[Literal["min", "max", "average"]] = None,
    by: Iterable[str] = [],
) -> DF:
    endpoint = {"min": "left", "max": "right", "average": "mid", None: None}[ties]

    ranked = df.with_columns(
        pl.col(col)
        .pipe(expr_quantile, w=w, endpoint=endpoint)
        .over(by or pl.lit(1))
        .alias(f"{col}_rank")
        for col in cols
    )

    if ties is None:
        return ranked

    # handle ties in ranking variables
    if ties == "average":
        return ranked.with_columns(
            pl.col(col + "_rank").pipe(mean, w=w).over(*by, col) for col in cols
        )

    if ties == "min":
        return ranked.with_columns(
            pl.col(col + "_rank").min().over(*by, col) for col in cols
        )

    if ties == "max":
        return ranked.with_columns(
            pl.col(col + "_rank").max().over(*by, col) for col in cols
        )

    else:
        raise ValueError(f"invalid tie method: {ties}")


def expr_xtile(
    x: pl.Expr,
    *,
    w: Optional[pt.IntoExprColumn] = None,
    n: int,
    label="{i}",
) -> pl.Expr:
    breaks = pl.linear_space(0, 1, n - 1, closed="none", eager=True).to_list()
    labels = [label.format(i=i + 1, n=n) for i in range(n)]

    return (
        x.pipe(expr_quantile, w=w, endpoint="mid")
        .cut(breaks, labels=labels)
        .alias(x.meta.output_name() + "_bin")
    )


def df_xtile(
    df: DF,
    cols: Iterable[str],
    *,
    w: Optional[pt.IntoExprColumn] = None,
    ties: Optional[Literal["min", "max", "average"]] = None,
    by: Iterable[str] = [],
    n: int,
    label="{i}",
):
    """
    Splits data into bins of roughly equal weight a la `xtile` in stata.
    """

    quantiles = df.pipe(df_quantile, cols, w=w, ties=ties, by=by)
    breaks = pl.linear_space(0, 1, n - 1, closed="none", eager=True).to_list()
    labels = [label.format(i=i + 1, n=n) for i in range(n)]

    return quantiles.with_columns(
        pl.col(c + "_rank").cut(breaks=breaks, labels=labels).alias(c + "_bin")
        for c in cols
    ).drop(c + "_rank" for c in cols)
