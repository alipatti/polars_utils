from typing import Optional
import polars as pl

from polars_utils import into_expr, IntoExpr
from polars_utils.weights import into_normalized_weight


def mean(x: pl.Expr, *, w: Optional[IntoExpr] = None) -> pl.Expr:
    """
    Computes the (weighted) mean of an expression.
    """
    if w is None:
        return x.mean()

    return x.dot(into_normalized_weight(w, null_mask=x))


def demean(x: pl.Expr, *, w: Optional[IntoExpr] = None) -> pl.Expr:
    """
    Subtracts off the (weighted) mean of an expression.
    """
    return x - x.pipe(mean, w=w)


def cov(
    x: IntoExpr,
    y: IntoExpr,
    *,
    centered: bool = False,
    w: Optional[IntoExpr] = None,
) -> pl.Expr:
    """
    Computes the (weighted) covaraince of an expression with another expression.
    """
    if w is None and not centered:
        return pl.cov(x, y, ddof=0)

    x = into_expr(x).pipe(demean, w=w) if not centered else into_expr(x)
    y = into_expr(y).pipe(demean, w=w) if not centered else into_expr(y)

    return (x * y).pipe(mean, w=w).alias("cov")


def var(
    x: pl.Expr,
    *,
    w: Optional[IntoExpr] = None,
    center_around: Optional[pl.Expr] = None,
):
    """
    Computes the (weighted) variance of an expression.
    """

    # TODO: handle bias correction:
    # https://en.wikipedia.org/wiki/Weighted_arithmetic_mean#Weighted_sample_variance
    # https://numpy.org/doc/stable/reference/generated/numpy.cov.html

    if w is None and center_around is None:
        return into_expr(x).var(ddof=0)

    center_around = center_around or x.pipe(mean, w=w)

    return (x - center_around).pow(2).pipe(mean, w=w)


def cor(x: IntoExpr, y: IntoExpr, *, w: Optional[IntoExpr] = None) -> pl.Expr:
    """
    Computes the (optionally weighted) Pearson correlation coefficient.

    See: https://en.wikipedia.org/wiki/Pearson_correlation_coefficient#Weighted_correlation_coefficient
    """
    x = into_expr(x)

    numerator = x.pipe(cov, y, w=w)
    denominator = (x.pipe(var, w=w) * into_expr(y).pipe(var, w=w)).sqrt()

    return (numerator / denominator).alias("cor")
