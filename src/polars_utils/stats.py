import polars as pl
import polars._typing as pt

from polars_utils import into_expr, match_name
from polars_utils.weights import Weight, into_normalized_weight


def mean(x: pl.Expr, *, w: Weight = None) -> pl.Expr:
    """
    Computes the (weighted) mean of an expression.
    """
    return x.dot(into_normalized_weight(w))


def cov(x: pl.Expr, other: pt.IntoExprColumn, *, w: Weight = None) -> pl.Expr:
    """
    Computes the (weighted) covaraince of an expression with another expression.
    """
    w = into_normalized_weight(w)
    y = into_expr(other)

    inside_sum = w * (x - x.dot(w)) * (y - y.dot(w))

    return inside_sum.sum().alias("cov")


def var(x: pl.Expr, *, w: Weight = None):
    """
    Computes the (weighted) variance of an expression.
    """
    w = into_normalized_weight(w)

    return (x - x.dot(w)).pow(2).dot(w).pipe(match_name, x)


def cor(x: pl.Expr, y: pt.IntoExprColumn, *, w: Weight = None) -> pl.Expr:
    """
    Computes the (optionally weighted) Pearson correlation coefficient.

    See: https://en.wikipedia.org/wiki/Pearson_correlation_coefficient#Weighted_correlation_coefficient
    """
    numerator = x.pipe(cov, y, w=w)
    denominator = (x.pipe(var, w=w) * into_expr(y).pipe(var, w=w)).sqrt()

    return (numerator / denominator).alias("cor")
