import polars as pl
import polars._typing as pt


def into_expr(w: pt.IntoExprColumn) -> pl.Expr:
    """
    Converts a string (column name) or Polars series into an expression.
    """
    if isinstance(w, str):
        return pl.col(w)

    elif isinstance(w, pl.Series):
        return pl.lit(w)

    elif isinstance(w, pl.Expr):
        return w

    else:
        raise ValueError


def normalize(x: pl.Expr) -> pl.Expr:
    """
    Normalizes an expression so that it sums to one.
    """
    return x / x.sum()
