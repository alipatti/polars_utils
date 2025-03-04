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


def match_name(x: pl.Expr, col_to_match: pl.Expr, *, fallback="literal") -> pl.Expr:
    """
    Changes the name of an expression to match the name of another expression.
    """
    return x.alias(
        col_to_match.meta.output_name(raise_if_undetermined=False) or fallback
    )
