import polars as pl
import polars._typing as pt

from polars_utils import into_expr, normalize

Weight = pt.IntoExprColumn


def into_normalized_weight(w: Weight) -> pl.Expr:
    w = into_expr(w)

    if w.meta.is_literal(allow_aliasing=True):
        raise ValueError("Literal weights not allowed")

    return w.pipe(normalize)
