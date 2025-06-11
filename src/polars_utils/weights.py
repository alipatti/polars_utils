from typing import Optional
import polars as pl

from polars_utils import IntoExpr, into_expr, normalize


def into_normalized_weight(w: IntoExpr, null_mask: Optional[pl.Expr] = None) -> pl.Expr:
    w = into_expr(w)

    if w.meta.is_literal(allow_aliasing=True):
        raise ValueError("Literal weights not allowed")

    return w.pipe(normalize, null_mask)
