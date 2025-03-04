from typing import Optional

import polars as pl
import polars._typing as pt

from polars_utils import into_expr, normalize

Weight = Optional[pt.IntoExprColumn]


def into_normalized_weight(w: Weight) -> pl.Expr:
    return into_expr(w or pl.lit(1)).pipe(normalize)
