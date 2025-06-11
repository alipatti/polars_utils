from itertools import product
from typing import Literal, Optional

import polars as pl

import numpy as np

from polars_utils import IntoExpr
from polars_utils.stats import cov


def covariance_matrix(
    df: pl.DataFrame,
    w: Optional[IntoExpr] = None,
    columns=pl.all(),
    nulls: Literal["drop_obs", "pairwise_complete", "error"] = "drop_obs",
) -> np.ndarray:
    columns = pl.selectors.expand_selector(df, columns, strict=False)

    if nulls == "drop_obs":
        df = df.drop_nulls(columns)
    elif nulls == "error":
        assert df.select(
            pl.all_horizontal(pl.col(c).is_null().any() for c in columns)
        ).item(), "nulls in data frame"
    elif nulls == "pairwise_complete":
        pass
    else:
        raise ValueError

    return (
        df.select(
            cov(a, b, w=w).alias(f"{a}|{b}") for a, b in product(columns, columns)
        )
        .to_numpy()
        .reshape((len(columns), len(columns)))
    )


