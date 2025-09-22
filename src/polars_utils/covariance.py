from itertools import combinations_with_replacement, product
from typing import Collection, Literal, Optional

import polars as pl

import numpy as np

from polars_utils import IntoExpr
from polars_utils.stats import cov


def covariance_matrix(
    df: pl.DataFrame,
    w: Optional[IntoExpr] = None,
    columns: IntoExpr | Collection[str] = pl.all(),
    nulls: Literal["drop_obs", "pairwise_complete", "error"] = "drop_obs",
) -> np.ndarray:
    if not isinstance(columns, Collection):
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

    covariances = df.select(
        cov(a, b, w=w).alias(f"Cov({a},{b})")
        for a, b in combinations_with_replacement(columns, 2)
    ).to_numpy()

    n = len(columns)
    sigma = np.empty((n, n))

    i, j = np.triu_indices(n)
    sigma[i, j] = covariances  # upper tri
    sigma[j, i] = covariances  # lower tri

    return sigma
