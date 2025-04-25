from typing import Any, Iterable, Literal, Optional
import polars as pl
import numpy as np
from rpy2.robjects import numpy2ri, Formula, packages, default_converter, NULL
from functools import cache

from tqdm import tqdm

@cache
def get_gam():
    return packages.importr("mgcv").gam


def smooth(
    y: pl.Expr | str,
    *xs: pl.Expr | str,
    weights: pl.Expr | str | None = None,
    progress=False,
    **kwargs,
) -> pl.Expr:
    if progress:
        bar = tqdm(desc = "smoothing...")
        bar.clear()

    def f(cols):
        if progress:
            bar.update() # type: ignore

        return _smooth(
            y=cols[0],
            xs=cols[1:-1],
            weights=cols[-1],
            **kwargs,
        )
    
    return pl.map_groups(
        [y, *xs, weights or pl.lit(None, dtype=pl.Null)],
        f,
        return_dtype=pl.Float64,
        returns_scalar=False,
    ).alias(y if isinstance(y, str) else y.meta.output_name())


def _smooth(
    y: pl.Series,
    xs: Iterable[pl.Series],
    *,
    weights: Optional[pl.Series] = None,
    k: Optional[tuple[int, ...]] = None,
    family: Literal["gaussian", "binomial"] | Any = "gaussian",
) -> pl.Series:
    formula = (
        "y ~ te("
        + f"{','.join(f'x{i}' for i, _ in enumerate(xs))}"  # covariates
        + (f", k=c({','.join(map(str, k))})" if k else "")  # knots
        + ")"
    )

    with (numpy2ri.converter + default_converter).context():
        f = Formula(formula)

        f.environment["y"] = y.to_numpy()

        for i, x in enumerate(xs):
            f.environment[f"x{i}"] = x.to_numpy()

        gam = get_gam()
        model = gam(
            f,
            family=family,
            weights=NULL
            if weights is None or (weights.len() == 1 and weights.item() is None)
            else weights.to_numpy(),
        )
        fitted_values = np.asarray(model.rx2("fitted.values"))

    return pl.Series(y.name, fitted_values, nan_to_null=True, dtype=pl.Float64)
