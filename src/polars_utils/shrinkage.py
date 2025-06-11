from typing import Optional
import polars as pl


from polars_utils import IntoExpr, into_expr
from polars_utils.stats import mean


def reliability(
    estimates: pl.Expr,
    *,
    variances: IntoExpr,
    w: Optional[IntoExpr] = None,
    mu: Optional[pl.Expr] = None,
):
    variances = into_expr(variances)

    # point around which we compute variance
    mu = estimates.pipe(mean, w=w) if mu is None else mu

    # calculate reliability
    signal_variance = ((estimates - mu).pow(2) - variances).pipe(mean, w=w)

    return (signal_variance / (variances + signal_variance)).alias("reliability")


def shrink(
    estimates: pl.Expr,
    *,
    variances: Optional[IntoExpr] = None,
    w: Optional[IntoExpr] = None,
    mu: Optional[pl.Expr] = None,  # mean to shrink to
    rho: Optional[pl.Expr] = None,  # reliability
) -> pl.Expr:
    """
    https://libgen.li/adsa82270d055f6ee991539ac0533036e0dO9ZBZS4G
    """

    # point we shrink to
    if mu is None:
        mu = estimates.pipe(mean, w=w)

    # reliability
    if rho is None:
        if variances is None:
            raise ValueError("Must pass either variances or reliabilities.")

        rho = reliability(estimates, variances=variances, w=w, mu=mu)

    # shrink towards mean based on reliability
    return estimates * rho + mu * (1 - rho)
