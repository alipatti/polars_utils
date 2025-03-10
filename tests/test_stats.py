import polars as pl
import numpy as np

from polars_utils import stats

x, y, w, n, c = (pl.col(c) for c in ("x", "y", "w", "n", "c"))
x_null, y_null, w_null = (pl.col(c + "_null") for c in ("x", "y", "w"))

RNG = np.random.default_rng(10498091)


def replace_with_nulls(col: pl.Expr, null_proportion=0.05):
    return (
        pl.when(
            pl.int_range(pl.len())
            .shuffle(RNG.integers(10_000))
            .lt(pl.len() * null_proportion)
        )
        .then(None)
        .otherwise(col)
    )


def create_test_data(n=10_000) -> pl.DataFrame:
    x = RNG.normal(size=n)

    data = dict(
        x=x,
        y=2 * x + RNG.normal(size=n),
        w=RNG.uniform(0, 1, size=n),
        n=RNG.integers(0, 1000, size=n),
        c=[100] * n,
    )

    return pl.DataFrame(data).with_columns(
        pl.col("x", "y", "w").pipe(replace_with_nulls).name.suffix("_null"),
    )


def test_mean():
    df = create_test_data()

    assert np.allclose(
        df.select(x.pipe(stats.mean))[0, 0],
        np.average(df["x"]),
    ), "Unweighted mean differs from Numpy"

    assert np.allclose(
        df.select(x.pipe(stats.mean, w=w))[0, 0],
        np.average(df["x"], weights=df["w"]),
    ), "Weighted mean differs from Numpy"

    assert np.allclose(
        df.select(x.pipe(stats.mean, w="c"))[0, 0],
        df.select(x.pipe(stats.mean))[0, 0],
    ), "Constant weights should equal no weights"


def test_var():
    df = create_test_data()

    assert np.allclose(
        df.select(x.pipe(stats.var))[0, 0],
        df["x"].var(ddof=0),  # type: ignore
    ), "Unweighted variance differs from Numpy"

    assert np.allclose(
        df.select(x.pipe(stats.var, w=w))[0, 0],
        np.cov(df["x"], aweights=df["w"], ddof=0),
    ), "Weighted variance differs from Numpy"


def test_cov():
    df = create_test_data()

    assert np.allclose(
        df.select(x.pipe(stats.var, w=w))[0, 0],
        df.select(x.pipe(stats.cov, x, w=w))[0, 0],
    ), "Variance does not equal covariance with self"

    assert np.allclose(
        df.select(x.pipe(stats.cov, y))[0, 0],
        np.cov(df["x"], df["y"], ddof=0)[1, 0],
    ), "Unweighted covariance differs from Numpy"

    assert np.allclose(
        df.select(x.pipe(stats.cov, y, w=w))[0, 0],
        np.cov(df["x"], df["y"], aweights=df["w"], ddof=0)[1, 0],
    ), "Weighted variance differs from Numpy"


def test_cor():
    df = create_test_data()

    assert np.allclose(df.select(x.pipe(stats.cor, x))[0, 0], 1.0), (
        "Self correlation is not 1"
    )

    assert np.allclose(
        df.select(x.pipe(stats.cor, y))[0, 0],
        np.corrcoef(df["x"], df["y"])[1, 0],
    ), "Correlation differs from Numpy"


def test_nulls():
    df = create_test_data()

    assert np.allclose(
        df.select(x_null.pipe(stats.mean))[0, 0],
        df.drop_nulls().select(x.pipe(stats.mean))[0, 0],
    ), "Nulls not correctly masked in unweighted mean"

    assert np.allclose(
        df.select(x_null.pipe(stats.mean, w=w))[0, 0],
        df.drop_nulls().select(x.pipe(stats.mean, w=w))[0, 0],
    ), "Nulls not correctly masked in weighted mean"

    assert np.allclose(
        df.select(x_null.pipe(stats.var, w=w))[0, 0],
        df.drop_nulls().select(x.pipe(stats.var, w=w))[0, 0],
    ), "Nulls not correctly masked in weighted variance"

    assert np.allclose(
        df.select(x_null.pipe(stats.cov, y_null, w=w))[0, 0],
        df.drop_nulls().select(x.pipe(stats.cov, y, w=w))[0, 0],
    ), "Nulls not correctly masked in weighted covariance"
