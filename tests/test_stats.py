import polars as pl
import numpy as np

from polars_utils import stats

x, y, w, n, c = (pl.col(c) for c in ("x", "y", "w", "n", "c"))


def create_test_data(seed=10845, n=10_000) -> pl.DataFrame:
    rng = np.random.default_rng(seed)

    data = dict(
        x=rng.normal(size=n),
        y=rng.normal(size=n),
        w=rng.uniform(0, 1, size=n),
        n=rng.integers(0, 1000, size=n),
        c=[100] * n,
    )

    return pl.DataFrame(data)


def test_mean():
    df = create_test_data()

    assert np.isclose(
        df.select(x.pipe(stats.mean))[0, 0],
        np.average(df["x"]),
    ), "Unweighted mean differs from Numpy"

    assert np.isclose(
        df.select(x.pipe(stats.mean, w=w))[0, 0],
        np.average(df["x"], weights=df["w"]),
    ), "Weighted mean differs from Numpy"

    assert np.isclose(
        df.select(x.pipe(stats.mean, w="c"))[0, 0],
        df.select(x.pipe(stats.mean))[0, 0],
    ), "Constant weights should equal no weights"


def test_var():
    df = create_test_data()

    assert np.isclose(
        df.select(x.pipe(stats.var))[0, 0],
        df["x"].var(ddof=0),  # type: ignore
    ), "Unweighted variance differs from Numpy"

    assert np.isclose(
        df.select(x.pipe(stats.var, w=w))[0, 0],
        np.cov(df["x"], aweights=df["w"], ddof=0),
    ), "Weighted variance differs from Numpy"


def test_cov():
    df = create_test_data()

    assert np.isclose(
        df.select(x.pipe(stats.var, w=w))[0, 0],
        df.select(x.pipe(stats.cov, x, w=w))[0, 0],
    ), "Variance does not equal covariance with self"

    assert np.isclose(
        df.select(x.pipe(stats.cov, y))[0, 0],
        df["x"].var(ddof=0),  # type: ignore
    ), "Unweighted covariance differs from Numpy"

    assert np.isclose(
        df.select(x.pipe(stats.var, w=w))[0, 0],
        np.cov(df["x"], aweights=df["w"], ddof=0),
    ), "Weighted variance differs from Numpy"


def test_cor():
    df = create_test_data()

    assert np.isclose(df.select(x.pipe(stats.cor, x))[0, 0], 1.0), (
        "Self correlation is not 1"
    )

    # TODO: check against numpy
    # assert np.isclose(
    #     df.select(x.pipe(stats.var, w=w))[0, 0],
    #     np.var(df["x"], weights=df["w"]),
    # )
