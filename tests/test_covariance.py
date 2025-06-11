import numpy as np
import polars as pl

from polars_utils.covariance import covariance_matrix


def test_covariance_matrix():
    iris = pl.read_csv(
        "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv"
    ).select(pl.col(float))

    missing_pandas = iris.to_pandas().cov(ddof=0).to_numpy()
    missing_mine = iris.pipe(covariance_matrix)

    assert np.allclose(missing_pandas, missing_mine), (
        "unweighted cov matrix not equal to pandas"
    )

    weighted_np = np.cov(
        iris.to_numpy(),
        rowvar=False,
        aweights=iris.get_column("sepal_length").to_numpy(),
        ddof=0,
    )

    weighted_mine = iris.pipe(covariance_matrix, w="sepal_length")

    assert np.allclose(weighted_np, weighted_mine), (
        "weighted cov matrix not equal to numpy"
    )
