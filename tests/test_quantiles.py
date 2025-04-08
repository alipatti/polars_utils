import polars as pl

from polars_utils.quantile import df_quantile, df_xtile


def test_discrete_quantiles_unweighted():
    ks = 4, 5, 20
    n = 1_000

    df = pl.select(i=pl.int_range(n)).with_columns(
        pl.col("i").mod(k).alias(f"cat_{k}") for k in ks
    )

    ranked = df.pipe(
        df_quantile,
        [f"cat_{k}" for k in ks],
        ties="average",
    )

    for k in ks:
        correct_rank = pl.col(f"cat_{k}") / k + 1 / (k * 2)

        is_correct = (
            ranked.group_by(f"cat_{k}")
            .agg(pl.col(f"cat_{k}_rank").sub(correct_rank).lt(10e-10).all())
            .get_column(f"cat_{k}_rank")
            .all()
        )

        assert is_correct, f"ranks incorrect for {k} discrete categories"

    assert (
        df.pipe(
            df_xtile,
            ["cat_20"],
            n=10,
        )
        .select(
            bins_correct=pl.col("cat_20")
            .floordiv(2)
            .add(1)
            .cast(pl.String)
            .cast(pl.Categorical)
            .eq(pl.col("cat_20_bin"))
        )
        .get_column("bins_correct")
        .all()
    ), "bins incorrect"


def test_discrete_quantiles_weighted():
    x = [1, 2, 3, 4]
    w = [2, 3, 4, 1]
    correct_ranks = [0.1, 0.35, 0.7, 0.95]

    df = pl.DataFrame(dict(x=x, w=w, correct_ranks=correct_ranks)).pipe(
        df_quantile, ["x"], w="w"
    )

    assert (
        df.select(pl.col("x_rank") == pl.col("correct_ranks"))
        .get_column("x_rank")
        .all()
    )
