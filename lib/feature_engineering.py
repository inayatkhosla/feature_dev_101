"""
Utility functions for generating features
"""
from typing import List
from itertools import product

import numpy as np
import pandas as pd


# TODO: Make pure function
def assign_rank(
    dataset: pd.DataFrame,
    cols: List[str],
    partition_col: str = None,
    method: str = "average",
    ascending: bool = True,
    suffix: str = "rank",
) -> pd.DataFrame:
    """
    Assign percentile-based ranks to specified columns in a DataFrame.

    This function calculates percentile-based ranks for the specified columns within
    the DataFrame. Ranks can be assigned either globally or partitioned by a specific
    column. The resulting DataFrame includes new columns containing the calculated ranks.

    Args:
        dataset (pd.DataFrame): The DataFrame containing the data to be ranked.
        cols (List[str]): A list of column names for which ranks should be calculated.
        partition_col (str, optional): The column by which to partition the ranking.
                                       If provided, ranks will be calculated within
                                       each partition. Default is None (global ranking).
        method (str, optional): The method used to assign ranks.
                               - "average": Average rank for tied values (default).
                               - "min": Minimum rank for tied values.
                               - "max": Maximum rank for tied values.
                               - "first": Assign ranks in order of appearance.
        ascending (bool, optional): If True (default), ranks are assigned in ascending
                                   order; if False, ranks are assigned in descending
                                   order.
        suffix (str, optional): The suffix to be added to the new rank columns' names.
                                Default is "rank".

    Returns:
        pd.DataFrame: A DataFrame with added columns containing the calculated ranks.

    Example:
        ranked_data = assign_rank(
            input_data,
            cols=["score", "revenue"],
            partition_col="category",
            method="average",
            ascending=False,
            suffix="percentile"
        )
    """
    for col in cols:
        if partition_col:
            dataset[f"{col}_{suffix}"] = round(
                dataset.groupby(partition_col)[col].rank(
                    method=method, pct=True, ascending=ascending
                )
                * 100,
                1,
            )
            continue
        dataset[f"{col}_{suffix}"] = round(
            dataset[col].rank(method="average", pct=True, ascending=ascending) * 100, 1
        )
    return dataset


def get_merchant_metrics_all_periods(
    merchant_metrics: pd.DataFrame,
    merchant_col: str = "merchant",
    period_col: str = "period",
) -> pd.DataFrame:
    """
    Expand merchant metrics to include all possible combinations of merchants and periods. Missing combinations are filled with NaN values.

    Args:
        merchant_metrics (pd.DataFrame): DataFrame containing merchant metrics data.
        merchant_col (str, optional): Name of the column containing merchant identifiers.
                                      Default is "merchant".
        period_col (str, optional): Name of the column containing period identifiers.
                                    Default is "period".

    Returns:
        pd.DataFrame: A DataFrame with all possible combinations of merchants and
                      periods, along with corresponding metrics. Missing values are
                      filled with NaN.

    Example:
        all_periods_metrics = get_merchant_metrics_all_periods(
            merchant_metrics_df,
            merchant_col="merchant_id",
            period_col="period_id",
        )
    """
    merchants = merchant_metrics[merchant_col].unique()
    periods = merchant_metrics[period_col].unique()
    combinations = list(product(merchants, periods))

    all_periods = pd.DataFrame(combinations, columns=[merchant_col, period_col])
    mm_all_periods = all_periods.merge(
        merchant_metrics, on=[merchant_col, period_col], how="left"
    )

    return mm_all_periods


# TODO: Make pure function
def add_trailing_aggregates(
    dataset: pd.DataFrame,
    cols: List[str],
    window: int,
    min_periods: int,
    prefix: str = "t",
    merchant_col: str = "merchant",
    period_col: str = "period",
):
    """
    Add trailing aggregates to specified columns based on a rolling window.

    Args:
        dataset (pd.DataFrame): The DataFrame containing the data to be aggregated.
        cols (List[str]): A list of column names for which trailing aggregates will be
                          calculated.
        window (int): The size of the rolling window for aggregation.
        min_periods (int): The minimum number of non-NaN values required within the
                           window to compute an aggregate.
        prefix (str, optional): The prefix to be added to the new aggregate columns'
                               names. Default is "t".
        merchant_col (str, optional): The name of the column containing merchant
                                      information. Default is "merchant".
        period_col (str, optional): The name of the column containing period
                                    information. Default is "period".

    Returns:
        pd.DataFrame: A DataFrame with added columns containing the trailing aggregates.

    Notes:
        - Trailing aggregates are calculated separately for each merchant, based on the
          provided merchant column.
        - The function supports calculating both sum and mean aggregates based on the
          column names. If a column name contains "volume" or "count", a sum aggregate
          will be calculated; otherwise, a mean aggregate will be calculated.

    Example:
        updated_data = add_trailing_aggregates(
            original_data,
            cols=["revenue", "volume"],
            window=7,
            min_periods=3,
            prefix="trailing",
            merchant_col="merchant_id",
            period_col="date",
        )
    """
    mg = dataset.groupby(merchant_col)
    for col in cols:
        if "volume" in col or "count" in col:
            dataset[f"{prefix}_{col}"] = (
                mg[col]
                .rolling(window=window, min_periods=min_periods)
                .sum()
                .reset_index(level=0, drop=True)
            )
            continue
        dataset[f"{prefix}_{col}"] = (
            mg[col]
            .rolling(window=window, min_periods=min_periods)
            .mean()
            .reset_index(level=0, drop=True)
        )
    return dataset


def get_change_prev_periods(
    merchant_metrics: pd.DataFrame,
    cols: List[str],
    shift_periods: int,
    prefix: str = "p",
    merchant_col: str = "merchant",
    period_col: str = "period",
):
    """
    Calculate changes in specified columns relative to previous periods. The changes can be expressed as absolute differences and proportional changes.

    Args:
        merchant_metrics (pd.DataFrame): DataFrame containing merchant metrics data.
        cols (List[str]): A list of column names for which changes will be calculated.
        shift_periods (int): The number of periods to shift for comparison.
        prefix (str, optional): The prefix to be added to the new columns' names.
                               Default is "p".
        merchant_col (str, optional): The name of the column containing merchant
                                      identifiers. Default is "merchant".
        period_col (str, optional): The name of the column containing period
                                    identifiers. Default is "period".

    Returns:
        pd.DataFrame: A DataFrame with added columns containing changes relative to
                      previous periods.

    Notes:
        - Changes are calculated separately for each merchant.
        - The function calculates absolute differences, proportional changes, and
          shifts the specified columns.
        - Proportional changes are calculated as (new_value - old_value) / old_value.
        - The resulting DataFrame may contain NaN values after division.

    Example:
        changes_data = get_change_prev_periods(
            merchant_metrics_df,
            cols=["revenue", "volume"],
            shift_periods=1,
            prefix="change",
            merchant_col="merchant_id",
            period_col="period_id",
        )
    """
    df = merchant_metrics[[merchant_col, period_col] + cols]
    df = df.sort_values([merchant_col, period_col])
    for col in cols:
        df[f"{prefix}_{shift_periods}_{col}"] = df.groupby(merchant_col)[col].shift(
            shift_periods
        )
        df[f"{prefix}_{shift_periods}_absdiff_{col}"] = (
            df[f"{col}"] - df[f"{prefix}_{shift_periods}_{col}"]
        )
        df[f"{prefix}_{shift_periods}_propch_{col}"] = round(
            df[f"{prefix}_{shift_periods}_absdiff_{col}"]
            / df[f"{prefix}_{shift_periods}_{col}"],
            1,
        )
        df.drop([col, f"{prefix}_{shift_periods}_{col}"], axis=1, inplace=True)
    df = df.replace([np.inf, -np.inf], np.nan)
    return df


# TODO: Make pure function
def get_change_peak(
    merchant_metrics: pd.DataFrame,
    cols: List[str],
    prefix: str = "peak",
    merchant_col: str = "merchant",
    period_col: str = "period",
):
    """
    Calculate changes in specified columns relative to peak values. The changes can be expressed as absolute differences
    and proportional changes.

    Args:
        merchant_metrics (pd.DataFrame): DataFrame containing merchant metrics data.
        cols (List[str]): A list of column names for which changes will be calculated.
        prefix (str, optional): The prefix to be added to the new columns' names.
                               Default is "peak".
        merchant_col (str, optional): The name of the column containing merchant
                                      identifiers. Default is "merchant".
        period_col (str, optional): The name of the column containing period
                                    identifiers. Default is "period".

    Returns:
        pd.DataFrame: A DataFrame with added columns containing changes relative to peak
                      values.

    Notes:
        - Changes are calculated separately for each merchant.
        - The function calculates absolute differences, proportional changes, and
          tracks peak values for the specified columns.
        - Proportional changes are calculated as (new_value / peak_value).
        - The resulting DataFrame may contain NaN values after division.

    Example:
        peak_changes_data = get_change_peak(
            merchant_metrics_df,
            cols=["revenue", "volume"],
            prefix="change_peak",
            merchant_col="merchant_id",
            period_col="period_id",
        )
    """

    df = merchant_metrics[[merchant_col, period_col] + cols]
    df = df.sort_values([merchant_col, period_col])
    for col in cols:
        df[f"{prefix}_{col}"] = df.groupby(merchant_col)[col].cummax()
        df[f"{prefix}_absdiff_{col}"] = df[col] - df[f"{prefix}_{col}"]
        df[f"{prefix}_prop_{col}"] = round(df[col] / df[f"{prefix}_{col}"], 1)
        df.drop([col, f"{prefix}_{col}"], axis=1, inplace=True)
    df = df.replace([np.inf, -np.inf], np.nan)
    return df


def get_max_historical_gap_bw_tx(
    dataset: pd.DataFrame,
    merchant_col: str = "merchant",
    period_col: str = "period",
    time_diff_col: str = "time_diff_days",
):
    """
    Calculate the maximum historical time gap between transactions for each merchant.

    Args:
        dataset (pd.DataFrame): DataFrame containing transaction data and relevant columns.
        merchant_col (str, optional): The name of the column containing merchant
                                      identifiers. Default is "merchant".
        period_col (str, optional): The name of the column containing period
                                    identifiers. Default is "period".
        time_diff_col (str, optional): The name of the column containing time
                                       differences between transactions. Default is
                                       "time_diff_days".

    Returns:
        pd.DataFrame: A DataFrame containing the maximum historical time gap between
                      transactions for each merchant and period.

    Example:
        max_gap_data = get_max_historical_gap_bw_tx(
            transaction_data_df,
            merchant_col="merchant_id",
            period_col="period_id",
            time_diff_col="time_diff_days",
        )
    """
    dataset["max_hist_time_diff"] = dataset.groupby(merchant_col)[
        "time_diff_days"
    ].cummax()
    df = (
        dataset.groupby([merchant_col, period_col])["max_hist_time_diff"]
        .max()
        .reset_index()
    )
    return df


def get_yoy_change(
    merchant_metrics: pd.DataFrame,
    prediction_frequency: str,
    merchant_col: str = "merchant",
    period_col: str = "period",
    volume_col: str = "total_tx_volume",
    suffix: str = "yoy",
) -> pd.DataFrame:
    """
    Calculate Year-over-Year (YoY) changes in transaction volume for each merchant. The result is a DataFrame containing the absolute and percentage differences in transaction volume between the current period and the same period from the previous year.

    Args:
        merchant_metrics (pd.DataFrame): DataFrame containing merchant metrics data.
        prediction_frequency (str): The frequency at which predictions are made
                                   (e.g., 'M' for monthly, 'Q' for quarterly).
        merchant_col (str, optional): The name of the column containing merchant
                                      identifiers. Default is "merchant".
        period_col (str, optional): The name of the column containing period
                                    information. Default is "period".
        volume_col (str, optional): The name of the column containing transaction
                                   volume data. Default is "total_tx_volume".
        suffix (str, optional): The suffix to be added to new columns' names.
                               Default is "yoy".

    Returns:
        pd.DataFrame: A DataFrame containing YoY changes in transaction volume for
                      each merchant and period.

    Example:
        yoy_changes_data = get_yoy_change(
            merchant_metrics_df,
            prediction_frequency="M",
            merchant_col="merchant_id",
            period_col="period_date",
            volume_col="transaction_volume",
            suffix="yoy_changes",
        )
    """
    df = merchant_metrics[[merchant_col, period_col, volume_col]]
    df[period_col] = df[period_col].dt.start_time
    df["prev_year_period"] = df[period_col] - pd.DateOffset(years=1)
    df = df.merge(
        df,
        how="left",
        left_on=[merchant_col, "prev_year_period"],
        right_on=[merchant_col, period_col],
        suffixes=("", "_prev"),
    )
    df[f"{suffix}_abs_diff"] = df[volume_col] - df[f"{volume_col}_prev"]
    df[f"{suffix}_pct_diff"] = round(
        (df[f"{suffix}_abs_diff"] / df[f"{volume_col}_prev"]) * 100, 1
    )
    df = df[[merchant_col, period_col, f"{suffix}_abs_diff", f"{suffix}_pct_diff"]]
    df[period_col] = df[period_col].dt.to_period(prediction_frequency)
    return df
