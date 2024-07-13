"""
Utility functions to generate labels
"""
import numpy as np
import pandas as pd


def generate_labels(
    data: pd.DataFrame,
    merchant_col: str,
    time_col: str,
    pred_freq: str,
    churn_cutoff_days: int,
    lookforward_period_days: int,
) -> pd.DataFrame():
    """
    Generate labels for churn prediction based on transaction data.

    This function takes transaction data along with various parameters and generates
    labels for churn prediction. Churn is determined based on the extended absence of user activity during a forward looking time period.

    Args:
        data (pd.DataFrame): The transaction data containing merchant, timestamp,
        and other relevant columns.
        merchant_col (str): The name of the column containing merchant information.
        time_col (str): The name of the column containing timestamp information.
        pred_freq (str): The frequency at which predictions will be made, e.g., 'M' for
        monthly, 'D' for daily.
        churn_cutoff_days (int): The number of days since last observed activity to mark a merchant as churned.
        lookforward_period_days (int): The number of days to look forward for the absence of merchant activity.

    Returns:
        pd.DataFrame: A DataFrame containing generated labels including merchant,
                      period, activity status, maximum timestamp before prediction
                      period, maximum timestamp within the lookforward period,
                      churn status, and other relevant columns.

    Notes:
        - The function calculates user activity based on the provided lookforward
          period and churn cutoff days.
        - Churn status is determined based on activity within the prediction period
          and the intra-period churn indicator.

    Example:
        labels_df = generate_labels(
            transaction_data,
            merchant_col="merchant_id",
            time_col="timestamp",
            pred_freq="M",
            churn_cutoff_days=60,
            lookforward_period_days=90
        )
    """

    data["period"] = data["time"].dt.to_period(pred_freq)
    labels = data.groupby([merchant_col, "period"])[time_col].max().reset_index()
    labels["lfp_start"] = (labels["period"].dt.end_time + pd.DateOffset(1)).dt.date
    labels["lfp_end"] = pd.to_datetime(
        labels["lfp_start"] + pd.DateOffset(lookforward_period_days)
    )
    labels.rename(columns={time_col: "max_ts_bp"}, inplace=True)

    merged_tx_df = labels.merge(
        data[[merchant_col, time_col]], on=merchant_col, how="left"
    )
    merged_tx_df["activity"] = (
        merged_tx_df[time_col].between(
            merged_tx_df["lfp_start"], merged_tx_df["lfp_end"], inclusive="left"
        )
    ).astype(int)
    activity_df = (
        merged_tx_df.groupby([merchant_col, "period"])["activity"].max().reset_index()
    )

    max_lfp_timestamp_df = (
        merged_tx_df[merged_tx_df["activity"] == 1]
        .groupby([merchant_col, "period"])[time_col]
        .max()
        .reset_index()
    )
    max_lfp_timestamp_df.rename(columns={time_col: "max_ts_lfp"}, inplace=True)
    activity_df = activity_df.merge(
        max_lfp_timestamp_df, on=[merchant_col, "period"], how="left"
    )
    labels = labels.merge(activity_df, on=[merchant_col, "period"])

    labels["churn_cutoff"] = labels["lfp_end"] - pd.DateOffset(churn_cutoff_days)
    labels["intra_period_churn"] = np.where(
        labels["max_ts_lfp"] < labels["churn_cutoff"], 1, 0
    )

    labels["churn"] = np.where(
        (labels["activity"] == 0) | (labels["intra_period_churn"] == 1), 1, 0
    )
    return labels
