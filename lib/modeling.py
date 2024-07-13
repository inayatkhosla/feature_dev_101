"""
Utiltity functions for modeling
"""
from typing import List

import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score


def train_test_split(
    data: pd.DataFrame, ts_col: str, train_start: str, test_start: str, test_end: str
):
    """
    Split time-series data into training and testing sets based on specified time periods.

    This function takes a DataFrame containing time-series data, and splits it into
    training and testing sets based on the provided time periods. The data is indexed
    using the values from the specified timestamp column.

    Parameters:
    data (pd.DataFrame): The DataFrame containing the time-series data.
    ts_col (str): The name of the column representing timestamps.
    train_start (str): The start of the training period in the format 'YYYY-MM-DD'.
    test_start (str): The start of the testing period in the format 'YYYY-MM-DD'.
    test_end (str): The end of the testing period in the format 'YYYY-MM-DD'.

    Returns:
    tuple: A tuple containing two DataFrames - the training set and the testing set.

    Example:
    train, test = train_test_split(data, 'timestamp', '2023-01-01', '2023-02-01', '2023-03-01')
    """
    data = data.set_index(pd.DatetimeIndex(data[ts_col]))
    train = data.loc[(data.index >= train_start) & (data.index < test_start)]
    test = data.loc[(data.index >= test_start) & (data.index < test_end)]
    for i in train, test:
        i.reset_index(drop=True, inplace=True)
    return train, test


def eval_classification(actuals: List[int], preds: List[int]) -> dict:
    """
    Evaluate the performance of a classification model using various metrics.

    This function calculates the F1 score, precision, and recall of a
    classification model's predictions compared to the actual labels.

    Parameters:
    actuals (array-like): The true labels or ground truth.
    preds (array-like): The predicted labels from a classification model.

    Returns:
    dict: A dictionary containing f1, precision, and recall scores.

    Example:
    eval_results = eval_classification(actual_labels, predicted_labels)
    """
    f1 = round(f1_score(actuals, preds), 2)
    precision = round(precision_score(actuals, preds), 2)
    recall = round(recall_score(actuals, preds), 2)

    return {"f1": f1, "precision": precision, "recall": recall}
