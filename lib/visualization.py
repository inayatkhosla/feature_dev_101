"""
Utility functions for visualization
"""
from typing import List

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve


def format_plot(title: str, xlabel: str, ylabel: str, legend: bool = False) -> None:
    """
    Format and display a matplotlib plot with specified title, x and y labels, and optional legend.

    Parameters:
        title (str): Title for the plot.
        xlabel (str): Label for the x-axis.
        ylabel (str): Label for the y-axis.
        legend (bool, optional): Whether to show a legend. Default is False.

    Returns:
        None

    Example:
    format_plot("Example Plot", "X-axis", "Y-axis", legend=True)

    """
    plt.ticklabel_format(style="plain", axis="y")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if legend:
        plt.legend(loc="upper right", ncol=1, frameon=True, fancybox=True, fontsize=10)
    sns.despine()
    plt.show()


# TODO: Make legend labels more customizable
def compare_dist(
    dataset: pd.DataFrame,
    col: str,
    split_col: str,
    title: str,
    xlabel: str,
    ylabel: str,
    cutoff: int = None,
    log_scale: bool = False,
) -> None:
    """
    Compare the distributions of a numeric column based on a binary split.

    Args:
        dataset (pd.DataFrame): The DataFrame containing the data for comparison.
        col (str): The name of the numeric column whose distribution will be compared.
        split_col (str): The name of the binary split column (0 or 1) for comparison.
        title (str): The title of the plot.
        xlabel (str): The label for the x-axis.
        ylabel (str): The label for the y-axis.
        cutoff (int, optional): If provided, values exceeding the absolute value of
                                this cutoff will be excluded from the analysis.
        log_scale (bool, optional): If True, the y-axis will be displayed in a log
                                   scale. Default is False.

    Returns:
        None

    Example:
        compare_dist(
            data,
            col="merchant_age",
            split_col="churn",
            title="Merchant Age by Churn",
            xlabel="",
            ylabel="Number of Merchants",
            cutoff=1000,
            log_scale=False,
        )
    """
    if cutoff:
        dataset = dataset[np.abs(dataset[col]) < cutoff]
    sns.histplot(
        data=dataset[dataset[split_col] == 1][col],
        log_scale=log_scale,
        alpha=0.5,
        color="darkblue",
        label=f"{split_col}",
    )
    sns.histplot(
        data=dataset[dataset[split_col] == 0][col],
        log_scale=log_scale,
        alpha=0.5,
        color="lightblue",
        label=f"No {split_col}",
    )
    format_plot(title, xlabel, ylabel, legend=True)


def plot_precision_recall_curve(y_val: List[int], predictions: List[int]) -> None:
    """
    Plot the precision-recall curve based on the provided true labels and prediction scores.

    Parameters:
    y_val (List[int]): True labels (ground truth) for the validation set.
    predictions (List[int]): Predicted scores or probabilities for the positive class.

    Returns:
    None

    plot_precision_recall_curve(y_val, predictions)
    """
    precision, recall, _ = precision_recall_curve(y_val, predictions)
    plt.figure(figsize=(8, 6))
    plt.step(recall, precision, color="b", alpha=0.7, where="post")
    plt.fill_between(recall, precision, step="post", alpha=0.3, color="b")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    format_plot("Precision Recall Curve", "Recall", "Precision")
