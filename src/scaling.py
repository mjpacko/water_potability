import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
import matplotlib.pyplot as plt

def normalize(train: pd.DataFrame, test: pd.DataFrame, Function=MinMaxScaler) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Normalize the data using the given scaler function.

    Parameters:
        train (pd.DataFrame): Training data.
        test (pd.DataFrame): Testing data.
        Function (function): Scaler function to use.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: Normalized training and testing data.
    """

    # Fit the normalizer on the training data
    normalizer = Function()
    normalizer.fit(train)

    # Transform the training and testing data
    train_norm = normalizer.transform(train)
    test_norm = normalizer.transform(test)

    # Convert the normalized data back to a DataFrame
    train_norm = pd.DataFrame(train_norm, columns=train.columns)
    test_norm = pd.DataFrame(test_norm, columns=test.columns)

    # Return the normalized data
    return train_norm, test_norm


def plot_correlation_heatmap(df: pd.DataFrame):
    """
    Plots a heatmap of the correlation matrix for the given DataFrame.

    Parameters:
        df (pd.DataFrame): The input DataFrame.

    Returns:
        None: Displays the heatmap.
    """
    
    # Compute the correlation matrix
    corr = np.abs(df.corr())

    # Create a mask for the upper triangle
    mask = np.zeros_like(corr, dtype=bool)
    mask[np.triu_indices_from(mask)] = True

    # Create a heatmap
    f, ax = plt.subplots(figsize=(10, 10))
    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    sns.heatmap(
        corr,
        mask=mask,
        vmax=1,
        square=True,
        linewidths=0.5,
        cbar_kws={ "shrink": 0.5 },
        annot = corr
    )

    # Show the plot
    plt.show()