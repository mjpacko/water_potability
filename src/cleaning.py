import pandas as pd

def lowercase_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Lowercase the column names of a dataframe.

    Parameters:
        df (pd.DataFrame): The dataframe to lowercase the column names.

    Returns:
        pd.DataFrame: The dataframe with the column names lowercased.
    """

    # Create a copy of the dataframe
    df_copy = df.copy()
    
    # Lowercase the column names
    df_copy.columns = df_copy.columns.str.lower()
    
    # Return the dataframe
    return df_copy

def fill_missing_values(df: pd.DataFrame, columns=[], agg="mean") -> pd.DataFrame:
    """
    Fill missing values in the specified columns with the specified aggregation.

    Parameters:
        df (pd.DataFrame): The dataframe to fill the missing values.
        columns (list or str): The columns to fill the missing values.
        agg (str): The aggregation to use to fill the missing values.

    Returns:
        pd.DataFrame: The dataframe with the missing values filled.
    """

    # Check if agg is a valid value
    if agg not in ["mean", "median", "mode"]:
        raise ValueError("agg must be either 'mean', 'median', or 'mode'")

    # Create a copy of the dataframe
    df_copy = df.copy()

    # If columns is a string, convert it to a list
    if isinstance(columns, str): columns = [columns]

    # Apply the aggregation to the columns
    for column in columns:
        if agg == "mean":
            df_copy[column] = df_copy[column].fillna(df_copy[column].mean())
        elif agg == "median":
            df_copy[column] = df_copy[column].fillna(df_copy[column].median())
        elif agg == "mode":
            df_copy[column] = df_copy[column].fillna(df_copy[column].mode()[0])

    # Return the dataframe
    return df_copy

def drop_outliers(df: pd.DataFrame, columns=[]) -> pd.DataFrame:
    """
    Drop the outliers in the specified columns.

    Parameters:
        df (pd.DataFrame): The dataframe to drop the outliers.
        columns (list or str): The columns to drop the outliers.

    Returns:
        pd.DataFrame: The dataframe without the outliers.
    """

    # Create a copy of the dataframe
    df_copy = df.copy()

    # If columns is a string, convert it to a list
    if isinstance(columns, str): columns = [columns]

    for column in columns:
        # Calculate the iqr
        q1 = df_copy[column].quantile(0.25)
        q3 = df_copy[column].quantile(0.75)
        iqr = q3 - q1

        # Drop the outliers
        df_copy = df_copy[(df_copy[column] >= q1 - 1.5 * iqr) & (df_copy[column] <= q3 + 1.5 * iqr)]

    # Return the dataframe
    return df_copy