import pandas as pd
import numpy as np


def method1_ECR_best(
        df: pd.DataFrame, 
        weight: float, 
        selected_scores: list, 
        id_column: str
        ) -> pd.DataFrame:
    """
    A method that calculates the ECR score for each ID in the rescored dataframe 
    and returns the ID for the pose with the best ECR rank.
    Args:
        df (DataFrame): The dataframe containing the rescored poses
        weight (float): The inclusion threshold for the poses
        selected_scores (list): The list of columns to be used for calculating the ECR best
        id_column (str): The column containing the ID of the poses
    Returns:
        DataFrame: The dataframe containing the ID and the best ECR score for each pose
    """
    sigma = weight * len(df)
    for col in selected_scores:
        df.loc[:, col] = df.loc[:, col].rank(method='min', ascending=False)
        df.loc[:, col] = (np.exp(-(df.loc[:, col] / sigma)) / sigma)    
    df[f'method1_ECR_best'] = df[selected_scores].sum(axis=1, numeric_only=True)
    df2 = df.sort_values(
        f'method1_ECR_best',
        ascending=False).drop_duplicates(
        [id_column])
    return df2[[id_column, f'method1_ECR_best']]


def method2_ECR_average(
        df: pd.DataFrame, 
        weight: float, 
        selected_scores: list, 
        id_column: str
        ) -> pd.DataFrame:
    """
    A method that calculates the ECR score for each ID in the rescored dataframe 
    and returns the ID with ECR average rank.
    Args:
        df (DataFrame): The dataframe containing the rescored poses
        weight (float): The inclusion threshold for the poses
        selected_scores (list): The list of columns to be used for calculating the ECR average
        id_column (str): The column containing the ID of the poses
    Returns:
        DataFrame: The dataframe containing the ID and the best ECR score for each pose
    """
    sigma = weight * len(df)
    for col in selected_scores:
        df.loc[:, col] = df.loc[:, col].rank(method='min', ascending=False)
        df.loc[:, col] = (np.exp(-(df.loc[:, col] / sigma)) / sigma)   

    df[f'method2_ECR_average'] = df[selected_scores].sum(axis=1, numeric_only=True)

    # Aggregate rows using mean ECR per ID
    df2 = df.groupby(id_column, as_index=False).mean(numeric_only=True)
    return df2[[id_column, f'method2_ECR_average']]


def method3_avg_ECR(
        df: pd.DataFrame, 
        weight: float, 
        selected_scores: list, 
        id_column: str
        ) -> pd.DataFrame:
    """
    A method that calculates the ECR score for each ID after taking 
    the average of ID's different poses and returns the ID with the ECR rank.
    Args:
        df (DataFrame): The dataframe containing the rescored poses
        weight (float): The inclusion threshold for the poses
        selected_scores (list): The list of columns to be used for calculating the average ECR
        id_column (str): The column containing the ID of the poses
    Returns:
        DataFrame: The dataframe containing the ID and the best ECR score for each pose
    """
    df = df.groupby(id_column, as_index=False).mean(numeric_only=True)
    sigma = weight * len(df)
    for col in selected_scores:
        df.loc[:, col] = df.loc[:, col].rank(method='min', ascending=False)
        df.loc[:, col] = (np.exp(-(df.loc[:, col] / sigma)) / sigma) 

    df[f'method3_avg_ECR'] = df[selected_scores].sum(axis=1, numeric_only=True)
    return df[[id_column, f'method3_avg_ECR']]


def method4_RbR(
        df: pd.DataFrame, 
        weight: float, 
        selected_scores: list, 
        id_column: str
        ) -> pd.DataFrame:
    """
    A method that calculates the Rank by Rank consensus score.

    Args:
        df (DataFrame): The dataframe containing the rescored poses
        weight (float): The inclusion threshold for the poses
        selected_scores (list): The list of columns to be used for calculating the Rank by Rank
        id_column (str): The column containing the ID of the poses
    Returns:
        DataFrame: The dataframe containing the ID and the best ECR score for each pose
    """
    df = df.groupby(id_column, as_index=False).mean(numeric_only=True).round(2)

    for col in selected_scores:
        df.loc[:, col] = df.loc[:, col].rank(method='min', ascending=True)

    df.loc[:, 'method4_RbR'] = df.loc[:, selected_scores].mean(axis=1)
    return df[[id_column, f'method4_RbR']]


def method5_RbV(
        df: pd.DataFrame, 
        weight: float, 
        selected_scores: list, 
        id_column: str
        ) -> pd.DataFrame:
    """
    A method that calculates the Rank by Vote consensus score.

    Args:
        df (DataFrame): The dataframe containing the rescored poses
        weight (float): The inclusion threshold for the poses
        selected_scores (list): The list of columns to be used for calculating the Rank by Vote
        id_column (str): The column containing the ID of the poses
    Returns:
        DataFrame: The dataframe containing the ID and the best ECR score for each pose
    """
    df['vote'] = 0
    top_percent_cutoff = max(int(len(df) * weight), 1)
    df = df.groupby(id_column, as_index=False).mean(numeric_only=True).round(2)

    for col in selected_scores:
        top_values_index = df[col].nlargest(top_percent_cutoff).index
        df.loc[:, col] = df.loc[:, col].rank(method='min', ascending=True)
        df.loc[top_values_index, 'vote'] += 1

    df.loc[:, 'method5_RbV'] = df.loc[:, selected_scores].mean(axis=1) - df['vote']

    return df[[id_column, f'method5_RbV']]


def method6_Zscore_best(
        df: pd.DataFrame, 
        weight: float, 
        selected_scores: list, 
        id_column: str
        ) -> pd.DataFrame:
    """
    Calculates the Z-score consensus scores for each row in the given DataFrame,
    and aggregates rows by selecting the pose with the best Z-score for each ID.
    Args:
        df (DataFrame): The dataframe containing the rescored poses
        weight (float): The inclusion threshold for the poses
        selected_scores (list): The list of columns to be used for calculating the Z-score best
        id_column (str): The column containing the ID of the poses
    Returns:
        DataFrame: The dataframe containing the ID and the best ECR score for each pose
    """
    df[selected_scores] = df[selected_scores].apply(
        pd.to_numeric, errors='coerce')
    z_scores = (df[selected_scores] - df[selected_scores].mean()
                ) / df[selected_scores].std()
    df[f'method6_Zscore_best'] = z_scores.mean(axis=1)
    # Aggregate rows using best Z-score per ID
    df = df.sort_values(
        f'method6_Zscore_best',
        ascending=False).drop_duplicates(
        [id_column])
    # df.set_index(id_column)
    return df[[id_column, f'method6_Zscore_best']]


def method7_Zscore_avg(
        df: pd.DataFrame, 
        weight: float, 
        selected_scores: list, 
        id_column: str
        ) -> pd.DataFrame:
    """
    Calculates the Z-score consensus scores for each row in the given DataFrame,
    and aggregates rows by averaging the Z-score for each ID.
    Args:
        df (DataFrame): The dataframe containing the rescored poses
        weight (float): The inclusion threshold for the poses
        selected_scores (list): The list of columns to be used for calculating the Z-score average
        id_column (str): The column containing the ID of the poses
    Returns:
        DataFrame: The dataframe containing the ID and the best ECR score for each pose
    """
    df[selected_scores] = df[selected_scores].apply(
        pd.to_numeric, errors='coerce')
    z_scores = (df[selected_scores] - df[selected_scores].mean()
                ) / df[selected_scores].std()
    df[f'method7_Zscore_avg'] = z_scores.mean(axis=1)
    # Aggregate rows using avg Z-score per ID
    df = df.groupby(id_column, as_index=False).mean(numeric_only=True)
    return df[[id_column, f'method7_Zscore_avg']]

def method8_RbN(
        df: pd.DataFrame, 
        weight: float, 
        selected_scores: list, 
        id_column: str
        ) -> pd.DataFrame:
    """
    A method that calculates the Rank by Number consensus.
    Args:
        df (DataFrame): The dataframe containing the rescored poses
        weight (float): The inclusion threshold for the poses
        selected_scores (list): The list of columns to be used for Rank by Number
        id_column (str): The column containing the ID of the poses
    Returns:
        DataFrame: The dataframe containing the ID and the best ECR score for each pose
    """
    df = df.groupby(id_column, as_index=False).mean(numeric_only=True).round(2)
    df[f'method8_RbN'] = df[selected_scores].mean(axis=1)

    return df[[id_column, f'method8_RbN']]


def method9_weighted_ECR_best(
        df: pd.DataFrame, 
        mapped_weights:dict, 
        selected_scores: list, 
        id_column:str
        ) -> pd.DataFrame:
    """
    A method that calculates the ECR (Exponential Consensus Ranking) score for each ID 
    in the rescored dataframe and returns the ID for the pose with the best ECR rank.
    Args:
        df (DataFrame): The dataframe containing the rescored poses
        selected_scores (list): The list of columns to be used for calculating the weighted ECR
        id_column (str): The column containing the ID of the poses
        mapped_weights (dict): The dictionary containing the weights for each column
    Returns:
        DataFrame: The dataframe containing the ID and the best ECR score for each pose
    """
    docking_tools_sigma = np.mean([mapped_weights[d] for d in df.docking_tool.unique()])
    try:
        for col in selected_scores:
            sigma = mapped_weights[col] * docking_tools_sigma * len(df)

            if sigma == 0:
                sigma = 0.01 * len(df)
                continue
            df.loc[:, col] = df.loc[:, col].rank(method='min', ascending=False)
            df.loc[:, col] = (np.exp(-(df.loc[:, col] / sigma)) / sigma)
        df[f'method9_weighted_ECR_best'] = df[selected_scores].sum(axis=1, numeric_only=True)
    except KeyError:
        print('The weights are not mapped for the selected columns')
        return None
    df2 = df.sort_values(
        f'method9_weighted_ECR_best',
        ascending=False).drop_duplicates(
        [id_column])

    return df2[[id_column, f'method9_weighted_ECR_best']]