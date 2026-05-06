import numpy as np
import pandas as pd

'''''
For more information on these methods, please refer to the following paper:

Lacour A, Ibrahim H, Volkamer A, Hirsch AKH. DockM8: An All-in-One Open-Source 
Platform for Consensus Virtual Screening in Drug Design. 
ChemRxiv. 2024; doi:10.26434/chemrxiv-2024-17k46.
'''''

def exponential_consensus_ranking(
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
        df.loc[:, col] = (np.exp(-(df.loc[:, col] / sigma)) / sigma) * 1000   
    df[f'exponential_consensus_ranking'] = df[selected_scores].sum(axis=1, numeric_only=True)
    df2 = df.sort_values(
        f'exponential_consensus_ranking',
        ascending=False).drop_duplicates(
        [id_column])
    return df2[[id_column, f'exponential_consensus_ranking']]




def rank_by_rank(
        df: pd.DataFrame, 
        weight: float, 
        selected_scores: list, 
        id_column: str
        ) -> pd.DataFrame:
    """
    A method that calculates the Rank by Rank consensus score. The higher the score the better the pose.

    Args:
        df (DataFrame): The dataframe containing the rescored poses
        weight (float): The inclusion threshold for the poses
        selected_scores (list): The list of columns to be used for calculating the Rank by Rank
        id_column (str): The column containing the ID of the poses
    Returns:
        DataFrame: The dataframe containing the ID and the best rank-by-rank score for each pose
    """
    


    for col in selected_scores:
        df.loc[:, col] = df.loc[:, col].rank(method='min', ascending=True)
    # df['best_pose'] = df[selected_scores].idxmax(axis=1)
    # df = df.groupby(id_column, as_index=False).mean(numeric_only=True).round(5)
    df.loc[:, 'rank_by_rank'] = df.loc[:, selected_scores].mean(axis=1)
    df = df.sort_values(
        'rank_by_rank',
        ascending=False).drop_duplicates(
        [id_column])
    return df[[id_column, f'rank_by_rank']]


def Zscore(
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
        DataFrame: The dataframe containing the ID and the Z scores for each pose
    """
    df[selected_scores] = df[selected_scores].apply(
        pd.to_numeric, errors='coerce')
    z_scores = (df[selected_scores] - df[selected_scores].mean()
                ) / df[selected_scores].std()
    df[f'Zscore'] = z_scores.mean(axis=1)
    # Aggregate rows using best Z-score per ID
    df = df.sort_values(
        f'Zscore',
        ascending=False).drop_duplicates(
        [id_column])
    # df.set_index(id_column)
    return df[[id_column, f'Zscore']]
