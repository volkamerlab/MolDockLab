import pandas as pd
import numpy as np


def method1_ECR_best(df, weight, selected_columns, id_column):
    '''
    A method that calculates the ECR (Exponential Consensus Ranking) score for each ID in the rescored dataframe and returns the ID for the pose with the best ECR rank.
    '''

    sigma = weight * len(df)

    for col in selected_columns:
        df.loc[:, col] = df.loc[:, col].rank(method='min', ascending=False)
        df.loc[:, col] = (np.exp(-(df.loc[:, col] / sigma)) / sigma)    
    
    df[f'method1_ECR_best'] = df[selected_columns].sum(axis=1, numeric_only=True)

    # Aggregate rows using best ECR per ID
    df2 = df.sort_values(
        f'method1_ECR_best',
        ascending=False).drop_duplicates(
        [id_column])
    return df2[[id_column, f'method1_ECR_best']]


def method2_ECR_average(df, weight, selected_columns, id_column):
    '''
    A method that calculates the ECR (Exponential Consensus Ranking) score for each ID in the rescored dataframe and returns the ID along with the average ECR rank accross the clustered poses.
    '''


    sigma = weight * len(df)

    # df[selected_columns] = df[selected_columns].rank(method='min', ascending=True)
    for col in selected_columns:
        df.loc[:, col] = df.loc[:, col].rank(method='min', ascending=False)
        df.loc[:, col] = (np.exp(-(df.loc[:, col] / sigma)) / sigma)   

    df[f'method2_ECR_average'] = df[selected_columns].sum(axis=1, numeric_only=True)

    # Aggregate rows using mean ECR per ID
    df2 = df.groupby(id_column, as_index=False).mean(numeric_only=True)
    return df2[[id_column, f'method2_ECR_average']]


def method3_avg_ECR(df, weight, selected_columns, id_column):
    '''
    A method that first calculates the average ranks for each pose in filtered dataframe (by ID) then calculates the ECR (Exponential Consensus Ranking) for the averaged ranks.
    '''
    # Aggregate rows using mean rank per ID
    df = df.groupby(id_column, as_index=False).mean(numeric_only=True)
    # df[selected_columns] = df[selected_columns].rank(
    #     method='average', ascending=1)
    

    sigma = weight * len(df)
    
    # df[selected_columns] = df[selected_columns].rank(
    #     method='average', ascending=1)
    # df[selected_columns] = df[selected_columns].rank(method='min', ascending=True)
    for col in selected_columns:
        # df.loc[:, col] = df.loc[:, col].rank(method='min', ascending=False)
        df.loc[:, col] = df.loc[:, col].rank(method='min', ascending=False)
        df.loc[:, col] = (np.exp(-(df.loc[:, col] / sigma)) / sigma) 

    df[f'method3_avg_ECR'] = df[selected_columns].sum(axis=1, numeric_only=True)
    return df[[id_column, f'method3_avg_ECR']]


def method4_RbR(df, weight, selected_columns, id_column):
    '''
    A method that calculates the Rank by Rank consensus.
    '''
    df = df.groupby(id_column, as_index=False).mean(numeric_only=True).round(2)

    for col in selected_columns:
        # Molecules that has higher affinities assigned to higher ranks to positively correlate with the log affinity
        df.loc[:, col] = df.loc[:, col].rank(method='min', ascending=True)

    df.loc[:, 'method4_RbR'] = df.loc[:, selected_columns].mean(axis=1)
    # df = df.sort_values(
    #     f'method4_RbR',
    #     ascending=True).drop_duplicates(
    #     [id_column])

    return df[[id_column, f'method4_RbR']]


def method5_RbV(df, weight, selected_columns, id_column):
    '''
    A method that calculates the Rank by Vote consensus.
    '''
    df['vote'] = 0
    top_percent_cutoff = max(int(len(df) * weight), 1)
    df = df.groupby(id_column, as_index=False).mean(numeric_only=True).round(2)

    for col in selected_columns:
    # Molecules that has higher affinities assigned to higher ranks to positively correlate with the log affinity
        top_values_index = df[col].nlargest(top_percent_cutoff).index
        df.loc[:, col] = df.loc[:, col].rank(method='min', ascending=True)
        df.loc[top_values_index, 'vote'] += 1

    df.loc[:, 'method5_RbV'] = df.loc[:, selected_columns].mean(axis=1) - df['vote']

    return df[[id_column, f'method5_RbV']]


def method6_Zscore_best(df, weight, selected_columns, id_column):
    '''
    Calculates the Z-score consensus scores for each row in the given DataFrame,
    and aggregates rows by selecting the pose with the best Z-score for each ID.
    '''



    df[selected_columns] = df[selected_columns].apply(
        pd.to_numeric, errors='coerce')
    z_scores = (df[selected_columns] - df[selected_columns].mean()
                ) / df[selected_columns].std()
    df[f'method6_Zscore_best'] = z_scores.mean(axis=1)
    # Aggregate rows using best Z-score per ID
    df = df.sort_values(
        f'method6_Zscore_best',
        ascending=False).drop_duplicates(
        [id_column])
    # df.set_index(id_column)
    return df[[id_column, f'method6_Zscore_best']]


def method7_Zscore_avg(df, weight, selected_columns, id_column):
    '''
    Calculates the Z-score consensus scores for each row in the given DataFrame,
    and aggregates rows by averaging the Z-score for each ID.
    '''


    df[selected_columns] = df[selected_columns].apply(
        pd.to_numeric, errors='coerce')
    z_scores = (df[selected_columns] - df[selected_columns].mean()
                ) / df[selected_columns].std()
    df[f'method7_Zscore_avg'] = z_scores.mean(axis=1)
    # Aggregate rows using avg Z-score per ID
    df = df.groupby(id_column, as_index=False).mean(numeric_only=True)
    return df[[id_column, f'method7_Zscore_avg']]

def method8_RbN(df, weight, selected_columns, id_column):
    '''
    A method that calculates the Rank by Rank consensus.
    '''
    df = df.groupby(id_column, as_index=False).mean(numeric_only=True).round(2)

    # for col in selected_columns:
        # Determine the top 5% values in each column
        # df.loc[:, col] = df.loc[:, col].rank(method='min', ascending=False)
    df[f'method8_RbN'] = df[selected_columns].mean(axis=1)
    # df = df.sort_values(
    #     f'method4_RbR',
    #     ascending=True).drop_duplicates(
    #     [id_column])

    return df[[id_column, f'method8_RbN']]


def method9_weighted_ECR_best(df, selected_columns, id_column, mapped_weights):
    '''
    A method that calculates the ECR (Exponential Consensus Ranking) score for each ID in the rescored dataframe and returns the ID for the pose with the best ECR rank.
    Args:
        df (DataFrame): The dataframe containing the rescored poses
        weight (str): The metric used for clustering the poses
        selected_columns (list): The list of columns to be used for calculating the ECR
        id_column (str): The column containing the ID of the poses
        mapped_weights (dict): The dictionary containing the weights for each column
    Returns:
        DataFrame: The dataframe containing the ID and the best ECR score for each pose
    '''

    docking_tools_sigma = np.mean([mapped_weights[d] for d in df.docking_tool.unique()])
    try:
        for col in selected_columns:
            sigma = mapped_weights[col] * docking_tools_sigma * len(df)

            if sigma == 0:
                sigma = 0.01 * len(df)
                continue
            df.loc[:, col] = df.loc[:, col].rank(method='min', ascending=False)
            df.loc[:, col] = (np.exp(-(df.loc[:, col] / sigma)) / sigma)
        df[f'method9_weighted_ECR_best'] = df[selected_columns].sum(axis=1, numeric_only=True)
    except KeyError:
        print('The weights are not mapped for the selected columns')
        return None
    # Aggregate rows using best ECR per ID
    df2 = df.sort_values(
        f'method9_weighted_ECR_best',
        ascending=False).drop_duplicates(
        [id_column])

    return df2[[id_column, f'method9_weighted_ECR_best']]