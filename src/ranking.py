import concurrent.futures
import os
from itertools import combinations, product
from multiprocessing import cpu_count
from pathlib import Path

from DockM8.consensus_methods import *
from scipy.stats import spearmanr
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from tqdm import tqdm

from utilities import split_list


def all_combinations(
        docking_programs: list,
        rescoring_programs: list,
):
    """
        Generate all possible combinations of docking and rescoring programs
        Args:
                docking_programs: list of docking programs
                rescoring_programs: list of rescoring programs
        Return:
                list of all possible combinations of docking and rescoring programs
    """

    all_comb_scoring_function = [
        list(
            combinations(
                rescoring_programs,
                r)) for r in range(
            1,
            len(rescoring_programs) +
            1)]
    all_comb_scoring_function = [
        item for sublist in all_comb_scoring_function for item in sublist]

    all_comb_docking_program = [list(combinations(docking_programs, r))
                                for r in range(1, len(docking_programs) + 1)]

    all_comb_docking_program = [
        item for sublist in all_comb_docking_program for item in sublist]
    return list(product(all_comb_docking_program, all_comb_scoring_function))


def runtime_cost_calculation(
        docking_tools,
        scoring_functions,
        num_poses=10) -> float:
    """
    Calculate the runtime cost of a pipeline
    Args:
            docking_tools: list of docking tools
            scoring_functions: list of scoring functions
            num_poses: number of poses
    Return:
            runtime cost of the pipeline
    """

    runtime_per_tool = {
        'gnina': 105.80,
        'plants': 6.82,
        'smina': 99.90,
        'flexx': 3.33,
        'localdiffdock': 407.50,
        'CNNscore': 0.31,
        'CNNaffinity': 0.31,
        'smina_affinity': 0.31,
        'ad4': 0.28,
        'LinF9': 0.24,
        'RTMScore': 0.41,
        'Vinardo': 0.29,
        'SCORCH': 4.63,
        'HYDE': 2.0,
        'CHEMPLP': 0.121,
        'rfscore_v1': 0.682,
        'rfscore_v2': 0.687,
        'rfscore_v3': 0.69,
        'vina_hydrophobic': 0.69,
        'vina_intra_hydrophobic': 0.69
    }
    runtime_docking_tools = [runtime_per_tool[tool.lower()]
                             for tool in docking_tools]
    runtime_scoring_tools = [runtime_per_tool[tool]
                             for tool in scoring_functions]
    return sum(runtime_docking_tools) + \
        (sum(runtime_scoring_tools) * num_poses * len(docking_tools))


def enrichment_factor_calc(df, percent=1, activity_class='activity_class'):
    """
    Calculate the enrichment factor of a dataframe
    Args:
            df: dataframe
            percent: percentage of the dataframe
            activity_class: column name of the activity class
    Return:
            enrichment factor
    """

    included_rows = round(percent / 100 * df.shape[0])
    df_copy = df.copy().apply(pd.to_numeric, errors='ignore')
    actives_in_percent = np.sum(df_copy.head(included_rows)[activity_class])

    quotient = actives_in_percent / included_rows

    divisor = len(df_copy[df_copy.activity_class == 1]) / len(df_copy)

    ef = quotient / divisor

    # if ef > 1:
    #         ef = 1

    return ef


def process_combination(
    splitted_comb,
    df_rescored,
    ranking_method,
    output_path,
    index,
    weights
):
    """
    Rank poses with different docking and scoring tools using different consensus methods

    Args:
            splitted_comb: list of splitted combinations
            df_rescored: dataframe with rescored poses
            df_scores: dataframe with ground truth scores with two columns: ID and score
            ranking_method: ranking method
            output_path: path to output folder
            index: index of the splitted_comb

    Return:
            Write the results of every ranking method to a csv file
    """
    corr_dict = {
        'docking_tool': [],
        'scoring_function': [],
        'spearman_correlation': [],
        'cost_per_pipeline': [],
        'enrichment_factor': []
    }
    ranking_methods_dict = {
        'best_ECR': method1_ECR_best,
        'ECR_average': method2_ECR_average,
        'average_ECR': method3_avg_ECR,
        'rank_by_rank': method4_RbR,
        'rank_by_vote': method5_RbV,
        'best_Zscore': method6_Zscore_best,
        'average_Zscore': method7_Zscore_avg,
        'rbv_all': method8_RbV_all,
        'weighted_ECR': method9_weighted_ECR_best,
    }
    # print(f'Running {ranking_method}...')

    df = df_rescored.copy()
    ranking_method_name = ranking_methods_dict[ranking_method].__name__

    for _, comb in enumerate(tqdm(splitted_comb, total=len(splitted_comb))):

        filtered_df = df[df['docking_tool'].isin(list(comb[0]))]
        try:

            if ranking_method == 'weighted_ECR':
                #        print(f"Using weighted ECR with weights: {weights}")
                df_rank = method9_weighted_ECR_best(
                    filtered_df.copy(),
                    'allposes',
                    list(comb[1]),
                    id_column='ID',
                    mapped_weights=weights
                )
            else:
                df_rank = ranking_methods_dict[ranking_method](
                    filtered_df.copy(), 0.05, list(comb[1]), id_column='ID')

        except (RuntimeError, TypeError, NameError, pd.errors.MergeError, KeyError) as err:
            print(f"Error occurred: {err}")
        try:
            df_rank_copy = df_rank.copy().apply(
                pd.to_numeric,
                errors='ignore'
            ).dropna().merge(
                filtered_df[['ID', 'true_value', 'activity_class', 'id']],
                on='ID',
                how='inner'
            )

            df_unique_sorted = df_rank_copy.sort_values(
                by=df_rank_copy.columns[1],
                ascending=False
            ).drop_duplicates(
                subset=['id']
            )
        except (RuntimeError, TypeError, NameError, pd.errors.MergeError, KeyError) as err:
            print(f"Error occurred: {err}")

        try:
            spearman_corr, _ = spearmanr(
                df_rank_copy.loc[:, ranking_method_name], df_rank_copy['true_value'])

            ef = enrichment_factor_calc(
                df_unique_sorted,
                percent=10,
                activity_class='activity_class')
            cost = runtime_cost_calculation(
                docking_tools=list(
                    comb[0]), scoring_functions=list(
                    comb[1]), num_poses=10)

        except (RuntimeError, TypeError, NameError, pd.errors.MergeError, KeyError) as err:
            print("df_filter", filtered_df)
            print(df_unique_sorted)
            print("df)rank copy", df_rank_copy.head(1))
            print(f"Error occurred in calculations: {err}")

        corr_dict['docking_tool'].append(list(comb[0]))
        corr_dict['scoring_function'].append(list(comb[1]))
        corr_dict['spearman_correlation'].append(spearman_corr)
        corr_dict['cost_per_pipeline'].append(cost)
        corr_dict['enrichment_factor'].append(ef)

    corr_dict['ranking_method'] = [ranking_method_name] * \
        len(corr_dict['docking_tool'])
    corr_df = pd.DataFrame.from_dict(corr_dict, orient='index').transpose()
    corr_df.to_csv(
        str(output_path / f'{ranking_method}_parallel_{index}.csv'), index=False)


def poses_ranking(
        ranking_methods: list,
        df_rescored: pd.DataFrame,
        output_path: Path,
        validation="general",
        ncpus=cpu_count() - 2,
        weights=None
):
    """
    Rank poses with different docking and scoring tools using different consensus methods

    Args:
            ranking_methods: list of ranking methods
            df_rescored: dataframe with rescored poses
            output_path: path to output folder
            validation: validation set
            ncpus: number of cpus to use
    Return:
            Write the results of every ranking method to a csv file
    """
    # df_rescored[['ID', 'docking_tool', 'pose']] = df_rescored['ID'].str.split('_', expand=True)
    df_rescored = df_rescored[df_rescored['docking_tool'].notna()]
    docking_programs = list(df_rescored['docking_tool'].unique())

    print(f'Number of docking programs: {
          len(docking_programs)}, {docking_programs}')
    rescoring_methods = [col for col in df_rescored.columns if col not in [
        'ID', 'id', 'pose', 'docking_tool', 'true_value', 'activity_class']]
    print(f'Number of rescoring methods: {
          len(rescoring_methods)}, {rescoring_methods}')
    all_comb = all_combinations(docking_programs, rescoring_methods)
    print(
        f'Number of possible combinations for every ranking method: {
            len(all_comb)}' f'\n With total combinations : {
            len(all_comb) *
            len(ranking_methods)}')
    splitted_comb = split_list(all_comb, ncpus)

    if validation:

        corr_file_path = output_path / f'correlations_{validation}'
        corr_file_path.mkdir(parents=True, exist_ok=True)

    for ranking_method in (ranking_methods):

        if os.path.exists(str(corr_file_path / 'all_ranked.csv')):
            print(f'All poses are ranked with all consensus methods ..')
            break
        if os.path.exists(
                str(corr_file_path / f'{ranking_method}_concat.csv')):

            print(f'File {ranking_method} exists. Skipping...')
            continue

        print(f'Parallelizing {ranking_method}...')

        with concurrent.futures.ProcessPoolExecutor(max_workers=ncpus) as executor:
            _ = [
                executor.submit(
                    process_combination,
                    comb,
                    df_rescored,
                    ranking_method,
                    corr_file_path,
                    i,
                    weights) for i,
                comb in enumerate(splitted_comb)]

        df = pd.concat([pd.read_csv(str(
            corr_file_path / f'{ranking_method}_parallel_{i}.csv')) for i in range(ncpus)])
        df.to_csv(str(corr_file_path /
                      f'{ranking_method}_concat.csv'), index=False)
        # delete the splitted files
        for i in range(ncpus):
            os.remove(str(corr_file_path /
                          f'{ranking_method}_parallel_{i}.csv'))
        print(f'Finished {ranking_method}...')
    # concatenate all the results
    if not os.path.exists(str(corr_file_path / 'all_ranked.csv')):
        df = pd.concat(
            [
                pd.read_csv(
                    str(
                        corr_file_path /
                        f'{ranking_method}_concat.csv')) for ranking_method in ranking_methods]).apply(
            pd.to_numeric,
            errors='ignore')
        [os.remove(str(corr_file_path / f'{ranking_method}_concat.csv'))
         for ranking_method in ranking_methods]
        df.sort_values(
            by='spearman_correlation',
            ascending=False,
            inplace=True)
        df.to_csv(str(corr_file_path / 'all_ranked.csv'), index=False)


def norm_scores(
    df,
    inversed_sf_cols=[
        'smina_affinity',
        'ad4',
        'LinF9',
        'Vinardo',
        'CHEMPLP',
        'HYDE',
        'vina_hydrophobic',
        'vina_intra_hydrophobic'],
) -> pd.DataFrame:
    """
    Normalize a DataFrame that has an ID in the first column numerical values of scoring functions.
    Certain specified columns will have their scaling inversed ('the lower the better').

    Args:
            df (pd.DataFrame): The DataFrame to normalize.
            inversed_sf_cols (List): List of scoring functions that in ascending order (The lower the better)

    Returns:
            pd.DataFrame: The normalized DataFrame.
    """
    df_copy = df.copy()

    numeric_cols = df_copy.select_dtypes(include=[np.number]).columns

    for col in inversed_sf_cols:
        if col not in df.columns:
            continue
        if col == 'HYDE':
            hyde_ranks = df_copy.loc[:, col].rank(ascending=False)
            df_copy.loc[:, col] = df_copy.loc[:, col].apply(
                lambda x: min(x, 10000)) + hyde_ranks
        min_value = df_copy.loc[:, col].min()

        # Shift data to be positive and invert the scale
        df.loc[:, col] = df_copy.loc[:, col] - min_value + 1
        max_value = df_copy.loc[:, col].max()
        df_copy.loc[:, col] = max_value + 1 - df[col]

    scaler = RobustScaler(quantile_range=(5, 95))
    # scaler = MinMaxScaler()

    scaled_numeric = scaler.fit_transform(df_copy[numeric_cols])
    scaled_numeric_df = pd.DataFrame(
        scaled_numeric,
        columns=numeric_cols,
        index=df_copy.index)
    df_copy.update(scaled_numeric_df)

    return df_copy
