from src.consensus_rank import *
from itertools import combinations, product
from multiprocessing import cpu_count
import concurrent.futures
from pathlib import Path
import os
from scipy.stats import spearmanr
from sklearn.preprocessing import RobustScaler

# Calculate Spearman correlation

def split_list(input_list, num_splits):
    '''
    Split a list into n parts.
    Args:
        input_list: list to be split
        num_splits: number of splits
    Returns: 
        list of splitted lists
    '''
    avg_size = len(input_list) // num_splits
    remain = len(input_list) % num_splits
    partitions = []
    i = 0
    for _ in range(num_splits):
        partition_size = avg_size + 1 if remain > 0 else avg_size
        partitions.append(input_list[i:i+partition_size])
        i += partition_size
        remain -= 1
        # print(len(partitions[-1]))
    return partitions
def all_combinations(docking_programs: list, rescoring_programs: list):
    """
    Generate all combinations of docking methods and scoring functions.
    Args:
        docking_programs: list of docking methods
        rescoring_programs: list of rescoring programs
    Returns: 
               list of tuples with all combinations of docking methods and scoring functions
    """
    all_comb_scoring_function = [item for r in range(1, len(rescoring_programs) + 1) 
                                 for item in combinations(sorted(rescoring_programs), r)]
    all_comb_docking_program = [item for r in range(1, len(docking_programs) + 1) 
                                 for item in combinations(sorted(docking_programs), r)]

    return list(product(all_comb_docking_program, all_comb_scoring_function))
def runtime_cost_calculation(docking_tools, scoring_functions, num_poses=10) -> float:
        runtime_per_tool = {
                'gnina': 105.80,
                'plants': 6.82,
                'smina': 99.90,
                'flexx': 3.33,
                'localdiffdock': 407.50,
                'diffdock': 407.50,
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
                'vina_intra_hydrophobic': 0.69,
                'KORP-PL': 0.2,
                'ConvexPLR': 0.2
        }
        runtime_docking_tools = [runtime_per_tool[tool.lower()] for tool in docking_tools]
        runtime_scoring_tools = [runtime_per_tool[tool] for tool in scoring_functions]
        return sum(runtime_docking_tools) + (sum(runtime_scoring_tools) * num_poses)


def enrichment_factor_calc(df, percent=1, activity_class='activity_class'):
    included_rows = round(percent / 100 * df.shape[0])
    if included_rows == 0:
        return included_rows
    df_copy = df.copy().apply(pd.to_numeric, errors='ignore')
    actives_in_percent = np.sum(df_copy.head(included_rows)[activity_class])
    quotient = actives_in_percent / included_rows
    divisor = len(df_copy[df_copy.activity_class == 1]) / len(df_copy)
    return quotient / divisor

def process_combination(
                splitted_comb, 
                df_rescored, 
                ranking_method,
                output_path,
                index,
                weights
        ):
        """
        Rank poses using different ranking methods
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
        corr_dict ={
               'docking_tool': [], 
                'scoring_function': [],  
                'spearman_correlation': [],
                'cost_per_pipeline': [],
                'enrichment_factor': []
                # 'p_value': [],
                # 'confidence_interval': [],
                # 'Std': []
                }
        ranking_methods_dict = {  
                'best_ECR' : method1_ECR_best, 
                'ECR_average' : method2_ECR_average, 
                'average_ECR' : method3_avg_ECR,
                'rank_by_rank' : method4_RbR,
                'rank_by_vote' : method5_RbV,
                'best_Zscore': method6_Zscore_best,
                'average_Zscore': method7_Zscore_avg,
                'rank_by_number': method8_RbN,
                'weighted_ECR': method9_weighted_ECR_best
                }
        print(f'Running {ranking_method}...')

        # df = df_rescored.copy().fillna(0)
        # else:
        df = df_rescored.copy()
        df = df.drop('pose', axis=1)
        ranking_method_name = ranking_methods_dict[ranking_method].__name__
        for i, comb in enumerate(splitted_comb):
                # if set(comb[0]) == set(['PLANTS']) and set(comb[1]) == set(['CNNaffinity', 'SCORCH', 'CHEMPLP', 'rfscore_v1', 'vina_hydrophobic']):
                filtered_df = df[df['docking_tool'].isin(list(comb[0]))]
                try:    
                        if ranking_method == 'weighted_ECR':
                               df_rank = method9_weighted_ECR_best(
                                      filtered_df.copy(),
                                      list(comb[1]), 
                                      id_column='ID', 
                                      mapped_weights=weights
                                      )
                        else:
                                df_rank = ranking_methods_dict[ranking_method](filtered_df.copy(), 0.05, list(comb[1]), id_column='ID')

                except(RuntimeError, TypeError, NameError, pd.errors.MergeError, KeyError) as err:

                     
                        print(f"Error occurred: {err}")

                try:
                        df_rank_copy = df_rank.copy().apply(
                        pd.to_numeric, 
                        errors='ignore'
                        ).dropna().merge(
                                filtered_df[['ID', 'true_value', 'activity_class','id']], 
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

                # assert np.isclose(df_rank_copy[df_rank_copy.id == 'HIPS314'].true_value.iloc[0], 0.299342), "merging with wrong columns, ABORT!"
                # assert np.isclose(df_rank_copy[df_rank_copy.id == 'HIPS6980'].true_value.iloc[0], 0.370802), "merging with wrong columns, ABORT!"
                ##@@@@ spearman with p-value

                try:
                        spearman_corr, _ = spearmanr(df_rank_copy.loc[:, ranking_method_name], df_rank_copy['true_value'])
                        
                        ef = enrichment_factor_calc(df_unique_sorted, percent=10, activity_class='activity_class')
                        cost = runtime_cost_calculation(docking_tools=list(comb[0]), scoring_functions=list(comb[1]), num_poses=10)

                except (RuntimeError, TypeError, NameError, pd.errors.MergeError, KeyError) as err:
                        print("df_filter", filtered_df)
                        print(df_unique_sorted)   
                        print("df)rank copy",df_rank_copy.head(1))
                        print(f"Error occurred in calculations: {err}")

                corr_dict['docking_tool'].append(list(comb[0]))
                corr_dict['scoring_function'].append(list(comb[1]))
                corr_dict['spearman_correlation'].append(spearman_corr)
                corr_dict['cost_per_pipeline'].append(cost)
                corr_dict['enrichment_factor'].append(ef)

        corr_df = pd.DataFrame.from_dict(corr_dict, orient='index').transpose()
        corr_df['ranking_method'] = [ranking_method_name] * len(corr_df)

        corr_df.to_csv(str(output_path / f'{ranking_method}_parallel_{index}.csv'), index=False)


def poses_ranking(
        ranking_methods: list,
        df_rescored: pd.DataFrame,
        output_path: Path,
        validation="general",
        weights=None
):
        """
        Rank poses using different ranking methods

        Args:
                ranking_methods: list of ranking methods
                df_rescored: dataframe with rescored poses
                output_path: path to output folder
                df_scores: dataframe with ground truth scores with two columns: ID and score

        Return: 
                Write the results of every ranking method to a big csv file and concat all the results to a big csv file
        """
        ncpus = cpu_count()

        df_rescored[['ID', 'docking_tool', 'pose']] = df_rescored['ID'].str.split('_', expand=True)
        df_rescored = df_rescored[df_rescored['docking_tool'].notna()]
        docking_programs = list(df_rescored['docking_tool'].unique())
        print(f'Number of docking programs: {len(docking_programs)}, {docking_programs}')
        rescoring_methods = [col for col in df_rescored.columns if col not in ['ID', 'id', 'pose', 'docking_tool', 'true_value', 'activity_class']]
        print(f'Number of rescoring methods: {len(rescoring_methods)}, {rescoring_methods}')
        all_comb = all_combinations(docking_programs, rescoring_methods)
        print(
        f'Number of possible combinations for every ranking method: {len(all_comb)}'
        f'\n With total combinations : {len(all_comb) * len(ranking_methods)}'
        )
        splitted_comb = split_list(all_comb, ncpus)
        corr_file_path = output_path / f'correlations'
        if validation:
                corr_file_path = output_path / f'correlations_{validation}'

        corr_file_path.mkdir(parents=True, exist_ok=True)

        
        for ranking_method in (ranking_methods):

                if os.path.exists(str(corr_file_path / 'all_ranked.csv')):
                        print(f'All poses are ranked with all consensus methods ..')
                        break
                if os.path.exists(str(corr_file_path / f'{ranking_method}_concat.csv')):

                        print(f'File {ranking_method} exists. Skipping...')
                        continue

                print(f'Parallelizing {ranking_method}...')
                
                with concurrent.futures.ProcessPoolExecutor(max_workers=ncpus) as executor:
                        futures = [
                        executor.submit(
                                process_combination, comb, df_rescored, ranking_method, corr_file_path, i, weights
                        ) for i, comb in enumerate(splitted_comb)
                        ]
        #concatenate all the results
                df = pd.concat([pd.read_csv(str(corr_file_path / f'{ranking_method}_parallel_{i}.csv')) for i in range(ncpus)])
                df.to_csv(str(corr_file_path / f'{ranking_method}_concat.csv'), index=False)
                #delete the splitted files
                for i in range(ncpus):
                        os.remove(str(corr_file_path / f'{ranking_method}_parallel_{i}.csv'))
                print(f'Finished {ranking_method}...')
        #concatenate all the results
        if not os.path.exists(str(corr_file_path / 'all_ranked.csv')):
                df = pd.concat([pd.read_csv(str(corr_file_path / f'{ranking_method}_concat.csv')) for ranking_method in ranking_methods]).apply(pd.to_numeric, errors='ignore')
                [os.remove(str(corr_file_path / f'{ranking_method}_concat.csv')) for ranking_method in ranking_methods]
                df.sort_values(by='spearman_correlation', ascending=False, inplace=True)
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