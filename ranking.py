from DockM8.consensus_methods import *
from itertools import combinations, product
from multiprocessing import cpu_count
from utilities import split_list
import concurrent.futures
import time

def all_combinations(
        docking_programs: list,
        rescoring_programs: list,
):
    """
    Generate all combinations of docking methods and scoring functions
        @param docking_programs: list of docking methods
        @param df_rescored: dataframe with rescored poses

        return: list of tuples with all combinations of docking methods and scoring functions       
    """

    all_comb_scoring_function = [list(combinations(rescoring_programs, r))
                                     for r in range(1, len(rescoring_programs) + 1)]
    all_comb_scoring_function = [item for sublist in all_comb_scoring_function for item in sublist]


    all_comb_docking_program = [list(combinations(docking_programs, r))
                                     for r in range(1, len(docking_programs) + 1)]

    all_comb_docking_program = [item for sublist in all_comb_docking_program for item in sublist]
    return list(product(all_comb_docking_program, all_comb_scoring_function))


def process_combination(
                splitted_comb, 
                df_rescored, 
                df_scores, 
                ranking_method,
                output_path,
                index):
        """
        Rank poses using different ranking methods
                @param splitted_comb: list of splitted combinations
                @param df_rescored: dataframe with rescored poses
                @param df_scores: dataframe with ground truth scores with two columns: ID and score
                @param ranking_method: ranking method
                @param output_path: path to output folder
                @param index: index of the splitted_comb

                return: Write the results of every ranking method to a csv file
        """
        corr_dict ={
               'docking_program': [], 
                'scoring_function': [],  
                'spearman_correlation': []
                }
        start_time = time.time() 
        ranking_methods_dict = {  
                'best_ECR' : method1_ECR_best, 
                'ECR_average' : method2_ECR_average, 
                'average_ECR' : method3_avg_ECR,
                'rank_by_rank' : method4_RbR,
                'rank_by_vote' : method5_RbV,
                'best_Zscore': method6_Zscore_best,
                'average_Zscore': method7_Zscore_avg
                }
        # print(f'Running {ranking_method}...')
        df = df_rescored.copy().fillna(0)
        for comb in tqdm(splitted_comb, total=len(splitted_comb)):
                
                filtered_df = df[df['docking_program'].isin(list(comb[0]))]
                df_rank = ranking_methods_dict[ranking_method](filtered_df.copy(),'allposes', list(comb[1])).merge(df_scores[['ID', 'true_value']], on='ID')
                df_rank_copy = df_rank.copy().apply(pd.to_numeric, errors='ignore')
                #@TODO normalize the df_rank

                df_rank_copy.iloc[:,2] = (df_rank_copy.iloc[:,2] - df_rank_copy.iloc[:,2].mean()) / df_rank_copy.iloc[:,2].std()
                df_rank_copy.iloc[:,1] = (df_rank_copy.iloc[:,1] - df_rank_copy.iloc[:,1].mean()) / df_rank_copy.iloc[:,1].std()

                # # pearson_corr = df_rank.iloc[:, 1].corr(df_rank.iloc[:, 2], method='pearson')
                spearmanr_corr = df_rank_copy.iloc[:,1].corr(df_rank_copy.iloc[:,2], method='spearman')
                # print(spearmanr_corr)
                corr_dict['docking_program'].append(list(comb[0]))
                corr_dict['scoring_function'].append(list(comb[1]))
                # 'pearson_correlation': pearson_corr, 
                corr_dict['spearman_correlation'].append(spearmanr_corr)

                # Append the new data to the original DataFrame
                # corr_df.loc[len(corr_df)] = new_row
                # display(corr_dict)
        # corr_dict['ranking_method'] = [ranking_method] * len(corr_dict['docking_program'] - 1)
        corr_df = pd.DataFrame.from_dict(corr_dict, orient='index').transpose()
        corr_df.to_csv(str(output_path / f'{ranking_method}_parallel_{index}.csv'), index=False)
        end_time = time.time()  # Record the current time after the code has been executed

        execution_time = end_time - start_time  # Calculate the difference
        print(f"Execution time: {execution_time} seconds")

def poses_ranking(
        ranking_methods: list,
        df_rescored: pd.DataFrame,
        output_path: Path,
        df_scores: pd.DataFrame
):
        """
        Rank poses using different ranking methods
        @param ranking_methods: list of ranking methods
        @param df_rescored: dataframe with rescored poses
        @param output_path: path to output folder
        @param df_scores: dataframe with ground truth scores with two columns: ID and score

        return: Write the results of every ranking method to a big csv file and concat all the results to a big csv file
        """
        ncpus = cpu_count() - 1 if cpu_count() > 1 else 1

        df_rescored[['ID', 'docking_program', 'pose']] = df_rescored['ID'].str.split('_', expand=True)
        docking_programs = list(df_rescored['docking_program'].unique())
        rescoring_methods = [col for col in df_rescored.columns if col not in ['ID', 'pose', 'docking_program']]

        all_comb = all_combinations(docking_programs, rescoring_methods)
        print(f'Number of possible combinations for every ranking method: {len(all_comb)}\n With total combinations : {len(all_comb) * len(ranking_methods)}')
        splitted_comb = split_list(all_comb, ncpus)

        corr_file_path = output_path / 'correlations'
        corr_file_path.mkdir(parents=True, exist_ok=True)

        
        for ranking_method in (ranking_methods):

                if os.path.exists(str(corr_file_path / 'all_ranked.csv')) or os.path.exists(str(corr_file_path / f'{ranking_method}_concat.csv')):
                        print(f'{ranking_method} is already ranked. Skipping...')
                        continue

                print(f'Parallelizing {ranking_method}...')
                with concurrent.futures.ProcessPoolExecutor(max_workers=ncpus) as executor:
                        futures = [
                        executor.submit(
                                process_combination, comb, df_rescored, df_scores, ranking_method, corr_file_path, i
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
        df = pd.concat([pd.read_csv(str(corr_file_path / f'{ranking_method}_concat.csv')) for ranking_method in ranking_methods])
        df.to_csv(str(corr_file_path / 'all_ranked.csv'), index=False)
