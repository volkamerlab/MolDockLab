import pandas as pd
import matplotlib.pyplot as plt
# import seaborn as sns
import ast
import numpy as np
import matplotlib.pyplot as plt
import logging
from pathlib import Path
import os
import seaborn as sns
from scipy.stats import spearmanr
from tqdm.autonotebook import tqdm
from src.data_preparation import run_gypsumdl
from IPython.display import display, Image
from sklearn.model_selection import train_test_split
from src.docking import docking, poses_checker
from src.utilities import get_selected_workflow
from src.rescoring import rescoring_function
from src.preprocessing import merge_activity_values, hdbscan_scaffold_split, cv_split, norm_scores
from src.pose_score_optimization import scores_preprocessing, score_pose_optimization, prepare_parameters, prediction, mapping_normalized_weights
from src.interaction_analysis import (
    split_sdf_path, 
    actives_extraction, 
    plipify_fp_interaction, 
    indiviudal_interaction_fp_generator, 
    read_interactions_json, 
    interactions_aggregation
)
from src.consensus_rank import *

from argparse import ArgumentParser, Namespace, FileType
from src.ranking import *
import matplotlib.pyplot as plt

from pathlib import Path

def get_parser():
    parser = ArgumentParser()
    parser.add_argument('--protein_path', type=str, default=None, help='Path to the protein file')
    parser.add_argument('--ref_ligand_path', type=str, default=None, help='Path to the reference ligand file')
    parser.add_argument('--known_ligands_path', type=str, default=None, help='Path to the experimentally validated ligands library')
    parser.add_argument('--sbvs_ligands_path', type=str, default=None, help='Path to the larger ligands library for SBVS')
    
    parser.add_argument('--corr_range', type=float, default=0.1, help='The allowed range of the Spearman correlation to select a pipeline with lowest runtime cost')
    parser.add_argument('--ef_range', type=float, default=0.5, help='The allowed enrichment factor of the highest Spearman correlation pipeline to select a pipeline with lowest runtime cost')

    parser.add_argument('--out_dir', type=str, default='output', help='The output directory to save the results.')

    parser.add_argument('--activity_col', type=str, default='activity_class', help='The column name of the activity class in the ligands library (0 inactive, 1 active)')
    parser.add_argument('--id_col', type=str, default='ID', help='The column name of the ID in the ligands library')
    parser.add_argument('--true_value_col', type=str, default=None, help='The column name of the true value in the ligands library')
    parser.add_argument('--protein_name', type=str, default=None, help='The name of the protein')
    parser.add_argument('--interacting_chains', nargs='+', default=['X'], help='The chains that included in the protein-ligand interactions')

    parser.add_argument('--docking_programs', nargs='+', default=['gnina', 'smina', 'diffdock', 'plants', 'flexx'], type=str, help='The docking tools to use. The docking tools are gnina, smina, diffdock, plants, and flexx')
    parser.add_argument('--n_poses', default=10, type=int, help='The number of poses to generate per docking tool')
    parser.add_argument('--exhaustiveness', default=8, type=int, help='The exhaustiveness of the docking program for SMINA and GNINA docking tools')
    parser.add_argument('--local_diffdock', default=False, type=bool, help='Whether to use local diffdock or not. Only recommended when in case DiffDock doesn\'t predict the binding pocket correctly')

    parser.add_argument('--pose_quality_checker', default=False, type=bool, help='Whether to use pose quality checker for generated poses using PoseBusters')
    parser.add_argument('--versatility_analysis', default=False, type=bool, help='Whether to use the versatility analysis to check the performance of the MolDockLab workflow')
    parser.add_argument('--rescoring', nargs='+', default=[
        'cnnscore', 'ad4', 'linf9', 'rtmscore', 'vinardo', 'chemplp', 'rfscore_v1', 'rfscore_v3', 'vina_hydrophobic', 'vina_intra_hydrophobic'
    ], type=str, help='The rescoring functions to use. The rescoring functions are cnnscore, cnnaffinity, smina_affinity, ad4, linf9, rtmscore, vinardo, scorch, hyde, chemplp, rfscore_v1, rfscore_v2, rfscore_v3, vina_hydrophobic, vina_intra_hydrophobic')

    parser.add_argument('--corr_threshold', default=0.9, type=float, help='The Spearman correlation threshold  to remove highly correlated scores from the rescoring results')
    # parser.add_argument('--activity_threshold', default=None, type=int, help='The activity threshold to categorize the ligands as active or inactive')

    parser.add_argument('--n_cpus', default=1, type=int, help='The number of CPUs to use in the workflow for Rescoring and ranking steps.')
    parser.add_argument('-l', '--log', '--loglevel', type=str, default='WARNING', dest="loglevel",
                        help='Log level. Default %(default)s')
    parser.add_argument('--ranking_method', nargs='+', default= ['best_ECR', 'rank_by_rank' , 'best_Zscore', 'weighted_ECR'], help='The ranking method to use. The ranking methods ,that can be selected, are : best_ECR, ECR_average, average_ECR, rank_by_rank, rank_by_vote, rank_by_number, best_Zscore, average_Zscore, weighted_ECR')    
    parser.add_argument('--runtime_reg', type=float, default=0.1, help='Regularization parameter for the runtime cost for each tool in pose score optimization. It can be list of floats or a float')
    
    return parser

def main(args):

    HERE = Path(__file__).resolve().parent
    OUTPUT = HERE / args.out_dir
    
    OUTPUT.mkdir(exist_ok=True, parents=True)
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    
    # Check if the required arguments are provided
    if args.protein_path is None:
        raise ValueError("Please provide the path to the protein file")
    else:
        if not Path(args.protein_path).exists():
            raise ValueError("The provided protein file does not exist")
    if args.ref_ligand_path is None:
        raise ValueError("Please provide the path to the reference ligand file")
    else:
        if not Path(args.ref_ligand_path).exists():
            raise ValueError("The provided reference ligand file does not exist")
    if args.known_ligands_path is None:
        raise ValueError("Please provide the path to the experimentally validated ligands library")
    else:
        if not Path(args.known_ligands_path).exists():
            raise ValueError("The provided experimentally validated ligands library does not exist")
    if args.sbvs_ligands_path is None:
        raise ValueError("Please provide the path to the larger ligands library for SBVS")
    else:
        if not Path(args.sbvs_ligands_path).exists():
            raise ValueError("The provided larger ligands library for SBVS does not exist")
    if args.true_value_col is None:
        raise ValueError("Please provide the column name of the true value in the ligands library")

    # if args.activity_threshold is None:
    #     raise ValueError("Please provide the activity threshold to categorize the ligands as active or inactive")
    
    if args.protein_name is None:
        protein_name = Path(args.protein_path).stem
    else:
        protein_name = args.protein_name
    
    if args.n_cpus is None:
        n_cpu = os.cpu_count() - 2
    else:
        n_cpu = args.n_cpus

    logger.info("\n\nStarting MolDockLab workflow ...\n\n")
    logger.info(f"{args}")
    logger.info("Preparing the ligands library for docking using Gypsum-DL 1.2.1")
    output_prepared_mols = OUTPUT / f"{Path(args.known_ligands_path).stem}_prepared.sdf"
    try:
        run_gypsumdl(
            ligand_library = (HERE / args.known_ligands_path), 
            prepared_library_path=output_prepared_mols, 
            id_column=args.id_col
            )
    except Exception as e:
        logger.error(f"An error occured while preparing the ligands library for docking: {e}")
        return
    
    logger.info(f"Docking the ligands library with experimental data using {args.docking_programs}...")
    try:
        docking(
                docking_methods=args.docking_programs,
                protein_file=HERE / args.protein_path,
                ligands_path=HERE / output_prepared_mols,
                ref_file=HERE / args.ref_ligand_path,
                exhaustiveness=args.exhaustiveness,
                n_poses=args.n_poses,
                OUTPUT=OUTPUT,
                local_diffdock=args.local_diffdock,
                )
    except Exception as e:
        logger.error(f"An error occured while docking the ligands library: {e}")
        return
    
    if args.pose_quality_checker:
        logger.info("Checking the quality of the generated poses using PoseBusters ...")
        _docking_tools_results = poses_checker(
            poses_path= OUTPUT / 'allposes.sdf', 
            protein_path= HERE / args.protein_path, 
            output_file= OUTPUT / f'{protein_name}_posebusters_results.csv'
            )
        logger.info(f"PoseBusters results are saved at {OUTPUT / f'{protein_name}_posebusters_results.csv'}")

    logger.info(f"Rescoring the docking results using SFs: {args.rescoring} ...")
    try:
        rescoring_function(
            rescoring_programs=args.rescoring, 
            protein_path=HERE / args.protein_path,
            docked_library_path=OUTPUT / 'allposes.sdf', 
            ref_file=HERE / args.ref_ligand_path,
            ncpu=n_cpu,
            )
        logger.info(f"Rescoring results are saved at {OUTPUT / 'rescoring_results' / 'all_rescoring_results.csv'}")
    except Exception as e:
        logger.error(f"An error occured while rescoring the docking results: {e}")
        return
    
    logger.info("Merging the activity values with scoring functions results ...")
    try:
        df_rescored_merged = merge_activity_values(
            norm_scored_path=OUTPUT / 'rescoring_results' / 'all_rescoring_results.csv',
            mols_true_value_path=(HERE / args.known_ligands_path), 
            true_value_col=args.true_value_col,
            scored_id_col=args.id_col,
            activity_col=args.activity_col,
            threshold=args.corr_threshold
            )
        print(df_rescored_merged)
        logger.info(f"Merged results are saved at {OUTPUT / 'rescoring_results' / 'all_rescoring_results_merged.csv'}")
    except Exception as e:
        logger.error(f"An error occured while merging the activity values with the docking results: {e}")
        return
    
    try:
        logger.info("Normalizing the predicted scores ...")
        df_rescored_norm = norm_scores(df_rescored_merged)
        df_rescored_norm.to_csv(str(OUTPUT / 'rescoring_results' / 'all_rescoring_results_merged_norm_robust.csv'), index=False)
        plt.figure(figsize=(20, 9))
        sns.boxplot(data=df_rescored_norm.drop(columns=['true_value']) ,linewidth=.85, showfliers=False)
        plt.ylim(-1, 1)
        plt.title('Box Plot of Normalized Predicted Scores of the Docking Poses')
        plt.xticks(rotation=45) 
        # save the plot
        plt.savefig(OUTPUT / 'normalized_scores_boxplot.png')
        logger.info(f"Normalized scores boxplot is saved at {OUTPUT / 'normalized_scores_boxplot.png'}")
    except Exception as e:
        logger.error(f"An error occured while normalizing the predicted scores: {e}")
        return
    
    if 'weighted_ECR' in args.ranking_method:
        try:
            logger.info("Performing the pose score optimization ...")
            X, y, docking_cost, scoring_cost, docking_tools, scoring_tools = scores_preprocessing(
                df_rescored_merged)
            print(X.shape, y.shape, docking_cost.shape, scoring_cost.shape, docking_tools, scoring_tools)
            if isinstance(args.runtime_reg, float):
                alpha = [args.runtime_reg]
            else:
                alpha = args.runtime_reg
            best_weights = score_pose_optimization(
                X=X, 
                y=y, 
                docking_cost=docking_cost, 
                scoring_cost=scoring_cost, 
                weights_path= OUTPUT / 'best_weights.pkl.npy', 
                alphas=alpha, 
                )
            logger.info(f"Best weights are saved at {HERE / 'test_data/best_weights.pkl.npy'}")
        except Exception as e:
            logger.error(f"An error occured while performing the pose score optimization: {e}")
            return
        
        try:
            logger.info("Normalize the optimized weights to the docking and scoring tools ...")
            for alpha in best_weights.keys():
                logger.info("Normalize the optimized weights to the docking and scoring tools ...")
                normalized_weights = mapping_normalized_weights(best_weights[alpha], scoring_tools, docking_tools)
                
        except Exception as e:
            logger.error(f"An error occured while normalizing the optimized weights to the docking and scoring tools: {e}")
            return
        
    try:
        logger.info(f"Ranking the ligands library selecting {args.ranking_method}...")
        df_rescored_norm = norm_scores(df_rescored_merged)
        if normalized_weights is None:
            poses_ranking(
            ranking_methods=args.ranking_method,
            df_rescored=df_rescored_norm,
            output_path=OUTPUT,
            )
        else:
            poses_ranking(
            ranking_methods=args.ranking_method,
            df_rescored=df_rescored_norm,
            output_path=OUTPUT,
            weights=normalized_weights,
            )
    except Exception as e:
        logger.error(f"An error occured while ranking the ligands library: {e}")
        return
    
    # selecting best balanced pipeline
    try:
        logger.info("Selecting the best balanced pipeline ...")
        corr_df = pd.read_csv(OUTPUT / 'correlations_general' /  'all_ranked.csv')
        range_workflows = corr_df[(corr_df['spearman_correlation'] >= corr_df['spearman_correlation'].max() - args.corr_range) &
                          (corr_df['enrichment_factor'] >= corr_df.loc[0, 'enrichment_factor'] - args.ef_range)]
        
        # select row with the minimum cost value
        selected_workflow = range_workflows.loc[range_workflows['cost_per_pipeline'].idxmin()]
        logger.info(
            f"The best pipeline uses for docking: {selected_workflow['docking_tool']} "
            f"and for rescoring: {selected_workflow['scoring_function']} " 
            f"with a Spearman correlation of {selected_workflow['spearman_correlation']}"
            f"a cost of {selected_workflow['cost_per_pipeline']}")
        
        selected_docking_tools = ast.literal_eval(selected_workflow['docking_tool'])
        selected_sfs = ast.literal_eval(selected_workflow['scoring_function'])
        selected_ranking_method = selected_workflow['ranking_method']
        saved_time =  corr_df.loc[0, 'cost_per_pipeline'] - selected_workflow['cost_per_pipeline']
        logger.info(f"By selecting the best balanced pipeline, you saved {saved_time} seconds per compound.")
    except Exception as e:
        logger.error(f"An error occured while selecting the best balanced pipeline: {e}")
        return

    # screen the larger ligands library for SBVS
    logger.info("Screening the larger ligands library for SBVS ...")
    larger_data_output = OUTPUT / 'larger_data_output'
    larger_data_output.mkdir(exist_ok=True, parents=True)

    logger.info("Preparing the larger ligands library for docking using Gypsum-DL 1.21")
    output_prepared_mols = OUTPUT / f"{Path(args.sbvs_ligands_path).stem}_prepared.sdf"
    try:
        run_gypsumdl(
            ligand_library = (HERE / args.sbvs_ligands_path), 
            prepared_library_path=output_prepared_mols, 
            id_column=args.id_col
            )
    except Exception as e:
        logger.error(f"An error occured while preparing the larger ligands library for docking: {e}")
        return
    
    logger.info(f"Docking the larger ligands library with unknown experimental data using {selected_workflow['docking_tool']}...")
    try:
        docking(
                docking_methods=selected_docking_tools,
                protein_file=HERE / args.protein_path,
                ligands_path=HERE / output_prepared_mols,
                ref_file=HERE / args.ref_ligand_path,
                exhaustiveness=args.exhaustiveness,
                n_poses=args.n_poses,
                OUTPUT=larger_data_output,
                local_diffdock=args.local_diffdock,
                )
    except Exception as e:
        logger.error(f"An error occured while docking the larger ligands library: {e}")
        return
    
    logger.info(f"Rescoring the docking results of unknown data using selected SFs {selected_workflow['scoring_function']} ...")
    try:
        rescoring_function(
            rescoring_programs=selected_sfs, 
            protein_path=HERE / args.protein_path,
            docked_library_path=larger_data_output / 'allposes.sdf', 
            ref_file=HERE / args.ref_ligand_path,
            ncpu=n_cpu,
            )
        logger.info(f"Rescoring results are saved at {larger_data_output / 'rescoring_results' / 'all_rescoring_results.csv'}")
    except Exception as e:
        logger.error(f"An error occured while rescoring the docking results: {e}")
        return

    logger.info(f"Ranking unknown poses using {selected_ranking_method}...")
    ranking_methods_dict = {  
        'method1_ECR_best' : method1_ECR_best, 
        'method2_ECR_average' : method2_ECR_average, 
        'method3_avg_ECR' : method3_avg_ECR,
        'method4_RbR' : method4_RbR,
        'method5_RbV' : method5_RbV,
        'method6_Zscore_best': method6_Zscore_best,
        'method7_Zscore_avg': method7_Zscore_avg,
        'method8_RbN': method8_RbN,
        'method9_weighted_ECR_best': method9_weighted_ECR_best
        }
    
    try:
        rescored_df_sbvs = pd.read_csv(larger_data_output / 'rescoring_results' / 'all_rescoring_results.csv')
        rescored_df_sbvs_norm = norm_scores(rescored_df_sbvs)

        if selected_ranking_method == 'weighted_ECR':
            ranked_sbvs_ligands = ranking_methods_dict[selected_ranking_method](
                df=rescored_df_sbvs_norm,
                selected_scores=selected_sfs,
                id_column=args.id_col,
                mapped_weights= normalized_weights,
                )
        ranked_sbvs_ligands = ranking_methods_dict[selected_ranking_method](
            df=rescored_df_sbvs_norm, 
            selected_scores=selected_sfs,
            id_column=args.id_col,
            weight=0.05,
            )
        ranked_sbvs_ligands.sort_values(by=selected_ranking_method, ascending=False, inplace=True)
        ranked_sbvs_ligands.to_csv(larger_data_output / 'ranked_ligands.csv', index=False)
        logger.info(f"Ranked unknown ligands are saved at {larger_data_output / 'ranked_ligands.csv'}")
        
    except Exception as e:
        logger.error(f"An error occured while ranking the ligands library: {e}")
        return

    # Interaction analysis
    logger.info("Performing the interaction analysis using PLIPify...")
    try:
        actives_path = actives_extraction( 
            OUTPUT / 'allposes.sdf', 
            OUTPUT / 'rescoring_results/all_rescoring_results_merged.csv', 
            docking_tool=selected_docking_tools
            )

        actives_paths = split_sdf_path(actives_path)
        
        for chain in args.interacting_chains:
            interx_csv = OUTPUT / f'{protein_name}_{chain}_interx.csv'
            if interx_csv.is_file():
                logger.info(f"Protein-ligand interactions with chain {chain} are already saved at {interx_csv}")
                continue
            fp_focused = plipify_fp_interaction(
                ligands_path=actives_paths, 
                protein_path=HERE / args.protein_path, 
                protein_name=protein_name, 
                chains=chain,
                output_file=OUTPUT / 'egfr_interactions_X.png'
                )
            fp_focused['total_interactions'] = fp_focused.sum(axis=1)
            fp_focused.to_csv(interx_csv)
            logger.info(f"Protein-ligand interactions with chain {chain} are saved at {interx_csv}")
    except Exception as e:
        logger.error(f"An error occured while performing the interaction analysis: {e}")
        return
    
    logger.info("Filtering compounds with specific interactions ...")
    try:
         
    # if args.versatility_analysis:
    #     logger.info("Performing the versatility analysis ...")
    #     try:
    #         clustered_df = hdbscan_scaffold_split(
    #             original_data_path=HERE / args.known_ligands_path, 
    #             min_cluster_size=2
    #             )
    #         cv_split(
    #             clustered_df=clustered_df, 
    #             df_rescored=df_rescored_merged, 
    #             idx_col=args.id_col, 
    #             n_splits=5, 
    #             output_path= DATA / 'versatility_analysis', 
    #             target_name=args.protein_name
    #             )
    #     except Exception as e:
    #         logger.error(f"An error occured while performing the versatility analysis: {e}")
    #         return
        

if __name__ == "__main__":
    _args = get_parser().parse_args()
    main(_args)