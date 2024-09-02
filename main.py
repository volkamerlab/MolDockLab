import pandas as pd
import matplotlib.pyplot as plt
# import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import logging
from pathlib import Path
import os
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
    
    parser.add_argument('--id_col', type=str, default='ID', help='The column name of the ID in the ligands library')
    parser.add_argument('--true_value_col', type=str, default=None, help='The column name of the true value in the ligands library')
    parser.add_argument('--protein_name', type=str, default=None, help='The name of the protein')
    parser.add_argument('n_cpus', type=int, default=1, help='The number of CPUs to use the workflow')

    parser.add_argument('--docking_programs', nargs='+', default=['gnina', 'smina', 'diffdock', 'plants', 'flexx'], type=str, help='The docking tools to use. The docking tools are gnina, smina, diffdock, plants, and flexx')
    parser.add_argument('--n_poses', default=10, type=int, help='The number of poses to generate per docking tool')
    parser.add_argument('--exhaustiveness', default=8, type=int, help='The exhaustiveness of the docking program for SMINA and GNINA docking tools')
    parser.add_argument('--local_diffdock', default=False, type=bool, help='Whether to use local diffdock or not. Only recommended when in case DiffDock doesn\'t predict the binding pocket correctly')

    parser.add_argument('--pose_quality_checker', default=False, type=bool, help='Whether to use pose quality checker for generated poses using PoseBusters')
    parser.add_argument('--versatility_analysis', default=False, type=bool, help='Whether to use the versatility analysis to check the performance of the MolDockLab workflow')
    parser.add_argument('--rescoring', nargs='+', default=[
        'cnnscore', 'ad4', 'linf9', 'rtmscore', 'vinardo', 'chemplp', 'rfscore_v1', 'rfscore_v3', 'vina_hydrophobic', 'vina_intra_hydrophobic'
    ], type=str, help='The rescoring functions to use. The rescoring functions are cnnscore, cnnaffinity, smina_affinity, ad4, linf9, rtmscore, vinardo, scorch, hyde, chemplp, rfscore_v1, rfscore_v2, rfscore_v3, vina_hydrophobic, vina_intra_hydrophobic')

    parser.add_argument('--collinearity_threshold', default=0.9, type=float, help='The collinearity threshold to remove highly correlated scores from the rescoring results')
    #parser.add_argument('--activity_threshold', default=None, type=int, help='The activity threshold to categorize the ligands as active or inactive')

    parser.add_argument('--ranking_method', nargs='+', default= ['best_ECR', 'rank_by_rank' , 'best_Zscore', 'weighted_ECR'], help='The ranking method to use. The ranking methods ,that can be selected, are : best_ECR, ECR_average, average_ECR, rank_by_rank, rank_by_vote, rank_by_number, best_Zscore, average_Zscore, weighted_ECR')    
    parser.add_argument('--runtime_reg', type=float, default=0.1, help='Regularization parameter for the runtime cost for each tool in pose score optimization. It can be list of floats or a float')
    return parser

def main(args):

    HERE = Path(__file__).resolve().parent
    DATA = HERE / 'data'
    OUTPUT = HERE / 'output'
    
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
    

    if args.protein_name is None:
        protein_name = Path(args.protein_path).stem
    else:
        protein_name = args.protein_name
    
    if args.n_cpus is None:
        n_cpu = os.cpu_count() - 2
    else:
        n_cpu = args.n_cpus

    logger.info("\n\nStarting MolDockLab workflow ...\n\n")

    logger.info("Preparing the ligands library for docking using Gypsum-DL 1.21")
    output_prepared_mols = DATA / f"{Path(args.known_ligands_path).stem}_prepared.sdf"
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
                ref_file=args.ref_ligand_path,
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
            protein_path=args.protein_path,
            docked_library_path=OUTPUT / 'allposes.sdf', 
            ref_file=args.ref_ligand_path,
            ncpu=n_cpu,
            )
        logger.info(f"Rescoring results are saved at {OUTPUT / 'rescored_results.csv'}")
    except Exception as e:
        logger.error(f"An error occured while rescoring the docking results: {e}")
        return
    
    logger.info("Merging the activity values with the docking results ...")
    try:
        df_rescored_merged = merge_activity_values(
            norm_scored_path=OUTPUT / 'all_rescoring_results.csv',
            mols_true_value_path=(HERE / args.known_ligands_path), 
            true_value_col=args.true_value_col,
            id_col=args.id_col,
            activity_threshold=args.activity_threshold
            )
        logger.info(f"Merged results are saved at {OUTPUT / 'merged_results.csv'}")
    except Exception as e:
        logger.error(f"An error occured while merging the activity values with the docking results: {e}")
        return
    
    if 'weighted_ECR' in args.ranking_method:
        try:
            logger.info("Performing the pose score optimization ...")
            X, y, docking_cost, scoring_cost, docking_tools, scoring_tools = scores_preprocessing(
                df_rescored_merged.drop(columns=['activity_class']))
            best_weights = score_pose_optimization(
                X=X, 
                y=y, 
                docking_cost=docking_cost, 
                scoring_cost=scoring_cost, 
                weights_path= DATA / 'best_weights.pkl.npy', 
                alphas=args.runtime_reg, 
                )
            logger.info(f"Best weights are saved at {DATA / 'best_weights.pkl.npy'}")
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