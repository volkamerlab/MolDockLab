import os
import ast
import logging

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from pathlib import Path
from rdkit import Chem
from rdkit.Chem import Draw, PandasTools
#from sklearn.model_selection import train_test_split
from argparse import ArgumentParser, ArgumentTypeError

from src.ranking import *
from src.consensus_rank import *
from src.data_preparation import run_gypsumdl
from src.diversity_selection import diversity_selection
from src.docking import docking, poses_checker
from src.rescoring import rescoring_function
from src.preprocessing import (
    merge_activity_values, 
#    hdbscan_scaffold_split, 
#    cv_split, 
    norm_scores
    )
from src.pose_score_optimization import (
    scores_preprocessing, 
    score_pose_optimization, 
    mapping_normalized_weights
    )
from src.interaction_analysis import (
    split_sdf_path, 
    actives_extraction, 
    plipify_fp_interaction, 
    indiviudal_interaction_fp_generator, 
    read_interactions_json, 
    interactions_aggregation
    )

ALLOWED_DOCKING_PROGRAMS = ['gnina', 'smina', 'diffdock', 'plants', 'flexx']

ALLOWED_SCORING_FUNCTIONS = ['cnnscore', 
                       'cnnaffinity', 
                       'smina_affinity', 
                       'ad4', 
                       'linf9', 
                       'rtmscore', 
                       'vinardo', 
                       'scorch', 
                       'hyde', 
                       'chemplp', 
                       'rfscore_v1', 
                       'rfscore_v2', 
                       'rfscore_v3', 
                       'vina_hydrophobic', 
                       'vina_intra_hydrophobic']

ALLOWED_RANKING_METHODS = ['best_ECR',
                            'ECR_average',
                            'average_ECR',
                            'rank_by_rank',
                            'rank_by_vote',
                            'rank_by_number',
                            'best_Zscore',
                            'average_Zscore',
                            'weighted_ECR']

def validate_docking_programs(programs):
    """Validate that the provided docking programs are in the allowed list."""
    for program in programs:
        if program not in ALLOWED_DOCKING_PROGRAMS and program != 'all':
            raise ArgumentTypeError(f"'{program}' is not a valid docking program. Allowed values are: {', '.join(ALLOWED_DOCKING_PROGRAMS)}")
    return programs

def validate_scoring_functions(functions):
    """Validate that the provided scoring functions are in the allowed list."""
    for function in functions:
        if function not in ALLOWED_SCORING_FUNCTIONS and function != 'all':
            raise ArgumentTypeError(f"'{function}' is not a valid scoring function. Allowed values are: {', '.join(ALLOWED_SCORING_FUNCTIONS)}")
    return functions

def validate_ranking_methods(methods):
    """Validate that the provided ranking methods are in the allowed list."""
    for method in methods:
        if method not in ALLOWED_RANKING_METHODS and method != 'all':
            raise ArgumentTypeError(f"'{method}' is not a valid ranking method. Allowed values are: {', '.join(ALLOWED_RANKING_METHODS)}")
    return methods
def valid_file_path(path):
    """Check if the provided path is a valid file."""
    if os.path.isfile(path):
        return path
    else:
        raise ArgumentTypeError(f"'{path}' is not a valid file.")
    
def get_parser():
    parser = ArgumentParser(description="MolDockLab Workflow Argument Parser")
    # Required arguments
    parser.add_argument('--protein_path', type=valid_file_path, required=True, help='Path to the protein file')
    parser.add_argument('--ref_ligand_path', type=valid_file_path, required=True, help='Path to the reference ligand file')
    parser.add_argument('--known_ligands_path', type=valid_file_path, required=True, help='Path to the experimentally validated ligands library. Known ligand library has to include the true value column and the activity class column')
    parser.add_argument('--sbvs_ligands_path', type=valid_file_path, required=True, help='Path to the larger ligands library for SBVS')
    parser.add_argument('--true_value_col', type=str, required=True, help='The column name of the true value in the ligands library')

    # Optional arguments 
    parser.add_argument('--activity_col', type=str, default='activity_class', help='The column name of the activity class in the ligands library (0 inactive, 1 active)')
    parser.add_argument('--id_col', type=str, default='ID', help='The column name of the ID in the ligands library')
    parser.add_argument('--protein_name', type=str, default=None, help='The name of the protein')
    parser.add_argument('--n_cpus', default=1, type=int, help='The number of CPUs to use in the workflow for Rescoring and ranking steps.')
    parser.add_argument('--out_dir', type=str, default='output', help='The output directory to save the results.')
    parser.add_argument('--verbose', action='store_true', default=False, help='Showing detailed output.')
    parser.add_argument('--true_value_scale', default=False,  action='store_true', help='Whether the true value is lower the better or not, e.g. (IC50 values). default is False (higher the better e.g. pIC50)')

    # docking args
    parser.add_argument(
        '--docking_programs', 
        nargs='+', 
        default=['gnina', 'smina', 'diffdock', 'plants'], 
        type=str,  
        help=f"The docking tools to use. Allowed values: {', '.join(ALLOWED_DOCKING_PROGRAMS)}"
    )
    parser.add_argument('--n_poses', default=10, type=int, help='The number of poses to generate per docking tool')
    parser.add_argument('--exhaustiveness', default=8, type=int, help='The exhaustiveness of the docking program for SMINA and GNINA docking tools')
    parser.add_argument('--local_diffdock', default=False,  action='store_true', help='Whether to use local diffdock or not. Only recommended when in case DiffDock doesn\'t predict the binding pocket correctly')

    # rescoring args
    parser.add_argument(
        '--rescoring', 
        nargs='+', 
        default=['cnnscore', 'ad4', 'linf9', 'rtmscore', 'vinardo', 'chemplp', 'rfscore_v1', 'rfscore_v3', 'vina_hydrophobic', 'vina_intra_hydrophobic'],
        type=str,
        help=f"The rescoring functions to use. Allowed values: {', '.join(ALLOWED_SCORING_FUNCTIONS)}"
    )
    parser.add_argument('--corr_threshold', default=0.9, type=float, help='The Spearman correlation threshold  to remove highly correlated scores from the rescoring results')

    # ranking args
    parser.add_argument(
        '--ranking_method', 
        nargs='+', 
        default=['best_ECR', 'rank_by_rank', 'best_Zscore', 'weighted_ECR'], 
        type=str,
        help=f"The ranking method to use. Allowed values: {', '.join(ALLOWED_RANKING_METHODS)}"
    )
    parser.add_argument('--runtime_reg', type=float, default=0.1, help='Regularization parameter for the runtime cost for each tool in pose score optimization. It can be list of floats or a float')
    
    # selecting best balanced pipeline args
    parser.add_argument('--corr_range', type=float, default=0.1, help='The allowed range of the Spearman correlation to select a pipeline with lowest runtime cost')
    parser.add_argument('--ef_range', type=float, default=0.5, help='The allowed enrichment factor of the highest Spearman correlation pipeline to select a pipeline with lowest runtime cost')


    # interaction analysis args
    parser.add_argument('--interacting_chains', nargs='+', default=['X'], help='The chains that included in the protein-ligand interactions')
    parser.add_argument('--key_residues', nargs='+', default=None, help='The key residues for protein-ligand interactions to consider in the interaction filtration. If None, The top four frequent interacting residues found in active compounds are considered. added by resdiue number + chain, e.g. 123A 124B , etc')

    # diversity selection args
    parser.add_argument('--diversity_selection', action='store_true', help='Whether to use diversity selection step or not')
    parser.add_argument('--n_clusters', type=int, default=5, help='The number of clusters that the centroids (central compounds) are selected in the diversity selection step')

    # Quality checker args
    parser.add_argument('--pose_quality_checker', action='store_true', help='Whether to use pose quality checker for generated poses using PoseBusters')
    parser.add_argument('--versatility_analysis', action='store_true', help='Whether to use the versatility analysis to check the performance of the MolDockLab workflow')


    return parser

VERBOSE = get_parser().parse_args().verbose

def main(args):

    HERE = Path(__file__).resolve().parent
    OUTPUT = HERE / args.out_dir
    
    OUTPUT.mkdir(exist_ok=True, parents=True)
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    
    if args.protein_name is None:
        protein_name = Path(args.protein_path).stem
    else:
        protein_name = args.protein_name
    
    if args.n_cpus is None:
        n_cpu = os.cpu_count() - 2
    else:
        n_cpu = args.n_cpus

    logger.info("\n\n🏁 Starting MolDockLab workflow 🏁\n\n")
    logger.info("🔷 Preparing the experimentally validated ligands library for docking using Gypsum-DL 1.2.1 ⏳")
    output_prepared_mols = OUTPUT / f"{Path(args.known_ligands_path).stem}_prepared.sdf"
    try:
        run_gypsumdl(
            ligand_library = (HERE / args.known_ligands_path), 
            prepared_library_path=output_prepared_mols, 
            id_column=args.id_col
            )
        logger.info(f"✅ Experimentally validated ligands library is prepared at {output_prepared_mols}")
    except Exception as e:
        logger.error(f"❗An error occured while preparing the ligands library for docking: {e}")
        return
    
    logger.info(f"🔷 Docking the experimentally validated ligands library with experimental data using {args.docking_programs}⏳⏳")
    try:
        if 'all' in args.docking_programs:
            
            args.docking_programs = ALLOWED_DOCKING_PROGRAMS
            print(args.docking_programs)
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
        logger.info(f"✅ Docking results of experimentally validated molecules are saved at {OUTPUT / 'allposes.sdf'}")
    except Exception as e:
        logger.error(f"❗An error occured while docking the ligands library: {e}")
        return
    
    if args.pose_quality_checker:
        logger.info("🔷 Checking the quality of the generated poses for experimentally validated ligands using PoseBusters ⏳")
        _docking_tools_results = poses_checker(
            poses_path= OUTPUT / 'allposes.sdf', 
            protein_path= HERE / args.protein_path, 
            output_file= OUTPUT / f'{protein_name}_posebusters_results.csv'
            )
        logger.info(f"✅ PoseBusters results are saved at {OUTPUT / f'{protein_name}_posebusters_results.csv'}")

    logger.info(f"🔷 Rescoring the docking results for experimentally validated ligands using SFs: {args.rescoring} ⏳⏳")
    try:
        if 'all' in args.rescoring:
            args.rescoring = ALLOWED_SCORING_FUNCTIONS
        rescoring_function(
            rescoring_programs=args.rescoring, 
            protein_path=HERE / args.protein_path,
            docked_library_path=OUTPUT / 'allposes.sdf', 
            ref_file=HERE / args.ref_ligand_path,
            ncpu=n_cpu,
            )
        logger.info(f"✅ Rescoring results are saved at {OUTPUT / 'rescoring_results' / 'all_rescoring_results.csv'}")
    except Exception as e:
        logger.error(f"❗An error occured while rescoring the docking results: {e}")
        return
    
    logger.info("🔷 Merging the activity values with SFs results of experimentally validated ligands ...")
    try:
        df_rescored_merged = merge_activity_values(
            norm_scored_path=OUTPUT / 'rescoring_results' / 'all_rescoring_results.csv',
            mols_true_value_path=(HERE / args.known_ligands_path), 
            true_value_col=args.true_value_col,
            scored_id_col=args.id_col,
            lower_better_true_value=args.true_value_scale,
            activity_col=args.activity_col,
            threshold=args.corr_threshold
            )
        logger.info(f"✅ Merged results are saved at {OUTPUT / 'rescoring_results' / 'all_rescoring_results_merged.csv'}")
    except Exception as e:
        logger.error(f"❗An error occured while merging the activity values with the docking results: {e}")
        return
    
    try:
        logger.info("🔷 Normalizing the predicted scores ...")
        df_rescored_norm = norm_scores(df_rescored_merged)
        df_rescored_norm.to_csv(str(OUTPUT / 'rescoring_results' / 'all_rescoring_results_merged_norm_robust.csv'), index=False)
        plt.figure(figsize=(20, 9))
        sns.boxplot(data=df_rescored_norm.drop(columns=['true_value']) ,linewidth=.85, showfliers=False)
        plt.ylim(-1, 1)
        plt.title('Box Plot of Normalized Predicted Scores of the Docking Poses')
        plt.xticks(rotation=45) 
        # save the plot
        plt.savefig(OUTPUT / 'normalized_scores_boxplot.png')
        logger.info(f"✅ Normalized scores boxplot is saved at {OUTPUT / 'normalized_scores_boxplot.png'}")
    except Exception as e:
        logger.error(f"❗An error occured while normalizing the predicted scores: {e}")
        return
    
    if 'all' in args.ranking_method:
        args.ranking_method = ALLOWED_RANKING_METHODS

    if 'weighted_ECR' in args.ranking_method:
        try:
            logger.info("🔷 Performing the pose score optimization for experimentally validated ligands ⏳")
            X, y, docking_cost, scoring_cost, docking_tools, scoring_tools = scores_preprocessing(
                df_rescored_merged)
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
            logger.info(f"✅ Best weights are saved at {HERE / 'test_data/best_weights.pkl.npy'}")
        except Exception as e:
            logger.error(f"❗An error occured while performing the pose score optimization: {e}")
            return
        
        try:
            logger.info("🔷 Normalize the optimized weights to the docking and scoring tools ...")
            for alpha in best_weights.keys():
                normalized_weights = mapping_normalized_weights(best_weights[alpha], scoring_tools, docking_tools)
        except Exception as e:
            logger.error(f"❗An error occured while normalizing the optimized weights to the docking and scoring tools: {e}")
            return
        
    try:
        logger.info(f"🔷 Ranking the experimentally validated ligands library selecting {args.ranking_method}⏳")
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
        logger.info(f"✅ Ranked ligands are saved at {OUTPUT / 'correlations_general' /  'all_ranked.csv'}")
    except Exception as e:
        logger.error(f"❗An error occured while ranking the ligands library: {e}")
        return
    
    # selecting best balanced pipeline
    try:
        logger.info("🔷 Selecting the best balanced pipeline for experimentally validated ligands ...")
        corr_df = pd.read_csv(OUTPUT / 'correlations_general' /  'all_ranked.csv')
        range_workflows = corr_df[(corr_df['spearman_correlation'] >= corr_df['spearman_correlation'].max() - args.corr_range) &
                          (corr_df['enrichment_factor'] >= corr_df.loc[0, 'enrichment_factor'] - args.ef_range)]
        
        # select row with the minimum cost value
        selected_workflow = range_workflows.loc[range_workflows['cost_per_pipeline'].idxmin()]
        logger.info(
            f"🕵️‍♂️ The best balanced pipeline uses for docking: {selected_workflow['docking_tool']}\n"
            f"\tand for SF(s): {selected_workflow['scoring_function']}\n" 
            f"\twith a Spearman correlation of {selected_workflow['spearman_correlation']}\n"
            f"\tand an enrichment factor of {selected_workflow['enrichment_factor']}\n"
            f"\ta cost of {selected_workflow['cost_per_pipeline']}\n")
        
        selected_docking_tools = ast.literal_eval(selected_workflow['docking_tool'])
        selected_sfs = ast.literal_eval(selected_workflow['scoring_function'])
        selected_ranking_method = selected_workflow['ranking_method']
        saved_time =  corr_df.loc[0, 'cost_per_pipeline'] - selected_workflow['cost_per_pipeline']
        logger.info(f"✅ By selecting the best balanced pipeline, you saved {saved_time} seconds per compound.🚀🚀")
    except Exception as e:
        logger.error(f"❗An error occured while selecting the best balanced pipeline: {e}")
        return

    # screen the larger ligands library for SBVS
    logger.info("Screening the larger ligands library for SBVS ⏳⏳⏳")
    larger_data_output = OUTPUT / Path(args.sbvs_ligands_path).stem
    larger_data_output.mkdir(exist_ok=True, parents=True)

    logger.info("🔷 Preparing the larger ligands library for docking using Gypsum-DL 1.2.1 ⏳")
    output_prepared_mols = OUTPUT / f"{Path(args.sbvs_ligands_path).stem}_prepared.sdf"
    try:
        run_gypsumdl(
            ligand_library = (HERE / args.sbvs_ligands_path), 
            prepared_library_path=output_prepared_mols, 
            id_column=args.id_col
            )
        logger.info(f"✅ Larger ligands library is prepared at {output_prepared_mols}")
    except Exception as e:
        logger.error(f"❗An error occured while preparing the larger ligands library for docking: {e}")
        return
    
    logger.info(f"🔷 Docking the larger ligands library with unknown experimental data using {selected_workflow['docking_tool']}⏳⏳")
    try:
        if not (larger_data_output / 'allposes.sdf').is_file():

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
        logger.info(f"✅ Docking results are saved at {larger_data_output / 'allposes.sdf'}")
    except Exception as e:
        logger.error(f"❗An error occured while docking the larger ligands library: {e}")
        return
    
    logger.info(f"🔷 Rescoring the docking results of unknown data using selected SFs {selected_workflow['scoring_function']} ⏳⏳")
    try:
        rescoring_function(
            rescoring_programs=selected_sfs, 
            protein_path=HERE / args.protein_path,
            docked_library_path=larger_data_output / 'allposes.sdf', 
            ref_file=HERE / args.ref_ligand_path,
            ncpu=n_cpu,
            )
        logger.info(f"✅ Rescoring results are saved at {larger_data_output / 'rescoring_results' / 'all_rescoring_results.csv'}")
    except Exception as e:
        logger.error(f"❗An error occured while rescoring the docking results: {e}")
        return

    logger.info(f"🔷 Ranking unknown poses using {selected_ranking_method} ...")
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
        if selected_ranking_method == 'method9_weighted_ECR_best':
            ranked_sbvs_ligands = ranking_methods_dict[selected_ranking_method](
                df=rescored_df_sbvs_norm,
                selected_scores=selected_sfs,
                id_column=args.id_col,
                mapped_weights= normalized_weights,
                )
        else:
            ranked_sbvs_ligands = ranking_methods_dict[selected_ranking_method](
                df=rescored_df_sbvs_norm, 
                selected_scores=selected_sfs,
                id_column=args.id_col,
                weight=0.05,
                )
        ranked_sbvs_ligands.sort_values(by=selected_ranking_method, ascending=False, inplace=True)
        ranked_sbvs_ligands.rename(columns={'ID': 'full_ID'}, inplace=True)
        ranked_sbvs_ligands['ID'] = ranked_sbvs_ligands.full_ID.str.split('_').str[0]
        ranked_sbvs_ligands.drop_duplicates(subset='ID', keep='first', inplace=True)
        ranked_sbvs_ligands.to_csv(larger_data_output / 'ranked_ligands.csv', index=False)
        logger.info(f"✅ Ranked unknown ligands are saved at {larger_data_output / 'ranked_ligands.csv'}")
        
    except Exception as e:
        logger.error(f"❗An error occured while ranking the ligands library: {e}")
        return

    # Interaction analysis
    logger.info("🔷 Performing the interaction analysis using PLIPify ⏳⏳")
    try:
        if args.key_residues is None:
            actives_path = actives_extraction( 
                OUTPUT / 'allposes.sdf', 
                OUTPUT / 'rescoring_results/all_rescoring_results_merged.csv', 
                docking_tool=selected_docking_tools
                )
            actives_paths = split_sdf_path(actives_path)
            os.remove(actives_path)
            for chain in args.interacting_chains:
                interx_csv = OUTPUT / f'{protein_name}_{chain}_interx.csv'
                if interx_csv.is_file():
                    fp_focused = pd.read_csv(interx_csv)
                    continue
                fp_focused = plipify_fp_interaction(
                    ligands_path=actives_paths, 
                    protein_path=HERE / args.protein_path, 
                    protein_name=protein_name, 
                    chains=chain,
                    output_file=OUTPUT / f'{protein_name}_interactions_{chain}.png'
                    )
                fp_focused['total_interactions'] = fp_focused.sum(axis=1)
                fp_focused.to_csv(interx_csv, index_label='residues')
                
            logger.info(f"✅ Protein-ligand interactions with chain {chain} are saved at {interx_csv}")
            for chain in args.interacting_chains:
                    fp_interx = pd.read_csv(interx_csv).sort_values(by='total_interactions', ascending=False)
                    key_interactions_resno = list(fp_interx.head(4).residues)
                    key_interactions_resno = [f'{resno}{chain}' for resno in key_interactions_resno]
        else:
            key_interactions_resno = args.key_residues
        logger.info(f"🔑 Key interactions for chain {args.interacting_chains} with residues are: {key_interactions_resno}")
    except Exception as e:
        logger.error(f"❗An error occured while performing the interaction analysis: {e}")
        return
    
    logger.info("🔷 Filtering compounds with specific interactions ⏳⏳")
    try:
        interactions_dict_path = larger_data_output / 'fp_allposes.json'
        selected_ligands_interx = larger_data_output / 'selected_ligands_interaction.csv'
        if selected_ligands_interx.is_file():
            logger.info(f"✅ Selected ligands with specific interactions are already saved at {selected_ligands_interx}")
        else:
            if not interactions_dict_path.is_file():
                ligands_paths = split_sdf_path(larger_data_output / 'allposes.sdf')
                allposes_interaction_fp = indiviudal_interaction_fp_generator(
                                    sdfs_path=ligands_paths, 
                                    protein_file=args.protein_path, 
                                    protein_name=protein_name, 
                                    included_chains=args.interacting_chains, 
                                    output_dir=interactions_dict_path
                                    )
            interactions_df = read_interactions_json(
                        json_file=interactions_dict_path, 
                        output_file=larger_data_output / 'allposes_interaction_fps_final.csv'
                        )
            agg_interx_df = interactions_aggregation(
                                interactions_df=interactions_df.reset_index(),
                                important_interactions=key_interactions_resno,
                                )
            agg_interx_df.replace(0, np.nan, inplace=True)
            agg_interx_df.dropna(inplace=True)
            agg_interx_df.to_csv(larger_data_output / 'selected_ligands_interaction.csv', index=False)
        
        logger.info(f"✅ Selected ligands with specific interactions are saved at {larger_data_output / 'selected_ligands_interaction.csv'}")
    except Exception as e:
        logger.error(f"❗An error occured while filtering compounds with specific interactions: {e}")
        return
        
    logger.info("🔷 Concatenating the selected ligands from the interaction analysis with the ranked ligands ...")
    try:
        ranked_ligands = pd.read_csv(larger_data_output / 'ranked_ligands.csv')
        selected_ligands = pd.read_csv(larger_data_output / 'selected_ligands_interaction.csv').rename(columns={'id': 'ID'})
        selected_ligands['passed_interx_filtration'] = 1
        merged_df = pd.merge(ranked_ligands, selected_ligands[['ID', 'passed_interx_filtration']], how='left').fillna(0)
        merged_df.to_csv(larger_data_output / 'ranked_selected_interx_ligands.csv', index=False)
        logger.info(f"✅ All ligands are saved at {larger_data_output / 'ranked_selected_interx_ligands.csv'}")
    except Exception as e:
        logger.error(f"❗An error occured while concatenating the selected ligands from the interaction analysis with the ranked ligands: {e}")
        return
    

    # @TODO : Add the diversity selection step and visualize the selected compounds
    if args.diversity_selection:
        logger.info("🔷 Select the most diverse number of compounds ...")
        try:
            if merged_df is None:
                merged_df = pd.read_csv(larger_data_output / 'ranked_selected_interx_ligands.csv')
            clustered_df = diversity_selection(ranked_interx_df=merged_df,
                                            sdf= HERE / args.sbvs_ligands_path,
                                            id_col=args.id_col,
                                            n_clusters=args.n_clusters
                                            )
            selected_diverse = clustered_df[clustered_df['diversity_selection'] == 1]

            # Visualize the selected compounds in a high quality photo
            options = Draw.MolDrawOptions()
            options.legendFontSize = 25  
            options.atomLabelFontSize = 20 
            drawer = Draw.MolDraw2DSVG(3*500, 3*500, 500, 500)  
            drawer.SetDrawOptions(options)

            smiles = [Chem.MolToSmiles(mol) for mol in selected_diverse['ROMol']]
            img = Draw.MolsToGridImage(
                [Chem.MolFromSmiles(smile) for smile in smiles], 
                molsPerRow=5, 
                subImgSize=(1000, 1000),
                legends=[f"{row['ID']}" for idx, row in selected_diverse.iterrows()],
                drawOptions=drawer.drawOptions()
            )

            # Save the image to a file
            img.save(larger_data_output / "final_compound_selection.png")
            PandasTools.WriteSDF(selected_diverse, larger_data_output / 'Final_compounds_selection.sdf', molColName='ROMol', idName='ID', properties=list(selected_diverse.columns))
            logger.info(f"✅ Selected diverse ligands are saved at {larger_data_output / 'Final_compounds_selection.csv'}")
            logger.info(f"The selected compounds are visualized in {larger_data_output / 'final_compound_selection.png'}")
        except Exception as e:
            logger.error(f"❗An error occured while selecting the most diverse number of compounds: {e}")
            return
    logger.info("\n\n🏁 MolDockLab workflow is completed successfully 🏁\n\n")
    
    # @TODO : Add versatility analysis step

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
    #         logger.error(f"❗An error occured while performing the versatility analysis: {e}")
    #         return
        

if __name__ == "__main__":
    args = get_parser().parse_args()

    validate_docking_programs(args.docking_programs)
    validate_scoring_functions(args.rescoring)
    validate_ranking_methods(args.ranking_method)
    VERBOSE = args.verbose
    main(args)