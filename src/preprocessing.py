import os

import hdbscan
import numpy as np
import pandas as pd
import sklearn.model_selection as skl_model_sel
from rdkit import Chem
from rdkit.Chem import AllChem, PandasTools
from rdkit.Chem.Scaffolds import MurckoScaffold

from pathlib import Path
from src.ranking import norm_scores
from src.utilities import handling_multicollinearity, run_command


def merge_activity_values(
    norm_scored_path : Path,
    true_value_path : Path,
    true_value_col : str,
    scored_id_col : str,
    activity_col : str,
    lower_better_true_value : bool,
    threshold : float
    ) -> pd.DataFrame:
    """
    Merging the experimental values and the activity class to the normalized scores of scoring function

    Args:
        norm_scored_path (pathlib.Path) : Path of normalized scores in csv format with ID contains name
                of the compound, docking tool and number of the pose, e.g. XXXX_plants_01
        true_value_path (pathlib.Path) : Path of true value in SDF format with the unique ID
        true_value_col (str) : column name of experimental value
        unique_id_col (str) : column name of ID
        activity_col (str) : column name of activity class
        threshol (float) : correlation threshold

    Return:
        DataFrame with experimental values and activity classes aligned with scores from scoring functions
        additionally it contains the number of poses and docking tools
    """
    df_rescored = pd.read_csv(
        str(norm_scored_path)).apply(
        pd.to_numeric,
        errors='ignore')
    df_rescored[['id', 'docking_tool', 'pose']
                ] = df_rescored[scored_id_col].str.split('_', expand=True)
    true_values_df = PandasTools.LoadSDF(str(true_value_path))

    for _, group in df_rescored.groupby(['id']):
        group.loc[:, true_value_col] = true_values_df[true_values_df['ID']
                                                      == group['id'].iloc[0]][true_value_col].values[0]
        group.loc[:, activity_col] = true_values_df[true_values_df['ID']
                                                    == group['id'].iloc[0]][activity_col].values[0]

        df_rescored.loc[group.index,
                        true_value_col] = group[true_value_col].values[0]
        df_rescored.loc[group.index,
                        activity_col] = group[activity_col].values[0]
    if lower_better_true_value:
        df_rescored.loc[:,
                        true_value_col] = df_rescored.loc[:,
                                                          true_value_col] * -1
    df_rescored.drop(['pose'], axis=1, inplace=True)
    df_rescored.rename(columns={true_value_col: 'true_value'}, inplace=True)

    not_collinear_df = handling_multicollinearity(
        df_rescored,
        threshold=threshold,
        true_value_col='true_value'
    )

    not_collinear_df.to_csv(str(norm_scored_path.parent /
                                'all_rescoring_results_merged.csv'), index=False)
    return df_rescored


def get_scaffold(mol : rdkit.Chem.Mol) -> rdkit.Chem.rdchem.Mol:
    """
    Get the Murcko scaffold of a molecule
    Args:
        mol (rdkit.Chem.rdchem.Mol) : RDKit molecule object
    Return:
        rdkit.Chem.rdchem.Mol : RDKit molecule object of the Murcko scaffold
    """
    return MurckoScaffold.GetScaffoldForMol(mol)


def get_fp(scaffold : rdkit.Chem.Mol) -> np.ndarray:
    """
    Get the Morgan fingerprint of a molecule
    Args:
        scaffold (rdkit.Chem.rdchem.Mol) : RDKit molecule object
    Return:
        numpy.ndarray : Morgan fingerprint of the molecule
    """

    return np.array(
        AllChem.GetMorganFingerprintAsBitVect(
            scaffold, radius=2, nBits=2048))


def get_cluster_labels(scaffold_fps : list, min_cluster_size : int=2) -> np.ndarray:
    """
    Cluster the molecules based on their scaffold fingerprints
    Args:
        scaffold_fps (list) : list of Morgan fingerprints
        min_cluster_size (int) : minimum number of molecules in a cluster
    Return:
        numpy.ndarray : array of cluster labels
    """

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        metric='jaccard').fit(scaffold_fps)
    return clusterer.labels_


def hdbscan_scaffold_split(original_data_path : Path, min_cluster_size : int) -> pd.DataFrame:
    """
    Split the dataset based on the scaffold of the molecules
    Args:
        original_data_path (pathlib.Path) : path of the original dataset in SDF format
        min_cluster_size (int) : minimum number of molecules in a cluster
    Return:
        DataFrame : DataFrame with the original dataset and the cluster labels
    """

    df = PandasTools.LoadSDF(str(original_data_path))
    df['scaffold'] = df.ROMol.apply(get_scaffold)

    unique_scaffolds = list(set(df['scaffold'].apply(Chem.MolToSmiles)))

    print(f'Number of unique scaffolds: {len(unique_scaffolds)}')
    df['scaffold_fp'] = df.scaffold.apply(get_fp)
    cluster_labels = get_cluster_labels(
        list(df['scaffold_fp']), min_cluster_size)
    print(f'Number of HDBSCAN clusters: {len(set(cluster_labels))}')
    df['hdbscan_scaffold_cluster'] = cluster_labels

    return df


def cv_split(
        clustered_df : pd.DataFrame,
        df_rescored : pd.DataFrame,
        idx_col : str,
        n_splits : int,
        output_path : Path,
        target_name : str
        ):
    """
    Split the dataset based on the scaffold of the molecules
    Args:
        clustered_df (DataFrame) : DataFrame with the original dataset and the cluster labels
        df_rescored (DataFrame) : DataFrame with the normalized scores and the experimental values
        idx_col (str) : column name of ID
        n_splits (int) : number of splits
        output_path (pathlib.Path) : path of the output directory
        target_name (str) : name of the target
    Return:
        split the dataset into train and test sets and save them as csv files
    """

    output_path.mkdir(parents=True, exist_ok=True)

    split_cv = skl_model_sel.GroupKFold(n_splits=n_splits)
    train_test_ind = split_cv.split(
        clustered_df, groups=np.array(
            clustered_df['hdbscan_scaffold_cluster']))
    # df_scores = PandasTools.LoadSDF(str(ligand_library))[['ID', 'score']]
    clustered_df.rename(columns={'ID': idx_col}, inplace=True)
    df_rescored['id'] = df_rescored['ID'].str.split('_', expand=True)[0]

    for i, (train_idx, test_idx) in enumerate(train_test_ind):

        train_id = clustered_df.iloc[train_idx][idx_col]
        test_id = clustered_df.iloc[test_idx][idx_col]
        print(
            f"Train dataset has {len(train_id)} and test dataset has {len(test_id)}"
            )
        train_df = df_rescored[df_rescored['id'].isin(list(train_id))]
        test_df = df_rescored[df_rescored['id'].isin(list(test_id))]

        train_df_norm = norm_scores(train_df)
        test_df_norm = norm_scores(test_df)

        train_df_norm.to_csv(str(output_path /
                                 f'{target_name}_train_{i}.csv'), index=False)
        test_df_norm.to_csv(str(output_path /
                                f'{target_name}_test_{i}.csv'), index=False)


def plants_preprocessing(
        protein_file : Path, 
        molecules_library : Path, 
        ref_file : Path
        ):
    # Convert protein file to .mol2 using open babel
    print("PLANTS preprocessing is running ...\n\t Converting to Mol2")
    for file in [protein_file, molecules_library, ref_file]:

        converted_file = file.with_suffix(".mol2")
        if converted_file.name not in os.listdir(converted_file.parent):

            obabel_command = (
                f'obabel -i{str(file)[-3:]} {str(file)}'
                f' -O {str(converted_file)}'
            )
            run_command(obabel_command)
        else:
            print(f"{file.stem} is already converted to mol2 format")

    return protein_file.with_suffix(".mol2"), molecules_library.with_suffix(
        ".mol2"), ref_file.with_suffix(".mol2")
