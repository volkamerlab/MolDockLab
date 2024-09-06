import os
import hdbscan
import numpy as np
import pandas as pd
import sklearn.model_selection as skl_model_sel

from pathlib import Path
from rdkit import Chem
from rdkit.Chem import PandasTools
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.preprocessing import RobustScaler
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator

from src.utilities import handling_multicollinearity, run_command


def  merge_activity_values(
    norm_scored_path : Path,
    mols_true_value_path : Path,
    true_value_col : str,
    scored_id_col : str,
    activity_col : str = "activity_class",
    lower_better_true_value : bool = False,
    threshold : float = 0.9
    ) -> pd.DataFrame:
    """
    Merging the experimental values and the activity class to the normalized scores of scoring function

    Args:
        norm_scored_path (pathlib.Path) : Path of normalized scores in csv format with ID contains name
                of the compound, docking tool and number of the pose, e.g. XXXX_plants_01
        mols_true_value_path (pathlib.Path) : Path of true value in SDF format with the unique ID
        true_value_col (str) : column name of experimental value
        scored_id_col (str) : column name of ID
        activity_col (str) : column name of activity class
        lower_better_true_value (bool) : True if the lower value of the experimental value is better (IC50)
        threshold (float) : correlation threshold

    Return:
        DataFrame with experimental values and activity classes aligned with scores from scoring functions
        additionally it contains the number of poses and docking tools
    """
    df_rescored = pd.read_csv(
        str(norm_scored_path))
    for col in df_rescored.columns:
        try:
            df_rescored[col] = pd.to_numeric(df_rescored[col])
        except:
            continue
    df_rescored[['id', 'docking_tool', 'pose']
                ] = df_rescored[scored_id_col].str.split('_', expand=True)
    true_values_df = PandasTools.LoadSDF(str(mols_true_value_path))

    for _, group in df_rescored.groupby(['id']):
        group.loc[:, true_value_col] = true_values_df[true_values_df['ID']
                                                      == group['id'].iloc[0]][true_value_col].values[0]
        group.loc[:, activity_col] = true_values_df[true_values_df['ID']
                                                    == group['id'].iloc[0]][activity_col].values[0]

        df_rescored.loc[group.index, true_value_col] = group[true_value_col].values[0]
        df_rescored.loc[group.index, activity_col] = group[activity_col].values[0]
    if lower_better_true_value:
        df_rescored.loc[:,true_value_col] = df_rescored.loc[:,true_value_col] * -1
    df_rescored.drop(['pose'], axis=1, inplace=True)
    df_rescored.rename(columns={true_value_col: 'true_value'}, inplace=True)

    collinear_sfs = handling_multicollinearity(
        df_rescored.drop([activity_col], axis=1),
        threshold=threshold,
        true_value_col='true_value'
    )
    df_rescored.drop(collinear_sfs, axis=1, inplace=True)
    df_rescored.to_csv(str(norm_scored_path.parent / 'all_rescoring_results_merged.csv'), index=False)
    return df_rescored


def _get_scaffold(mol : Chem.Mol) ->Chem.rdchem.Mol:
    """
    Get the Murcko scaffold of a molecule
    Args:
        mol (rdkit.Chem.rdchem.Mol) : RDKit molecule object
    Return:
        rdkit.Chem.rdchem.Mol : RDKit molecule object of the Murcko scaffold
    """
    return MurckoScaffold.GetScaffoldForMol(mol)


def get_fp(scaffold : Chem.Mol) -> np.ndarray:
    """
    Get the Morgan fingerprint of a molecule
    Args:
        scaffold (rdkit.Chem.rdchem.Mol) : RDKit molecule object
    Return:
        numpy.ndarray : Morgan fingerprint of the molecule
    """
    morgan_fp_gen = GetMorganGenerator(radius=2, fpSize=2048)
    fp = morgan_fp_gen.GetFingerprint(scaffold)
    return fp


def _get_cluster_labels(scaffold_fps : list, min_cluster_size : int=2) -> np.ndarray:
    """
    Cluster the molecules based on their scaffold fingerprints
    Args:
        scaffold_fps (list) : list of Morgan fingerprints
        min_cluster_size (int) : minimum number of molecules in a cluster
    Returns:
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
    Returns:
        Dataframe : DataFrame with the original dataset and the cluster labels
    """

    df = PandasTools.LoadSDF(str(original_data_path))
    df['scaffold'] = df.ROMol.apply(_get_scaffold)

    unique_scaffolds = list(set(df['scaffold'].apply(Chem.MolToSmiles)))

    print(f'Number of unique scaffolds: {len(unique_scaffolds)}')
    df['scaffold_fp'] = df.scaffold.apply(_get_fp)
    cluster_labels = _get_cluster_labels(
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
        ) -> tuple:
    """
    Convert the protein, molecules library and reference file to MOL2 format
    Args:
        protein_file (pathlib.Path) : path of the protein file
        molecules_library (pathlib.Path) : path of the molecules library file
        ref_file (pathlib.Path) : path of the reference file
    Return:
        protein_file, molecules_library, ref_file paths in MOL2 format
    """
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
            print(f"\t\t{file.stem} is already converted to MOL2 format")

    return (protein_file.with_suffix(".mol2"), 
            molecules_library.with_suffix(".mol2"), 
            ref_file.with_suffix(".mol2"))

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
        'vina_intra_hydrophobic'
        ],
) -> pd.DataFrame:
    """
    Normalize a DataFrame that has an ID in the first column numerical values of scoring functions.
    Certain specified columns will have their scaling inversed ('the lower the better').

    HYDE SF is treated differently due to its wide range of values. 
    A hard cutoff of 10000 is set and ranks of the molecules is added to the original value.

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
    scaled_numeric = scaler.fit_transform(df_copy[numeric_cols])
    scaled_numeric_df = pd.DataFrame(
        scaled_numeric,
        columns=numeric_cols,
        index=df_copy.index)
    df_copy.update(scaled_numeric_df)

    return df_copy