import pandas as pd
import numpy as np

from pathlib import Path
from rdkit.Chem import PandasTools
from sklearn_extra.cluster import KMedoids

from src.preprocessing import get_fp

def diversity_selection(
        ranked_interx_df : pd.DataFrame, 
        sdf : Path, 
        id_col : str, 
        n_clusters : int = 5
        ) -> pd.DataFrame:
    """
    Function to select n diverse set of compounds from a ranked list of compounds

    Args:
        ranked_interx_df (pd.DataFrame) : ranked list of compounds with the interaction analysis
        sdf (Path) : path to the sdf file containing the molecules for SBVS
        id_col (str) : the column name of the id in the sdf file
        n_clusters (int) : number of clusters/compounds to select

    Returns:
        merged_df (pd.DataFrame) : ranked list of compounds with the interaction filtration 
        and diversity selection columns
    """
    mols_df = PandasTools.LoadSDF(str(sdf), idName=id_col)
    merged_df = pd.merge(
        left=ranked_interx_df, 
        right=mols_df[[id_col, 'ROMol']], 
        how='left', 
        left_on='ID', 
        right_on=id_col
        )
    print(merged_df)
    merged_df = merged_df[merged_df['passed_interx_filtration'] == 1].reset_index(drop=True)
    merged_df = merged_df.head(int(0.01*len(merged_df)))
    fps = np.array([np.array(list(get_fp(mol))) for mol in merged_df['ROMol']])
    kmedoids = KMedoids(n_clusters=n_clusters, metric='jaccard', random_state=42)
    _ = kmedoids.fit_predict(fps)
    medoid_indices = kmedoids.medoid_indices_
    merged_df['diversity_selection'] = 0
    merged_df.loc[medoid_indices, 'diversity_selection'] = 1
    return merged_df