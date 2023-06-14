from rdkit.Chem import Draw
from rdkit.Chem import rdFMCS
import pandas as pd
from rdkit import Chem
from rdkit.Chem import PandasTools

def find_MCS(df_scores, df_IC50, activity_threshold, sort_order):
    '''''
    This function gives the Most common substructure of input data that has calculated scores and IC50.

    @Param:
    df_scores : dataframe that has calculated scores

    df_IC50 : dataframe that has IC50

    activity_threshold : cutoff of IC50 values of included data to get MCS.

    sort_order : How data is sorted according to IC50 or score

    @Return:
    MCS of structures within given threshold and draw it.
    '''''

    if sort_order == 'IC50':
        sort_order = 'Activity'

    merged_df = pd.merge(df_scores, df_IC50, on="HIPS code", how="left")
    merged_df[['score', 'Activity']] = merged_df[['score', 'Activity']].apply(pd.to_numeric)
    merged_df = merged_df.sort_values(by=sort_order).reset_index()

    #Set cutoff to determined threshold
    cutoff = merged_df['Activity'] > activity_threshold

    #determine first occurance of exceeding threshold and stop.
    first_occurrence_index = merged_df.index[cutoff].min()
    cutoff_data = merged_df.loc[:first_occurrence_index-1]
    
    PandasTools.WriteSDF(cutoff_data, f'data/A/cutoff20.sdf', idName='HIPS code', molColName='ROMol', properties=cutoff_data.columns)

    mcs = rdFMCS.FindMCS(cutoff_data.ROMol)
    
    m1 = Chem.MolFromSmarts(mcs.smartsString)
    Draw.MolToImage(m1, legend=f"{sort_order} threshold {activity_threshold}")
    
    return mcs


def write_mcs_file(df, mcs, data_name):
    
    def has_scaffold(mol):
        return mol.HasSubstructMatch(mcs)

    IC50_df = df[df['ROMol'].apply(has_scaffold)][['HIPS code', 'ROMol', 'score']]
    PandasTools.WriteSDF(IC50_df, f'data/ligands/{data_name}.sdf',idName="HIPS code", molColName='ROMol', properties=IC50_df.columns)

    return data_name, IC50_df.shape[0]