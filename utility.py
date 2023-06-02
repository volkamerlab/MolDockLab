import csv
import os
from rdkit import Chem
from rdkit.Chem import PandasTools, SDMolSupplier, AllChem, rdFingerprintGenerator
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.ML.Cluster import Butina
import numpy as np

def prepare_diffdock_input(protein_path, ligand_path, output_path):

#Every line has path to same target and different smiles code.
    header = ['complex_name', 'protein_path', 'ligand_description', 'protein_sequence']
    df = PandasTools.LoadSDF(ligand_path, idName='ID', molColName="Molecule")
    smiles = [Chem.MolToSmiles(mol) for mol in df.Molecule]
    with open(output_path, 'w', newline='') as file:

        # Create the CSV writer object
        writer = csv.writer(file)

        # Write the header row
        writer.writerow(header)

        for i, mol in df.iterrows():
            writer.writerow(['', protein_path, smiles[i], ''])

    #print(f'input csv file for DiffDock is ready in {output_path}')
from scipy.stats import spearmanr
from matplotlib import pyplot as plt

def rank_correlation(results_path):
    '''''
    It's a function that evaluates smina and gnina docking tools. It calculates pearson's correlation and spearsman correlation.

    @Param:
    results_path : Path of SDF file which contains the true value column called "Activity"
    
    @Return:
    Draw a plot that has both correlations
    '''''
    
    docked_df = PandasTools.LoadSDF(results_path, idName='ID', molColName='Molecule', strictParsing=False)
    
    if "score" in docked_df.columns:
        docked_df.rename(columns = {'score':'Activity'}, inplace = True)

    if "CNNaffinity" in docked_df.columns:
        predicted_score = "CNNaffinity"
        dock_tool = "GNINA v 1.0"
        arrange = False
    elif "RFScoreVS_v2" in docked_df.columns:
        predicted_score = "RFScoreVS_v2"
        dock_tool = "RFScoreVS_v2"
        arrange = False
        
    elif "rfscore_v2" in docked_df.columns:
        predicted_score = "rfscore_v2"
        dock_tool = "RFScoreVS_v2"
        arrange = False
    elif "minimizedAffinity" in docked_df.columns:
        predicted_score = "minimizedAffinity"
        dock_tool = "SMINA"
        arrange = True

    # Keep best predicted affinity for every compound.
    docked_df[[predicted_score, 'Activity']] = docked_df[[predicted_score, 'Activity']].apply(pd.to_numeric)
    docked_df = docked_df.sort_values(predicted_score).drop_duplicates(subset="ID")

    # Find Pearson and spearsman correlation between ranks of true and predicted values
    docked_df['docked rank'] = docked_df[predicted_score].rank(ascending=arrange)
    docked_df['true rank'] = docked_df['Activity'].rank()

    spearman_corr = spearmanr(docked_df['true rank'], docked_df['docked rank'])[0]
    pearson_corr = docked_df['true rank'].corr(docked_df['docked rank'])

    correlation_matrix = docked_df[['true rank', 'docked rank']].corr(method='pearson')
    print(correlation_matrix)
    print(pearson_corr)

    plt.scatter(docked_df['true rank'], docked_df['docked rank'])
    plt.xlabel('True rank')
    plt.ylabel('Docked rank')
    plt.title(f'{dock_tool}, snapshot A\n Pearson corr = {pearson_corr:.4f}\n Spearman corr = {spearman_corr:.4f}')
    plt.show()

    # Find Pearson and spearsman correlation between true and predicted values
    pearson_corr = docked_df['Activity'].corr(docked_df[predicted_score])
    correlation_matrix = docked_df[['Activity', predicted_score]].corr(method='pearson')
    print(pearson_corr)
    print(correlation_matrix)
    spearman_corr = spearmanr(docked_df['Activity'], docked_df[predicted_score])[0]

    plt.scatter(docked_df['Activity'], docked_df[predicted_score])
    plt.xlabel('IC50')
    plt.ylabel('Predicted Affinity')
    plt.title(f'{dock_tool}, snapshot A\n Pearson corr = {pearson_corr:.4f}\n Spearman corr = {spearman_corr:.4f}')
    plt.show()


