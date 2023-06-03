import csv
from rdkit import Chem
from rdkit.Chem import PandasTools
import pandas as pd
from rdkit import Chem
from scipy.stats import spearmanr
from matplotlib import pyplot as plt

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


def rank_correlation(results_path):
    '''''
    It's a function that evaluates smina and gnina docking tools. It calculates pearson's correlation and spearsman correlation.
    Dataframe has to be output from docking tool contains the true score of ranking and predicted score.
    @Param:
    results_path : Path of SDF file which contains the true value column called "Activity"
    
    @Return:
    Draw a plot that has both correlations
    '''''
    
    docked_df = PandasTools.LoadSDF(results_path, idName='ID', molColName='Molecule', strictParsing=False)
    docked_method = results_path.split('_')[1]
    scoring_method = results_path.split('_')[2]

    if "score" in docked_df.columns:
        docked_df.rename(columns = {'score':'Activity'}, inplace = True)


    if scoring_method == "rf-score-vs":
        predicted_score = "RFScoreVS_v2"
        arrange = False

    elif scoring_method in ['vinardo', 'ad4']:
        predicted_score = "minimizedAffinity"
        arrange = False

    else:
        if docked_method == 'smina':
            predicted_score = "minimizedAffinity"

            arrange = True
        elif docked_method == 'gnina':
            predicted_score = "CNNaffinity"
            arrange = True
        scoring_method = ''

    # Keep best predicted affinity for every compound.
    docked_df[[predicted_score, 'Activity']] = docked_df[[predicted_score, 'Activity']].apply(pd.to_numeric)
    docked_df = docked_df.sort_values(predicted_score).drop_duplicates(subset="ID")

    # Find Pearson and spearsman correlation between ranks of true and predicted values
    docked_df['docked rank'] = docked_df[predicted_score].rank(ascending=arrange)
    docked_df['true rank'] = docked_df['Activity'].rank()

    spearman_corr = spearmanr(docked_df['true rank'], docked_df['docked rank'])[0]
    pearson_corr = docked_df['true rank'].corr(docked_df['docked rank'])

    # correlation_matrix = docked_df[['true rank', 'docked rank']].corr(method='pearson')
    # print(correlation_matrix)
    print(pearson_corr)

    plt.scatter(docked_df['true rank'], docked_df['docked rank'])
    plt.xlabel('True rank')
    plt.ylabel('Docked rank')
    plt.title(f'{docked_method.upper()} {scoring_method.upper()}, snapshot A\n Pearson corr = {pearson_corr:.4f}\n Spearman corr = {spearman_corr:.4f}')
    plt.show()

    # Find Pearson and spearsman correlation between true and predicted values
    pearson_corr = docked_df['Activity'].corr(docked_df[predicted_score])
    # correlation_matrix = docked_df[['Activity', predicted_score]].corr(method='pearson')
    # print(correlation_matrix)
    #print(pearson_corr)

    spearman_corr = spearmanr(docked_df['Activity'], docked_df[predicted_score])[0]

    plt.scatter(docked_df['Activity'], docked_df[predicted_score])
    plt.xlabel('IC50')
    plt.ylabel('Predicted Affinity')
    plt.title(f'{docked_method.upper()} {scoring_method.upper()}, snapshot A\n Pearson corr = {pearson_corr:.4f}\n Spearman corr = {spearman_corr:.4f}')
    plt.show()


def prepare_df_for_comparison(results_path, ligand_library):

    '''''
    This function takes a path of gnina results and path of true molecules and returns 
    output a dataframe with true scores the HIPS code for every ID and predicted affinity.

    @Param : 
    results_path --> output file path of docking tool e.g.(SMINA or GNINA)
    ligand_library --> input data with true score values and true ID

    @Return:
    dataframe contains HIPS code as ID, true and predicted scores. 
    '''''
    docked_df = PandasTools.LoadSDF(results_path, idName='ID', molColName='Molecule', strictParsing=False)
    if all(docked_df[docked_df['ID'].str.startswith('StarDrop')]):
        docked_df = PandasTools.LoadSDF(results_path, idName='ID', molColName='Molecule', strictParsing=False)
        docked_df['ID'] = docked_df['ID'].str.split('_').str[0]
    true_df = PandasTools.LoadSDF(ligand_library, idName='ID', molColName='Molecule', strictParsing=False)[['HIPS code', 'ID', 'score']]
    display(docked_df)
    merged_df = pd.merge(docked_df, true_df, on='ID').drop('ID', axis=1)
    merged_df.rename(columns = {'score':'Activity', 'HIPS code':'ID'}, inplace = True)
    return merged_df