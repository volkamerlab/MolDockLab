import os
import csv
from rdkit import Chem
from rdkit.Chem import PandasTools
import pandas as pd
from rdkit import Chem
from scipy.stats import spearmanr
from matplotlib import pyplot as plt
from rdkit import RDLogger
from DockM8.scripts import consensus_methods


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
def common_molecules(df_IC50, df_scores):
    
    common_molecules = df_scores[df_scores['HIPS code'].isin(df_IC50['HIPS code'])]['HIPS code']

def rank_correlation(results_path, common_ID):
    '''''
    It's a function that evaluates smina and gnina docking tools. It calculates pearson's correlation and spearsman correlation.
    Dataframe has to be output from docking tool contains the true score of ranking and predicted score.
    @Param:
    results_path : Path of SDF file which contains the true value column called "Activity"
    
    @Return:
    Draw a plot that has spearman and pearson correlation
    '''''
    RDLogger.DisableLog('rdApp.*')  
    # with open(subprocess.DEVNULL, 'w') as devnull:
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
    #print(pearson_corr)

    # Specify different colors for the differentiated points
    score_of_commonIDs = docked_df.loc[docked_df['ID'].isin(common_ID), 'Activity'].to_list()
    predicted_of_commonIDs = docked_df.loc[docked_df['ID'].isin(common_ID), predicted_score].to_list()



    # plt.scatter(docked_df['true rank'], docked_df['docked rank'], label='All Points')
    
    # plt.xlabel('True rank')
    # plt.ylabel('Docked rank')
    # plt.title(f'{docked_method.upper()} {scoring_method.upper()}, snapshot A\n Pearson corr = {pearson_corr:.4f}\n Spearman corr = {spearman_corr:.4f}')
    # plt.legend()
    # plt.show()

    # Find Pearson and spearsman correlation between true and predicted values
    pearson_corr = docked_df['Activity'].corr(docked_df[predicted_score])
    spearman_corr = spearmanr(docked_df['Activity'], docked_df[predicted_score])[0]
    
    plt.figure(figsize=(10,8))	

    plt.scatter(docked_df['Activity'], docked_df[predicted_score], c='blue', label='Calculated Scores')
    plt.scatter(score_of_commonIDs, predicted_of_commonIDs, c='red', label='IC50')

    plt.xlabel('Score')
    plt.ylabel('Predicted Affinity')
    plt.title(f'{docked_method.upper()} {scoring_method.upper()}, snapshot A\n Pearson corr = {pearson_corr:.4f}\n Spearman corr = {spearman_corr:.4f}')
    plt.legend()
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


def prcoess_ranking_data(df, true_df, method):
    '''''
    @Param : df = gnerated df from DockM8, true_df = the first input, method = the type of the metric used
    ----------------
    @Return : dataframe have the true score , HIPS codes and the consensus score 
    '''''
    df = df.sort_values(by=[method]).drop_duplicates(subset="ID")

    #Add info to stardrop ID first
    df_stardrop = df[df['ID'].str.startswith('StarDrop')].sort_values(by = "ID")
    filtered_df = true_df[true_df['ID'].isin(df_stardrop['ID'])].sort_values(by = "ID")

    df_stardrop['ID'] = filtered_df['HIPS code'].to_list()
    df_stardrop['score'] = filtered_df['score'].to_list()

    #Add info to HIPS ID second
    df_HIPS = df[df['ID'].str.startswith('HIPS')].sort_values(by = "ID")
    filtered_df2 = true_df[true_df['HIPS code'].isin(df_HIPS['ID'])].sort_values(by = "HIPS code")
    df_HIPS['score'] = filtered_df2['score'].to_list()

    #Merge both again
    merged_df = pd.concat([df_HIPS, df_stardrop]).drop_duplicates('ID')
    merged_df[[method, 'score']] = merged_df[[method, 'score']].apply(pd.to_numeric)

    return merged_df

def correlation_mapping(df, common_ID):

        method_name = df.columns[1]
        print(method_name)

        score_of_commonIDs = df.loc[df['ID'].isin(common_ID), 'score'].to_list()
        predicted_of_commonIDs = df.loc[df['ID'].isin(common_ID), method_name].to_list()
        spearman_corr = spearmanr(df['score'], df[method_name])[0]
        pearson_corr = df['score'].corr(df[method_name])

        plt.figure(figsize=(10,8))	

        plt.scatter(df['score'], df[method_name], c='blue', label='Calculated Scores')
        plt.scatter(score_of_commonIDs, predicted_of_commonIDs, c='red', label='IC50')
        plt.xlabel('Scores')
        plt.ylabel('Predicted Affinity')
        plt.title(f'DockM8 {method_name}, snapshot A\n Pearson corr = {pearson_corr:.4f}\n Spearman corr = {spearman_corr:.4f}')
        plt.legend()
        plt.show()

        
def consensus_ranking_generator(cons_method, common_ID):
    '''''
    This function takes the output from ranking step in DockM8 tool and assign ECR scores according to DockM8 tools.
    @Param : cons_method --> Consensus method will be used accordingly

    @Output is the calcuated correlation between true and claculated scores and scatter plot
    '''''
    true_df = PandasTools.LoadSDF('data/ligands/ecft_scores.sdf', idName="ID")[['ID','HIPS code', 'score']]
    true_df['old rank'] = true_df['score'].apply(pd.to_numeric).rank(method='min')
    for f in os.listdir('data/A/dockm8/ranking/'):
        df = pd.read_csv('data/A/dockm8/ranking/'+f)
        selected_col = df.columns[1:-1]

        metric = f.split('.')[0]


        if cons_method == 'method1':
            cons_df = consensus_methods.method1_ECR_best(df, metric, selected_col)
            cons_metric= f'Method1_ECR_{metric}'
        elif cons_method == 'method2':
            cons_df = consensus_methods.method2_ECR_average(df, metric, selected_col)
            cons_metric= f'Method2_ECR_{metric}'
        elif cons_method == 'method6':
            cons_df = consensus_methods.method6_Zscore_best(df, metric, selected_col)
            cons_metric= f'Method6_Zscore_{metric}'

        cons_df = prcoess_ranking_data(cons_df,true_df, cons_metric)

        correlation_mapping(cons_df, common_ID)

