import os
import re

import csv
from rdkit import Chem
from rdkit.Chem import PandasTools
import pandas as pd
from rdkit import Chem
from scipy.stats import spearmanr
from matplotlib import pyplot as plt
from rdkit import RDLogger
from DockM8.scripts import consensus_methods


def rank_correlation(results_path, common_ID):
    '''''
    Evaluates smina and gnina docking tools by calculating pearson and spearsman correlation and draw a scatter plot.


    @Param:

    results_path : Path of SDF file which contains the true value column called "Activity"
    
    common_ID : ID that used in score dataframe has IC50 values


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

    elif scoring_method in ['vinardo', 'ad4']:
        predicted_score = "minimizedAffinity"

    else:
        if docked_method == 'smina':
            predicted_score = "minimizedAffinity"

        elif docked_method == 'gnina':
            predicted_score = "CNNaffinity"

        elif docked_method == 'diffdock':
            predicted_score = 'confidence_score'
        scoring_method = ''


    plt.figure(figsize=(10,8))	


    # Keep best predicted affinity for every compound.
    docked_df[[predicted_score, 'Activity']] = docked_df[[predicted_score, 'Activity']].apply(pd.to_numeric)
    docked_df = docked_df.sort_values(predicted_score).drop_duplicates(subset="ID")

    # Specify different colors for the differentiated points
    score_of_commonIDs = docked_df.loc[docked_df['ID'].isin(common_ID), 'Activity'].to_list()
    predicted_of_commonIDs = docked_df.loc[docked_df['ID'].isin(common_ID), predicted_score].to_list()

    # Find Pearson and spearsman correlation between true and predicted values
    pearson_corr = docked_df['Activity'].corr(docked_df[predicted_score])
    spearman_corr = spearmanr(docked_df['Activity'], docked_df[predicted_score])[0]


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
    Calculate ECR scores according to DockM8 tools of outputs from ranking step in DockM8 tool 

    @Param : 
    
    cons_method : Consensus method used

    @Output :
    
    Draw scatter plot with pearson and spearman correlation
    '''''
    true_df = PandasTools.LoadSDF('data/ligands/ecft_scores.sdf', idName="ID")[['ID','HIPS code', 'score']]
    true_df['old rank'] = true_df['score'].apply(pd.to_numeric).rank(method='min')
    for f in os.listdir(f'data/A/dockm8/ranking/'):
        df = pd.read_csv(f'data/A/dockm8/ranking/'+f)
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

def oddt_correlation(results_path, common_ID):

    docked_df = PandasTools.LoadSDF(results_path, idName='ID', molColName='Molecule', strictParsing=False)    
    docked_method = results_path.split('_')[1]

    if "score" in docked_df.columns:
        docked_df.rename(columns = {'score':'Activity'}, inplace = True)

    predicted_scores = ['rfscore_v1', 'rfscore_v2', 'rfscore_v3']
    docked_df[predicted_scores] = docked_df[predicted_scores].apply(pd.to_numeric)
    docked_df['Activity'] = docked_df[['Activity']].apply(pd.to_numeric)

    correlations = []

    for score in predicted_scores:

        docked_df = docked_df.sort_values(score).drop_duplicates(subset="ID")
        pearson_corr = docked_df['Activity'].corr(docked_df[score])
        spearman_corr = spearmanr(docked_df['Activity'], docked_df[score])[0]
        # correlations.append(zip(pearson_corr, spearman_corr))

    figure, axis = plt.subplots(1, 3)
    counter = 0
    

    # plt.scatter(docked_df['Activity'], docked_df[predicted_score], c='blue', label='Calculated Scores')
    # plt.scatter(score_of_commonIDs, predicted_of_commonIDs, c='red', label='IC50')

    figure.set_figheight(10)
    figure.set_figwidth(30)

    for j in range(3):

        score_of_commonIDs = docked_df.loc[docked_df['ID'].isin(common_ID), 'Activity'].to_list()
        predicted_of_commonIDs = docked_df.loc[docked_df['ID'].isin(common_ID),predicted_scores[counter]].to_list()

        docked_df = docked_df.sort_values(score).drop_duplicates(subset="ID")
        pearson_corr = docked_df['Activity'].corr(docked_df[predicted_scores[counter]])
        spearman_corr = spearmanr(docked_df['Activity'], docked_df[predicted_scores[counter]])[0]

        axis[j].scatter(docked_df['Activity'], docked_df[predicted_scores[counter]], c='blue', label='Calculated Scores')
        axis[j].scatter(score_of_commonIDs, predicted_of_commonIDs, c='red', label='IC50')
        

        axis[j].set_title(f'{docked_method.upper()} {predicted_scores[counter]}, snapshot A\n Pearson corr = {pearson_corr:.4f}\n Spearman corr = {spearman_corr:.4f}')


        counter += 1

        if counter == 3:
            plt.legend()
            plt.show()
            break

