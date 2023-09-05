import os
import re
import glob

import csv
import itertools
from rdkit import Chem
from rdkit.Chem import PandasTools
import pandas as pd
from rdkit import Chem
from scipy.stats import spearmanr
from matplotlib import pyplot as plt
from rdkit import RDLogger
from DockM8.scripts import consensus_methods
from pymol import cmd, stored

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
    docked_df = PandasTools.LoadSDF(
        results_path,
        idName='ID',
        molColName='Molecule',
        strictParsing=False)
    docked_method = results_path.split('_')[1]
    scoring_method = results_path.split('_')[2]

    if "score" in docked_df.columns:
        docked_df.rename(columns={'score': 'Activity'}, inplace=True)

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

    plt.figure(figsize=(10, 8))

    # Keep best predicted affinity for every compound.
    docked_df[[predicted_score, 'Activity']] = docked_df[[
        predicted_score, 'Activity']].apply(pd.to_numeric)
    docked_df = docked_df.sort_values(
        predicted_score).drop_duplicates(subset="ID")

    # Specify different colors for the differentiated points
    score_of_commonIDs = docked_df.loc[docked_df['ID'].isin(
        common_ID), 'Activity'].to_list()
    predicted_of_commonIDs = docked_df.loc[docked_df['ID'].isin(
        common_ID), predicted_score].to_list()

    # Find Pearson and spearsman correlation between true and predicted values
    pearson_corr = docked_df['Activity'].corr(docked_df[predicted_score])
    spearman_corr = spearmanr(
        docked_df['Activity'],
        docked_df[predicted_score])[0]

    plt.scatter(
        docked_df['Activity'],
        docked_df[predicted_score],
        c='blue',
        label='Calculated Scores')
    plt.scatter(
        score_of_commonIDs,
        predicted_of_commonIDs,
        c='red',
        label='IC50')

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
    docked_df = PandasTools.LoadSDF(
        results_path,
        idName='ID',
        molColName='Molecule',
        strictParsing=False)
    if all(docked_df[docked_df['ID'].str.startswith('StarDrop')]):
        docked_df = PandasTools.LoadSDF(
            results_path,
            idName='ID',
            molColName='Molecule',
            strictParsing=False)
        docked_df['ID'] = docked_df['ID'].str.split('_').str[0]
    true_df = PandasTools.LoadSDF(ligand_library,
                                  idName='ID',
                                  molColName='Molecule',
                                  strictParsing=False)[['HIPS code',
                                                        'ID',
                                                        'score']]
    merged_df = pd.merge(docked_df, true_df, on='ID').drop('ID', axis=1)
    merged_df.rename(
        columns={
            'score': 'Activity',
            'HIPS code': 'ID'},
        inplace=True)
    return merged_df


def prcoess_ranking_data(df, true_df, method):
    '''''
    @Param : df = gnerated df from DockM8, true_df = the first input, method = the type of the metric used
    ----------------
    @Return : dataframe have the true score , HIPS codes and the consensus score
    '''''
    df = df.sort_values(by=[method]).drop_duplicates(subset="ID")

    # Add info to stardrop ID first
    df_stardrop = df[df['ID'].str.startswith('StarDrop')].sort_values(by="ID")
    filtered_df = true_df[true_df['ID'].isin(
        df_stardrop['ID'])].sort_values(by="ID")

    df_stardrop['ID'] = filtered_df['HIPS code'].to_list()
    df_stardrop['score'] = filtered_df['score'].to_list()

    # Add info to HIPS ID second
    df_HIPS = df[df['ID'].str.startswith('HIPS')].sort_values(by="ID")
    filtered_df2 = true_df[true_df['HIPS code'].isin(
        df_HIPS['ID'])].sort_values(by="HIPS code")
    df_HIPS['score'] = filtered_df2['score'].to_list()

    # Merge both again
    merged_df = pd.concat([df_HIPS, df_stardrop]).drop_duplicates('ID')
    merged_df[[method, 'score']] = merged_df[[
        method, 'score']].apply(pd.to_numeric)

    return merged_df


def correlation_mapping(df, common_ID):

    method_name = df.columns[1]
    print(method_name)

    score_of_commonIDs = df.loc[df['ID'].isin(common_ID), 'score'].to_list()
    predicted_of_commonIDs = df.loc[df['ID'].isin(
        common_ID), method_name].to_list()
    spearman_corr = spearmanr(df['score'], df[method_name])[0]
    pearson_corr = df['score'].corr(df[method_name])

    plt.figure(figsize=(10, 8))

    plt.scatter(
        df['score'],
        df[method_name],
        c='blue',
        label='Calculated Scores')
    plt.scatter(
        score_of_commonIDs,
        predicted_of_commonIDs,
        c='red',
        label='IC50')
    plt.xlabel('Scores')
    plt.ylabel('Predicted Affinity')
    plt.title(
        f'DockM8 {method_name}, snapshot A\n Pearson corr = {pearson_corr:.4f}\n Spearman corr = {spearman_corr:.4f}')
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
    true_df = PandasTools.LoadSDF(
        'data/ligands/ecft_scores.sdf', idName="ID")[['ID', 'HIPS code', 'score']]
    true_df['old rank'] = true_df['score'].apply(
        pd.to_numeric).rank(
        method='min')
    for f in os.listdir(f'data/A/dockm8/ranking/'):
        df = pd.read_csv(f'data/A/dockm8/ranking/' + f)
        selected_col = df.columns[1:-1]

        metric = f.split('.')[0]

        if cons_method == 'method1':
            cons_df = consensus_methods.method1_ECR_best(
                df, metric, selected_col)
            cons_metric = f'Method1_ECR_{metric}'
        elif cons_method == 'method2':
            cons_df = consensus_methods.method2_ECR_average(
                df, metric, selected_col)
            cons_metric = f'Method2_ECR_{metric}'
        elif cons_method == 'method6':
            cons_df = consensus_methods.method6_Zscore_best(
                df, metric, selected_col)
            cons_metric = f'Method6_Zscore_{metric}'

        cons_df = prcoess_ranking_data(cons_df, true_df, cons_metric)

        correlation_mapping(cons_df, common_ID)


def oddt_correlation(results_path, common_ID):

    docked_df = PandasTools.LoadSDF(
        results_path,
        idName='ID',
        molColName='Molecule',
        strictParsing=False)
    docked_method = results_path.split('_')[1]

    if "score" in docked_df.columns:
        docked_df.rename(columns={'score': 'Activity'}, inplace=True)

    predicted_scores = ['rfscore_v1', 'rfscore_v2', 'rfscore_v3']
    docked_df[predicted_scores] = docked_df[predicted_scores].apply(
        pd.to_numeric)
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

        score_of_commonIDs = docked_df.loc[docked_df['ID'].isin(
            common_ID), 'Activity'].to_list()
        predicted_of_commonIDs = docked_df.loc[docked_df['ID'].isin(
            common_ID), predicted_scores[counter]].to_list()

        docked_df = docked_df.sort_values(score).drop_duplicates(subset="ID")
        pearson_corr = docked_df['Activity'].corr(
            docked_df[predicted_scores[counter]])
        spearman_corr = spearmanr(
            docked_df['Activity'], docked_df[predicted_scores[counter]])[0]

        axis[j].scatter(docked_df['Activity'],
                        docked_df[predicted_scores[counter]],
                        c='blue',
                        label='Calculated Scores')
        axis[j].scatter(
            score_of_commonIDs,
            predicted_of_commonIDs,
            c='red',
            label='IC50')

        axis[j].set_title(
            f'{docked_method.upper()} {predicted_scores[counter]}, snapshot A\n Pearson corr = {pearson_corr:.4f}\n Spearman corr = {spearman_corr:.4f}')

        counter += 1

        if counter == 3:
            plt.legend()
            plt.show()
            break


def test_localdock_diffdock():

    def edit_file(file_paths, comb):

        for file_path in file_paths:
            with open(file_path, 'r') as file:
                lines = file.readlines()

            with open(file_path, 'w') as file:
                for line in lines:

                    if line.startswith('tr_sigma_max:'):
                        new_line = f"tr_sigma_max: {comb[0]:.1f}"
                        file.write(new_line + '\n')

                    elif line.startswith('tor_sigma_max:'):
                        new_line = f"tor_sigma_max: {comb[1]:.2f}"
                        file.write(new_line + '\n')

                    elif line.startswith('rot_sigma_max:'):
                        new_line = f"rot_sigma_max: {comb[2]:.2f}"
                        file.write(new_line + '\n')

                    else:
                        file.write(line)

    # Example usage
    cwd = os.getcwd()
    # path of diffdock
    file_paths = [
        cwd + '/workdir/paper_confidence_model/model_parameters.yml',
        cwd + '/workdir/paper_score_model/model_parameters.yml']

    tr_sigma_max = [10.0]
    tor_sigma_max = [1.64]
    rot_sigma_max = [1.40]

    for _ in range(30):
        tr_sigma_max.append((tr_sigma_max[-1] + 2))
        tor_sigma_max.append((tor_sigma_max[-1] + 0.2))
        rot_sigma_max.append((rot_sigma_max[-1] + 0.2))

    combinations = list(
        itertools.product(
            tr_sigma_max,
            tor_sigma_max,
            rot_sigma_max))

    for comb in combinations:
        edit_file(file_paths, comb)
        diffdock_cmd = f"python -m inference --protein_path data/ecft/protein_protoss_noligand.pdb --ligand '' --out_dir results/local_dock/tr{comb[0]}_tor{comb[1]}_rot_{comb[2]} --inference_steps 20 --samples_per_complex 5 --batch_size 10 --actual_steps 18 --no_final_step_noise"
        os.system(diffdock_cmd)


def split_sdf(sdf_path, docked_group, number_of_poses):
    '''''
    This function takes a path of sdf file and split it into multiple sdf files with the same name of the ligand

    @Param :
    sdf_path --> path of sdf file
    docked_group --> name of the group that the sdf file belongs to
    number_of_poses --> number of poses that you want to split the sdf file into

    @Return :
    multiple sdf files with the same name of the ligand
    '''''
    suppl = Chem.SDMolSupplier(sdf_path)
    os.makedirs(f"dcc_data/{docked_group}", exist_ok=True)
    pose = 0
    for mol in suppl:
        if mol is None: continue
        #variable that take ID of sdf file from the first row of the sdf file by reading it as a string
        pose += 1
        ligand_name = Chem.MolToMolBlock(mol).split()[0] 
        #save sdf in a sdf file with the ligand name
        writer = Chem.SDWriter(f"dcc_data/{docked_group}/{ligand_name}_{pose}.sdf")
        writer.write(mol)
        writer.close()
        if pose % number_of_poses == 0:
            pose = 0   
            
def interaction_fp_generator(chain, ligands_path):
    complex_files = list(Path(ligands_path / "ligand_protein_complex").glob(f"*{chain}.pdb"))

    structures = [Structure.from_pdbfile(str(pdb), ligand_name="HIT") for pdb in tqdm(complex_files)]

    fp = InteractionFingerprint().calculate_fingerprint(
            structures, # see comment above excluded Mpro-z structure
            labeled=True, 
            as_dataframe=True, 
            remove_non_interacting_residues=True,
            remove_empty_interaction_types=True,
            ensure_same_sequence=False,
        )

    if not fp.values.shape[0]:
        raise ValueError("Fingerprint is empty!")
    fp_focused = fp[fp.sum(axis=1) > 5]
    print(f"\n\n{ligands_path.stem}\n\nChain {chain}")
    display(fp_focused.style.background_gradient(axis=None, cmap="YlGnBu"))
    display(fingerprint_barplot(fp_focused))

def create_2dposeview(docked_group):

    '''''
    This function takes a path of sdf files and create 2D interaction images for each sdf file
    @Param :
    docked_group --> name of the group that the sdf file belongs to
    @Output :
    2D interaction images for each sdf file in 2D_interactions folder
    '''''
    
    #create folder for the photos
    os.makedirs(f"dcc_data/{docked_group}/2D_interactions", exist_ok=True)
    #create list of all sdf files in the folder
    sdf_files = glob.glob(f"dcc_data/{docked_group}/*.sdf")
    for sdf in sdf_files:
        os.system(f"./software/poseview-1.1.2-Linux-x64/poseview -l {sdf} -p data/A/protein_protoss_noligand.pdb -o dcc_data/{docked_group}/2D_interactions/{sdf.split('/')[-1].split('.')[0]}.png")

def plipify_ligand_protein_preparation(ligands_path, protein_path, protein_name):
    ''''
    This function loads ligands and protein using pymol script commands and save both protein and ligand as a complex as pdb file.
    It splits Chain C and D to separate pdb file and change ligand according to chain.

    Parameters:
    ------------
    ligands_path: path to folder containing ligands in sdf format
    protein_path: path to protein in pdb format
    protein_name: name of protein
    
    Returns:
    ------------
    PDB file of ligand protein complex of chain C and D
    '''
    
    #load ligands as list of sdf files
    ligand_files = list(ligands_path.glob("*.pdb"))

    #create folder to save ligand protein complex
    if os.path.exists(ligands_path / "ligand_protein_complex"):
        return
    
    os.makedirs(ligands_path / "ligand_protein_complex", exist_ok=True)

    #load protein and ligand and save as pdb file
    for sdf in ligand_files:
        
        ligand_name = sdf.stem



        cmd.load(protein_path)
        cmd.load(sdf, "LIG")
        # select chain C and D
        cmd.alter("all", "q=1.0")

        chains = ['C', 'D']
        for chain_id in chains:
            
            complex_name = f"{ligand_name}_{protein_name}_{chain_id}.pdb"
            pdb_output = ligands_path / "ligand_protein_complex" / complex_name
            cmd.select(f'chain_{chain_id}E', f'chain {chain_id}+E')
            cmd.save(pdb_output, f'chain_{chain_id}E') 


            # open pdb file and remove line starts with TER and write it at after line starts with HETATM.
            with open(pdb_output, "r") as f:
                    lines = f.readlines()
                    new_lines = []
                    for line in lines:
                        if line.startswith("CONECT"):
                            new_lines.append(f"TER \nEND\n")
                            break
                        if line.startswith("TER"):
                            continue
                        else:
                            new_lines.append(line)
                    with open(pdb_output, "w") as f:
                        for line in new_lines:
                            f.write(line)
        cmd.delete("all")



def sdf_preprocessing(sdf_files):
    for sdf in sdf_files:
        cmd.load(sdf)
        cmd.alter('resi 0', 'resi = 287')
        cmd.alter('resn UNK', 'resn = "HIT"')
        cmd.alter('HETATM', 'chain="E"')
        cmd.save(sdf.parent / f"{sdf.stem}.pdb") 
        cmd.delete("all")

