from rdkit.Chem import PandasTools
import pandas as pd
import os
import csv
from rdkit import Chem
import multiprocessing
from rdkit import RDLogger
import re


def write_diffdock_input(df, output_path):

    df['smiles'] = df['ROMol'].apply(Chem.MolToSmiles)
#Every line has path to same target and different smiles code.
    header = ['ID', 'ligand_description']

    with open(output_path, 'w', newline='') as file:

        # Create the CSV writer object
        writer = csv.writer(file)

        # Write the header row
        writer.writerow(header)

        for i, mol in df.iterrows():

            writer.writerow([ mol['HIPS code'],mol['smiles'] ])



def create_IC50_sample():
    if 'IC50_mol_only.sdf' not in os.listdir('data/ligands/'):
        print("Extracting IC50 molcules ...")
        suppl = Chem.SDMolSupplier("data/ligands/Symeres.sdf")
        suppl2 = Chem.SDMolSupplier("data/ligands/Hit1.sdf")
        HIPS_code = []
        structure = []
        IC_50 = []
        sheets = [suppl, suppl2]
        for sheet in sheets:
            for m in sheet:
                if m.GetProp('IC50 [µM]') != '':
                    HIPS_code.append(m.GetProp('HIPS code '))
                    IC_50.append(float( m.GetProp('IC50 [µM]')))
                    structure.append(Chem.MolToSmiles(m))
                
        df = pd.DataFrame({'ID': HIPS_code, 'Activity': IC_50, 'smiles': structure})
        df.sort_values(['Activity'], inplace=True)
    
        PandasTools.AddMoleculeColumnToFrame(df, 'smiles', 'Molecule')
        PandasTools.RenderImagesInAllDataFrames(True)
        PandasTools.WriteSDF(df, 'data/ligands/IC50_mol_only.sdf',idName="ID", molColName='Molecule', properties=['ID', 'Activity', 'smiles'])
    else:
        print("IC50 Molecules already exist.")
        return PandasTools.LoadSDF('data/ligands/IC50_mol_only.sdf',idName="ID", molColName='Molecule')



def add_negative_data(data_name):
    '''''
    It loads the scores from "scores_ecft.csv" and loads the structures from "HIPS compounds 001-7433.sdf" to have in a single dataframe

    @Param: name of output file (without .sdf)

    @Return: Size of data and the dataframe
    '''''
    RDLogger.DisableLog('rdApp.*')  
    train_df = pd.read_csv('data/ligands/scores_ecft.csv').sort_values("compound")
    alldata_df = PandasTools.LoadSDF('data/ligands/HIPS compounds 001-7433.sdf')
    train_df_struct = alldata_df[alldata_df['HIPS code'].isin(train_df['compound'])].sort_values("HIPS code")
    train_df_struct['score'] = train_df['score'].to_list()

    PandasTools.WriteSDF(train_df_struct, f'data/ligands/{data_name}.sdf',idName="ID", molColName='ROMol', properties=train_df_struct.columns)
    return train_df_struct.shape[0], train_df_struct

def run_gypsumdl(ligand_library, output):

    ncpus = multiprocessing.cpu_count()
    gypsum_dl_command = f'python software/gypsum_dl-1.2.0/run_gypsum_dl.py -s {ligand_library} -o {os.path.dirname(ligand_library)} --job_manager multiprocessing -p '+str(ncpus)+' -m 1 -t 10 --skip_adding_hydrogen --skip_alternate_ring_conformations --skip_making_tautomers --skip_enumerate_chiral_mol --skip_enumerate_double_bonds --max_variants_per_compound 1 '
    if output not in os.listdir('data/ligands'):
        os.system(gypsum_dl_command)  
            # Clean output data (Remove the first row) and remove old one
        gypsum_df = PandasTools.LoadSDF('data/ligands/gypsum_dl_success.sdf', idName='ID', molColName='Molecule', strictParsing=True)
        cleaned_df = gypsum_df.iloc[1:, :]
        cleaned_df = cleaned_df[['Molecule', 'ID']]
        PandasTools.WriteSDF(cleaned_df, f'data/ligands/{output}.sdf', idName='ID', molColName='Molecule', properties=cleaned_df.columns)
        os.remove('data/ligands/gypsum_dl_success.sdf')  
    else:
        print("Molecules are already prepared")
    
    return f'data/ligands/{output}.sdf'

def read_diffdock_output(df, results_path):
    '''''
    Read Rank 1 of every docked molecule using DiffDock and write a SDF file with ID, confidence score and Predicted structure

    @Param :

    df : true df data that has score and HIPS code as columns
    results_path : Path of diffdock results

    @Return:
    Write a SDF file with ID, confidence score and Predicted structure
    '''''

    ids = []
    confidence_score = []
    mols = []
    cwd = os.getcwd()
    for dir in os.listdir(f'{cwd}/{results_path}'):
        ids.append(dir)
        for file in os.listdir(f'{cwd}/{results_path}{dir}'):

            if file.startswith('index'):
                for dir2 in os.listdir(f'{cwd}/{results_path}{dir}/{file}'):
                    if dir2 == 'rank1.sdf':


                        supplier = Chem.SDMolSupplier(f'{cwd}/{results_path}{dir}/{file}/{dir2}')
                        for molecule in supplier:
                                if molecule is not None:
                                    mols.append(molecule)
                    if dir2.startswith('rank1_conf'):
                        match = re.search(r"[-+]?\d+(\.\d+)", dir2)
                        if match:
                            number = match.group(0)
                            confidence_score.append(float(number))

    diffdock_df = pd.DataFrame({'HIPS code': ids, 
                                'confidence_score': confidence_score, 
                                'Molecules': mols})

    merged_df = pd.merge(diffdock_df, df,on='HIPS code', how='inner')[['Molecules', 'HIPS code', 'score', 'confidence_score']]
    merged_df.rename(columns = {'HIPS code':'ID'}, inplace = True)

    PandasTools.WriteSDF(merged_df, 'data/A/docked_diffdock_poses_A_212.sdf',idName='ID', molColName='Molecules', properties=merged_df.columns)