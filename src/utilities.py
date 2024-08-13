import csv
import os
import subprocess
from pathlib import Path

import pandas as pd
from Bio import SeqIO
from Bio.Data import IUPACData
from Bio.PDB import PDBParser
from rdkit import Chem
from itertools import combinations, product


def run_command(cmd):

    try:
        subprocess.call(cmd,
                        shell=True,
                        #stdout=subprocess.DEVNULL,
                        #stderr=subprocess.STDOUT
                        )
    except Exception as e:
        print(e)
        print(f'Error occured while running {cmd}')
        return False


def get_ligand_coordinates_centers(ligand_molecule):
    '''This function returns the center of the ligand molecule in x, y and z coordinates
    Args:
        ligand_molecule: ligand molecule in pdb or sdf format

    Returns: 
        center of the ligand molecule in x, y and z coordinates
    '''
    if ligand_molecule.endswith('.sdf'):
        supplier = Chem.SDMolSupplier(
            ligand_molecule, sanitize=False, removeHs=False)
        mol = supplier[0]
    elif ligand_molecule.endswith('.pdb'):
        mol = Chem.MolFromPDBFile(
            ligand_molecule,
            removeHs=False
        )
    else:
        print("Please provide a valid ligand file with .pdb or .sdf extension")
    ligand_conformer = mol.GetConformers()[0]
    coordinates = ligand_conformer.GetPositions()
    df = pd.DataFrame(
        coordinates,
        columns=[
            "x_coord",
            "y_coord",
            "z_coord"])
    center_x = str(df['x_coord'].mean().round(2))
    center_y = str(df['y_coord'].mean().round(2))
    center_z = str(df['z_coord'].mean().round(2))

    return center_x, center_y, center_z


def plants_pocket_generation(protein_file_mol2, ref_file_mol2):

    plants_pocket_cmd = f"./software/PLANTS --mode bind {str(ref_file_mol2)} {str(protein_file_mol2)}"
    run_command(plants_pocket_cmd)
    print('PLANTS bind mode was executed.')


def pocket_coordinates_generation(
        protein_mol2,
        ref_file_mol2,
        pocket_coordinates_path='bindingsite.def'
        ):
    '''
    This function generates the pocket coordinates of the protein file and returns the center of the pocket in x, y and z coordinates and the radius of the pocket

    Args:
        protein_mol2: protein file in mol2 format
        ref_file_mol2: reference file in mol2 format
        pocket_coordinates_path: path to the pocket coordinates file

    Returns:
        center_x: x coordinate of the pocket center
        center_y: y coordinate of the pocket center
        center_z: z coordinate of the pocket center
        radius: radius of the pocket
    '''

    plants_pocket_generation(protein_mol2, ref_file_mol2)

    # Open and read the file
    with open(Path.cwd() / pocket_coordinates_path, 'r') as file:
        for line in file:
            # Split the line into words
            parts = line.split()
            # Check if the line contains the bindingsite center
            if parts[0] == 'bindingsite_center':
                center_x, center_y, center_z = map(float, parts[1:4])
            # Check if the line contains the bindingsite radius
            elif parts[0] == 'bindingsite_radius':
                radius = float(parts[1])
    return center_x, center_y, center_z, radius


def pdb2resname_resno(input_file):
    '''
    This function extracts the residue name and residue number from the protein file and returns a list of tuples of resname and resnumber

    Args:
        input_file: protein file in pdb format

    Returns: 
        a list of tuples of chain_ID, resname and resnumber
    '''

    resname_resnumbers = []
    # Read the lines of the file
    with open(input_file, 'r') as pdb_file:
        pdb_lines = pdb_file.readlines()
        for pdb_line in pdb_lines:
            if pdb_line.startswith('ATOM'):
                pdb_line_list = pdb_line.split()
                if (pdb_line_list[4], pdb_line_list[3],
                        pdb_line_list[5]) in resname_resnumbers:
                    continue

                resname_resnumbers.append(
                    (pdb_line_list[4], pdb_line_list[3], pdb_line_list[5]))

    return (resname_resnumbers)


def extract_binding_pocket(protein_file, output_file):
    '''
    This function extracts the binding pocket residue indices from the protein file
    using the binding pocket and writes the residue number to a csv file which is suitable more for local diffdock

    Binding pocket file has to be named  the same as the protein file with the suffix '_pocket.pdb'
    
    Args:
        protein_file: protein file in pdb format
        output_file: output file in csv format

    Returns:
      csv file with residue numbers of binding pocket residues in results folder
    '''
    binding_pocket = protein_file.with_name(protein_file.stem + '_pocket.pdb')
    if binding_pocket.exists():
        whole = pdb2resname_resno(str(protein_file))
        ref = pdb2resname_resno(str(binding_pocket))
    else:
        print('Binding pocket file does not exist. Please provide a binding pocket file')
        return

    indices_dict = {element: [] for element in ref}

    for i, element in enumerate(whole):
        if element in indices_dict:
            indices_dict[element] = (whole.index(element))

    with open(str(output_file), 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        # Combine all items in the dictionary values into a single row
        row = []
        for _, item in indices_dict.items():
            row.append(item)
        
        # Write the single row to the CSV file
        writer.writerow(row)

def any_in_list(list1, list2):
    return any(i in list2 for i in list1)


def split_sdf(file_path, scoring_function, thread_num):
    """
    Split SDF file into chunks
    Args:
        file_path: Path to SDF file
        thread_num: Number of threads to split the file into

    Returns: 
        List of paths to splitted files
    """
    # @ TODO : Convert splitted to pdbqt format in case of CHEMPLP and SCORCH

    output_folders = []
    with open(file_path, 'r') as file:
        lines = file.readlines()

    molecules = ''.join(lines).split('$$$$\n')
    molecules_per_chunk = len(molecules) // thread_num

    for i in range(thread_num):

        # every chuncks indices
        start_index = i * molecules_per_chunk
        end_index = start_index + molecules_per_chunk if i != thread_num - 1 else None

        split_folder = file_path.parent / 'sdf_split'
        split_folder.mkdir(exist_ok=True)

        # write chunk to file
        with open(str(split_folder / f"sdf_{i}.sdf"), "w") as out_file:
            out_file.write('$$$$\n'.join(
                molecules[start_index:end_index]) + '$$$$\n')
        if 'chemplp' == scoring_function:
            # convert to mol2
            if not (split_folder / f"sdf_{i}.mol2").exists():
                run_command(f"obabel -isdf {split_folder / f'sdf_{i}.sdf'}"
                            f" -O {split_folder / f'sdf_{i}.mol2'}"
                            )
                print(f'sdf_{i}.sdf is converted to mol2')
            output_folders.append(split_folder / f"sdf_{i}.mol2")
        elif 'scorch' == scoring_function:
            # convert to pdbqt
            if not (split_folder / f"sdf_{i}.pdbqt").exists():
                run_command(f"obabel {split_folder / f'sdf_{i}.sdf'}"
                            f" -O {split_folder / f'sdf_{i}.pdbqt'}"
                            " --partialcharges gasteiger")

                print(f'sdf_{i}.sdf is converted to pdbqt')
            output_folders.append(split_folder / f"sdf_{i}.pdbqt")
        else:
            output_folders.append(split_folder / f"sdf_{i}.sdf")

    return output_folders


def pdb_converter(
    protein_file,
    rescore_programs,
):
    '''
    The function converts the protein file to pdbqt format in case of SCORCH SF
    and to mol2 format in case of CHEMPLP SF

    Args: 
        protein_file: protein file in pdb format
        rescore_programs: list of rescoring programs
    '''
    if "chemplp" in rescore_programs:
        if f'{protein_file.stem}.mol2' in os.listdir(protein_file.parent):
            print('protein is already converted to mol2')
        else:
            # convert protein to pdbqt
            run_command(f"obabel -ipdb {str(protein_file)}" f" -O {str(protein_file.parent / f'{protein_file.stem}.mol2')}")

    if "scorch" in rescore_programs:

        if f'{protein_file.stem}.pdbqt' in os.listdir(protein_file.parent):
            print('protein is already converted to pdbqt')
        else:
            # convert protein to pdbqt
            run_command(f"obabel -ipdb {str(protein_file)}"
                        f" -O {str(protein_file.parent / f'{protein_file.stem}.pdbqt')}"
                        " --partialcharges gasteiger -p -h")


def split_list(input_list, num_splits):
    """
    Split a list into a number of splits
    Args:
        input_list: List to split
        num_splits: Number of splits
    Returns:
        partitions: List of partitions
    """
    avg_size = len(input_list) // num_splits
    remain = len(input_list) % num_splits
    partitions = []
    i = 0
    for _ in range(num_splits):
        partition_size = avg_size + 1 if remain > 0 else avg_size
        partitions.append(input_list[i:i + partition_size])
        i += partition_size
        remain -= 1
        # print(len(partitions[-1]))
    return partitions


def read_posebusters_data(df):
    """
    Read the PoseBusters data and return the number of docked molecules for each docking tool and the number of failed molecules 
    Args:
        df: DataFrame of the PoseBusters data
    Returns:
        df_filtered: DataFrame with the number of docked molecules for each docking tool
    """ 
    df[['original_id', 'docking_tool', 'pose']
       ] = df['molecule'].str.split('_', expand=True)
    # Group by the 'docking_tool' and sum the False values for every column.
# Convert boolean columns to integers for summing (False becomes 0, True becomes 1).
# Then subtract the sum from the count of rows to get the number of False
# values.
    df_grouped = df.groupby('docking_tool').apply(
        lambda x: x.shape[0] -
        x.select_dtypes(
            include=['bool']).astype(int).sum())

    unique_id_count = df.groupby('docking_tool')['original_id'].nunique().reset_index(name='unique_id_count')

    df_grouped['Number of Docked Molecules'] = list(
        unique_id_count['unique_id_count'])
    # df_grouped['original_id'] = df.original_id.value_counts()
    # # Reset the index to have 'docking_tool' as a column
    df_grouped.reset_index(inplace=True)
    df_filtered = df_grouped.loc[:, (df_grouped != 0).any(axis=0)]
    return df_filtered

# make a function that takes the path of the csv file and generate
# correlation matrix and check pairs that has correlation of 0.9 or higher


def generate_correlation_matrix(df):
    """
    Generate correlation matrix from a csv file
    Args:
        df: DataFrame of the csv file
    Returns:
        correlation_matrix: Correlation matrix of the numerical columns
    """
    for col in df.columns:
        try:
            df[col] = pd.to_numeric(df[col])
        except:
            continue
    correlation_matrix = df.select_dtypes('number').corr()
    return correlation_matrix


def check_correlation_pairs(correlation_matrix, threshold=0.9):
    """
    Check pairs of columns that have correlation of a threshold, 0.9 is the default threshold
    Args:
        correlation_matrix: Correlation matrix of the numerical columns
        threshold: Threshold for correlation
    Returns:
        pairs: List of pairs of columns that have correlation of a threshold
    """
    pairs = []
    for i in range(correlation_matrix.shape[0]):
        for j in range(correlation_matrix.shape[1]):
            if i != j and correlation_matrix.iloc[i, j] >= threshold:
                pairs.append(
                    (correlation_matrix.columns[i],
                     correlation_matrix.columns[j]))
    return pairs


def handling_multicollinearity(df, threshold=0.9, true_value_col='true_value'):
    """
    Remove columns that are highly correlated with each other and keep the ones that are more correlated with the 'True Value'
    Args:
        df: dataframe
        threshold: Threshold for correlation
        true_value_col: Name of the column that is the true value
    Returns:
        df: DataFrame without correlated columns
    """

    corr_matrix = generate_correlation_matrix(df)
    display(corr_matrix.style.background_gradient(cmap='coolwarm'))
    pairs = check_correlation_pairs(corr_matrix, threshold)
    columns_to_remove = set()
    for col1, col2 in pairs:

        corr_with_true_value_col1 = corr_matrix.loc[true_value_col, col1]
        corr_with_true_value_col2 = corr_matrix.loc[true_value_col, col2]
        if corr_with_true_value_col1 > corr_with_true_value_col2:
            columns_to_remove.add(col2)
        else:
            columns_to_remove.add(col1)
    print(f"Scores of {columns_to_remove} were found to highly correlate. Therefore, they are removed")
    df.drop(columns=list(columns_to_remove), inplace=True)
    return df

def split_list(input_list, num_splits):
    '''
    Split a list into n parts.
    Args:
        input_list: list to be split
        num_splits: number of splits
    Returns: 
        list of splitted lists
    '''
    avg_size = len(input_list) // num_splits
    remain = len(input_list) % num_splits
    partitions = []
    i = 0
    for _ in range(num_splits):
        partition_size = avg_size + 1 if remain > 0 else avg_size
        partitions.append(input_list[i:i+partition_size])
        i += partition_size
        remain -= 1
        # print(len(partitions[-1]))
    return partitions
def workflow_combinations(docking_programs: list, rescoring_programs: list):
    """
    Generate all combinations of docking methods and scoring functions.
    Args:
        docking_programs: list of docking methods
        rescoring_programs: list of rescoring programs
    Returns: 
               list of tuples with all combinations of docking methods and scoring functions
    """
    all_comb_scoring_function = [item for r in range(1, len(rescoring_programs) + 1) 
                                 for item in combinations(sorted(rescoring_programs), r)]
    all_comb_docking_program = [item for r in range(1, len(docking_programs) + 1) 
                                 for item in combinations(sorted(docking_programs), r)]

    return list(product(all_comb_docking_program, all_comb_scoring_function))