import os
import ast
import csv
import subprocess
import requests
import zipfile
from pathlib import Path

import pandas as pd

from rdkit import Chem
from Bio import SeqIO
from Bio.Data import IUPACData
from Bio.PDB import PDBParser
from itertools import combinations, product


def run_command(cmd: str):
    """
    Run a command in the shell
    Args:
        cmd(str): Command to run
    """
    try:
        subprocess.call(cmd,
                        shell=True,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.STDOUT
                        )
    except Exception as e:
        print(e)
        print(f'Error occured while running {cmd}')
        return False

def _plants_pocket_generation(protein_file_mol2: Path, ref_file_mol2: Path):
    """
    This function generates the pocket coordinates of the protein file

    Args:
        protein_file_mol2(pathlib.Path): protein file in mol2 format
        ref_file_mol2(pathlib.Path): reference file in mol2 format
    """
    plants_pocket_cmd = f"./software/PLANTS --mode bind {str(ref_file_mol2)} {str(protein_file_mol2)}"
    run_command(plants_pocket_cmd)
    print('PLANTS bind mode is executed.')

def pocket_coordinates_generation(
        protein_mol2: Path,
        ref_file_mol2: Path,
        pocket_coordinates_path : str ='bindingsite.def'
        ):
    """
    This function generates the pocket coordinates of the protein file and 
    returns the center of the pocket in x, y and z coordinates and the radius of the pocket

    Args:
        - protein_file_mol2 (pathlib.Path): protein file in mol2 format
        - ref_file_mol2 (pathlib.Path) : reference file in mol2 format
        - pocket_coordinates_path (str) : Optional; path to the pocket coordinates file

    Returns:
        tuple: A tuple containing the x, y, and z coordinates of the pocket center,
               and the radius of the pocket. Specifically:
               - center_x (float): x coordinate of the pocket center.
               - center_y (float): y coordinate of the pocket center.
               - center_z (float): z coordinate of the pocket center.
               - radius (float): radius of the pocket.
    """
    _plants_pocket_generation(protein_mol2, ref_file_mol2)

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

def _pdb2resname_resno(input_file : str) -> list:
    """
    This function extracts the residue name and residue number from the protein file 
    and returns a list of tuples of resname and resnumber

    Args:
        input_file(str): protein file in pdb format

    Returns: 
        resname_resnumbers(list): a list of tuples of chain_ID, resname and resnumber
    """
    resname_resnumbers = []
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


def extract_binding_pocket(protein_file : Path, output_file : Path) :
    """
    This function extracts the binding pocket residue indices
    from the protein file using the binding pocket and writes 
    the residue number to a csv file which is suitable more for local diffdock

    Binding pocket file has to be named  the same as the protein file with 
    the suffix '_pocket.pdb'
    
    Args:
        protein_file(pathlib.Path): protein file in pdb format
        output_file(pathlib.Path): output file in csv format

    Returns:
      csv file with residue numbers of binding pocket residues in results folder
    """
    binding_pocket = protein_file.with_name(protein_file.stem + '_pocket.pdb')
    if binding_pocket.exists():
        whole = _pdb2resname_resno(str(protein_file))
        ref = _pdb2resname_resno(str(binding_pocket))
    else:
        print('Binding pocket file does not exist. Please provide a binding pocket file')
        return

    indices_dict = {element: [] for element in ref}

    for i, element in enumerate(whole):
        if element in indices_dict:
            indices_dict[element] = (whole.index(element))

    with open(str(output_file), 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)

        row = []
        for _, item in indices_dict.items():
            row.append(item)
        
        writer.writerow(row)


def split_sdf(file_path, scoring_function, thread_num):
    """
    Split SDF file into chunks
    Args:
        file_path(pathlib.Path): Path to SDF file
        scoring_function(str): Scoring function to use
        thread_num(int): Number of threads to split the file into

    Returns: 
        List of paths to splitted files
    """
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
    protein_file : Path,
    rescore_programs : list,
):
    """
    The function converts the protein file to pdbqt format in case of SCORCH SF
    and to mol2 format in case of CHEMPLP SF

    Args: 
        protein_file(pathlib.Path): protein file in pdb format
        rescore_programs(list): list of rescoring programs
    """
    if "chemplp" in rescore_programs:
        if f'{protein_file.stem}.mol2' in os.listdir(protein_file.parent):
            print('protein is already converted to mol2')
        else:
            run_command(f"obabel -ipdb {str(protein_file)}" f" -O {str(protein_file.parent / f'{protein_file.stem}.mol2')}")

    if "scorch" in rescore_programs:

        if f'{protein_file.stem}.pdbqt' in os.listdir(protein_file.parent):
            print('protein is already converted to pdbqt')
        else:
            run_command(f"obabel -ipdb {str(protein_file)}"
                        f" -O {str(protein_file.parent / f'{protein_file.stem}.pdbqt')}"
                        " --partialcharges gasteiger -p -h")


def split_list(input_list, num_splits):
    """
    Split a list into a number of splits
    Args:
        input_list(list): List to split
        num_splits(int): Number of splits
    Returns:
        partitions(list): List of partitions
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
    return partitions


def read_posebusters_data(df : pd.DataFrame) -> pd.DataFrame:
    """
    Read the PoseBusters data and return the number of docked molecules 
    for each docking tool and the number of failed molecules 
    Args:
        df(pd.DataFrame): DataFrame of the PoseBusters data
    Returns:
        df_filtered(pd.DataFrame): DataFrame with the number of docked molecules for each docking tool
    """ 
    df[['original_id', 'docking_tool', 'pose']] = df['molecule'].str.split('_', expand=True)
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
    df_grouped.reset_index(inplace=True)
    df_filtered = df_grouped.loc[:, (df_grouped != 0).any(axis=0)]
    return df_filtered

def _generate_correlation_matrix(df : pd.DataFrame) -> pd.DataFrame:
    """
    Generate correlation matrix from a csv file
    Args:
        df (pd.DataFrame): DataFrame of the csv file
    Returns:
        correlation_matrix(pd.DataFrame): Correlation matrix of the numerical columns
    """
    for col in df.columns:
        try:
            df[col] = pd.to_numeric(df[col])
        except:
            continue
    correlation_matrix = df.select_dtypes('number').corr()
    return correlation_matrix

def check_correlation_pairs(
        correlation_matrix : pd.DataFrame, 
        threshold : float =0.9
        ) -> list[tuple]:
    """
    Check pairs of columns that have correlation of a threshold, 0.9 is the default threshold
    Args:
        correlation_matrix (pd.DataFrame): Correlation matrix of the numerical columns
        threshold (float): Threshold for correlation
    Returns:
        pairs (list): List of pairs of columns that have correlation of a threshold
    """
    pairs = []
    for i in range(correlation_matrix.shape[0]):
        for j in range(correlation_matrix.shape[1]):
            if i != j and correlation_matrix.iloc[i, j] >= threshold:
                pairs.append(
                    (correlation_matrix.columns[i],
                     correlation_matrix.columns[j]))
    return pairs

def handling_multicollinearity(
        df : pd.DataFrame, 
        threshold : float =0.9, 
        true_value_col : str ='true_value'
        ) -> pd.DataFrame:
    """
    Remove columns that are highly correlated with each other 
    and keep the ones that are more correlated with the 'True Value'.
    Args:
        df (pd.DataFrame): dataframe
        threshold (int): Threshold for correlation
        true_value_col (str): Name of the column that is the true value
    Returns:
        df (pd.DataFrame): DataFrame without correlated columns
    """

    corr_matrix = _generate_correlation_matrix(df)
    # display(corr_matrix.style.background_gradient(cmap='coolwarm'))
    pairs = check_correlation_pairs(corr_matrix, threshold)
    columns_to_remove = set()
    for col1, col2 in pairs:

        corr_with_true_value_col1 = corr_matrix.loc[true_value_col, col1]
        corr_with_true_value_col2 = corr_matrix.loc[true_value_col, col2]
        if corr_with_true_value_col1 > corr_with_true_value_col2:
            columns_to_remove.add(col2)
        else:
            columns_to_remove.add(col1)
    print(f"Scores of {columns_to_remove} were found to highly correlate. Therefore, they are removed.")
    return columns_to_remove

def split_list(input_list : list, num_splits : int) -> list:
    """
    Split a list into n parts.
    Args:
        input_list (list): list to be split
        num_splits (int): number of splits
    Returns: 
        list of splitted lists
    """
    avg_size = len(input_list) // num_splits
    remain = len(input_list) % num_splits
    partitions = []
    i = 0
    for _ in range(num_splits):
        partition_size = avg_size + 1 if remain > 0 else avg_size
        partitions.append(input_list[i:i+partition_size])
        i += partition_size
        remain -= 1
    return partitions

def workflow_combinations(docking_programs: list, rescoring_programs: list) -> list:
    """
    Generate all combinations of docking methods and scoring functions.
    Args:
        docking_programs (list): list of docking methods
        rescoring_programs (list): list of rescoring programs
    Returns: 
        list of tuples with all combinations of docking methods and scoring functions
    """
    all_comb_scoring_function = [item for r in range(1, len(rescoring_programs) + 1) 
                                 for item in combinations(sorted(rescoring_programs), r)]
    all_comb_docking_program = [item for r in range(1, len(docking_programs) + 1) 
                                 for item in combinations(sorted(docking_programs), r)]

    return list(product(all_comb_docking_program, all_comb_scoring_function))


def download_and_extract_zip(url, extract_to):
    """
    Download a ZIP file from a URL and extract its contents into a specific folder.

    Args:
        url (str): The URL of the ZIP file to download.
        extract_to (str): The directory path to extract the contents of the ZIP file.
    Returns:
        Downloaded CSV file in the extract_to directory
    """
    # Ensure the directory exists where files will be extracted
    if not os.path.exists(extract_to):
        os.makedirs(extract_to)

    # Download the file
    response = requests.get(url)
    if response.status_code == 200:
        zip_path = os.path.join(extract_to, 'temp.zip')
        with open(zip_path, 'wb') as f:
            f.write(response.content)
        print("Downloaded the ZIP file successfully.")

        # Extract the ZIP file
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        print(f"Extracted the ZIP file contents to {extract_to}")

        # Clean up the temporary ZIP file
        os.remove(zip_path)
        print("Removed the temporary ZIP file.")
    else:
        print("Failed to download the file. Status code:", response.status_code)