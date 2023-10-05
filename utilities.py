import pandas as pd
from rdkit import Chem
import csv
from Bio import SeqIO
from Bio.PDB import PDBParser
from Bio.Data import IUPACData
import subprocess
import os

def run_command(cmd):
    subprocess.call(cmd, 
                    shell=True, 
                    # stdout=subprocess.DEVNULL, 
                    # stderr=subprocess.STDOUT
                    )

def get_ligand_coordinates_centers(ligand_molecule):
    '''This function returns the center of the ligand molecule in x, y and z coordinates
    @param ligand_molecule: ligand molecule in pdb or sdf format

    @return: center of the ligand molecule in x, y and z coordinates
    '''
    if ligand_molecule.endswith('.sdf'):
        supplier = Chem.SDMolSupplier(
            ligand_molecule, sanitize=False, removeHs=False)
        mol = supplier[0]
    elif ligand_molecule.endswith('.pdb'):
        mol = Chem.MolFromPDBFile(
            ligand_molecule,
            removeHs=False)
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


def pdb2resname_resno (input_file):

    '''This function extracts the residue name and residue number from the protein file and returns a list of tuples of resname and resnumber
    
    @param input_file: protein file in pdb format

    @return: a list of tuples of chain_ID, resname and resnumber
    '''

    resname_resnumbers = []
    # Read the lines of the file
    with open(input_file, 'r') as pdb_file:
        pdb_lines = pdb_file.readlines()
        for pdb_line in pdb_lines:
            if pdb_line.startswith('ATOM'):
                pdb_line_list = pdb_line.split()
                if (pdb_line_list[4] , pdb_line_list[3] , pdb_line_list[5]) in resname_resnumbers:
                    continue

                resname_resnumbers.append(( pdb_line_list[4] , pdb_line_list[3] , pdb_line_list[5]))
    
    return (resname_resnumbers)

def extract_binding_pocket (protein_file, ref_file, output_file):
    '''This function extracts the binding pocket residues from the protein file and writes the residue number to a csv file''

    @param protein_file: protein file in pdb format
    @param ref_file: reference file in pdb format
    @param output_file: output file in csv format

    @return: a csv file with residue numbers of binding pocket residues in results folder
    '''

    whole = pdb2resname_resno(str(protein_file))
    ref = pdb2resname_resno(str(ref_file))

    indices_dict = {element: [] for element in ref}

    for i, element in enumerate(whole):
        if element in indices_dict:
            indices_dict[element] = (whole.index(element))

    # print(indices_dict)
    with open(str(output_file), 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        # Write each item in the list to the CSV as a separate row
        for key, item in indices_dict.items():
            writer.writerow([item])

def any_in_list(list1, list2):
    return any(i in list2 for i in list1)

def split_sdf(file_path, scoring_function, thread_num):
    """
    Split SDF file into chunks
    @Params:
    file_path: Path to SDF file
    thread_num: Number of threads to split the file into

    @Return: List of paths to splitted files
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
        
        split_folder = file_path.parent/  'sdf_split'
        split_folder.mkdir(exist_ok=True)

        # write chunk to file
        with open(str(split_folder / f"sdf_{i}.sdf"), "w") as out_file:
            out_file.write('$$$$\n'.join(molecules[start_index:end_index]) + '$$$$\n')
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
        '''''
        The function converts the protein file to pdbqt format in case of SCORCH SF
        and to mol2 format in case of CHEMPLP SF

        @param protein_file: protein file in pdb format
        @param rescore_programs: list of rescoring programs

        @return: None
        '''''
        if "chemplp" in rescore_programs:
            if f'{protein_file.stem}.mol2' in os.listdir(protein_file.parent):
                print('protein is already converted to mol2')
            else:
                # convert protein to pdbqt
                run_command(f"obabel -ipdb {str(protein_file)}"
                        f" -O {str(protein_file.parent / f'{protein_file.stem}.mol2')}")

        elif "scorch" in rescore_programs:
            
            if f'{protein_file.stem}.pdbqt' in os.listdir(protein_file.parent):
                print('protein is already converted to pdbqt')
            else:
                # convert protein to pdbqt
                run_command(f"obabel -ipdb {str(protein_file)}"
                        f" -O {str(protein_file.parent / f'{protein_file.stem}.pdbqt')}"
                        " --partialcharges gasteiger -p -h")
                


def split_list(input_list, num_splits):
    """Split a list into n equal parts."""
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