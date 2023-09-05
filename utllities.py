import pandas as pd
from rdkit import Chem
import csv
from Bio import SeqIO
from Bio.PDB import PDBParser
from Bio.Data import IUPACData

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
