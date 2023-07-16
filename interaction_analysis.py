from plipify.fingerprints import InteractionFingerprint
from plipify.visualization import (
    fingerprint_barplot, fingerprint_heatmap, fingerprint_table, 
    fingerprint_nglview, PymolVisualizer, nglview_color_side_chains_by_frequency,
    fingerprint_writepdb
)
import os
from pathlib import Path
from plipify.core import Structure
from tqdm.auto import tqdm
import glob 
from pymol import cmd
from utility import sdf_preprocessing



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
    sdfs_path = list(ligands_path.glob("*.sdf"))

    sdf_preprocessing(sdfs_path)
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
    display(fingerprint_heatmap(fp_focused))
    display(fingerprint_table(fp_focused))
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
