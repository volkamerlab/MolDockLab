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
from rdkit import Chem
from rdkit.Chem.rdMolAlign import GetBestRMS
from rdkit.Chem import AllChem, PandasTools


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

def create_pymol_image(fp, structure, output):

    pdb_interactions = fingerprint_writepdb(fingerprint_df=fp, structure=structure, output_path=output, ligand=True, summed=True, verbose=False)
    spectrum_colors = {
    "hydrophobic": "white_forest", 
    "hbond-don": "white_deepblue", 
    "hbond-acc": "white_olive", 
    "saltbridge": "white_purple", 
    "pistacking": "white_brown", 
    "pication": "white_lightpink", 
    "halogen": "white_gray", 
    "summed_interactions": "white_yellow"
}
    for interaction_name, interaction_structure in pdb_interactions.items():

        v = PymolVisualizer(pdb=interaction_structure._path, verbose=False)
        v.set_style()
        v.create_image(
            surface=True, ligand_col="cyan", spectrum_col=spectrum_colors[interaction_name], 
            viewport_x=900, viewport_y=900
        )
        v.render(name=f"{interaction_name}_pymol_image", save_path=str(output))


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
    # display(fp_focused.style.background_gradient(axis=None, cmap="YlGnBu"))
    display(fingerprint_barplot(fp_focused))
    # display(fingerprint_heatmap(fp_focused))
    # display(fingerprint_table(fp_focused))


def create_2dposeview(docked_group, docking_method):

    '''''
    This function takes a path of sdf files and create 2D interaction images for each sdf file
    @Param :
    docked_group --> name of the sdf file without the extension
    @Output :
    2D interaction images for each sdf file in 2D_interactions folder
    '''''
    if len(os.listdir(f"dcc_data/{docked_group}/2D_interactions")) != 0:
        print(f"Skipping {docked_group}, already exists")
        return
    if docking_method == 'gnina':
        predicted_score = 'CNNaffinity'
    elif docking_method == 'seesar':
        predicted_score = 'BIOSOLVEIT.HYDE_ESTIMATED_AFFINITY_LOWER_BOUNDARY [nM]'

    
    # if os.path.exists(f"dcc_data/{docked_group}/2D_interactions"):
    #     return
    #create folder for the photos
    os.makedirs(f"dcc_data/{docked_group}/2D_interactions", exist_ok=True)
    #create list of all sdf files in the folder
    
    sdf_files = glob.glob(f"dcc_data/{docked_group}/*.sdf")
    for sdf in sdf_files:
        try:
            predicted_affinity = float(PandasTools.LoadSDF(sdf)[predicted_score][0])
            float(PandasTools.LoadSDF('dcc_data/docked_gnina_pose_A_hittwo_30pose/HIPS6706_11.sdf')['CNNaffinity'][0])
        except:
            print(f"Error in {sdf}, no {predicted_score} column")
        #if 2d_interactions folder is not empty, skip the sdf file

        os.system(f"./software/poseview-1.1.2-Linux-x64/poseview -l {sdf} -p data/A/protein_protoss_noligand.pdb -o dcc_data/{docked_group}/2D_interactions/{Path(sdf).stem}.png -t {Path(sdf).stem}:{predicted_affinity:.2f}")


def rmsd_calculation(
        docking_method,
        number_of_poses
):
    ref = Chem.SDMolSupplier('data/A/ref_ligand.sdf')[0]
    rmsds = {}

    # Load molecules
    if docking_method == 'gnina':
        for i in range(1, int(number_of_poses)+1):
            mol = Chem.SDMolSupplier(f'dcc_data/docked_gnina_pose_A_hitone_30pose/HIPS6790_{i}.sdf')[0]
            
            
        # Need to have the same atom ordering
            mol = AllChem.AssignBondOrdersFromTemplate(ref, mol)

            rmsd = GetBestRMS(ref, mol)
            rmsds[i] = float(rmsd)

            # print(f'RMSD of pose {i}:', rmsd)
        return rmsds

    if docking_method == 'seesar':
        for i in list(glob.glob('dcc_data/docked_seesar_pose_A_hitone_30pose/HIPS6790*.sdf')):
            mol = Chem.SDMolSupplier(i)[0]
        # Need to have the same atom ordering
            mol = AllChem.AssignBondOrdersFromTemplate(ref, mol)
            pose = Path(i).stem.split('_')[2]
            rmsd = GetBestRMS(ref, mol)
            
            rmsds[pose] = float(rmsd)
        return rmsds