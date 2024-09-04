import glob
import json
import os
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from PIL import Image
from software.plipify.plipify.core import Structure
from software.plipify.plipify.fingerprints import InteractionFingerprint
from software.plipify.plipify.visualization import fingerprint_barplot
from pymol import cmd
from rdkit import Chem
from rdkit.Chem import AllChem, PandasTools
from tqdm.auto import tqdm


def plipify_fp_interaction(
        ligands_path:Path,
        protein_path:Path,
        protein_name: str,
        chains: list,
        output_file:Path
        ) -> dict:
    '''
    This function loads ligands and protein using pymol script commands and save both protein and ligand as a complex as pdb file.
    It splits Chain C and D to separate pdb file and change ligand according to chain.

    Args:
        ligands_path: single ligand path or multiple ligand paths in a list
        protein_path: path to protein in pdb format
        protein_name: name of protein
        chains: list of chains to split
        output_file: single or multiple PLIPify visualization, if single give a single sdf structure as a path, 
                if multiple give a list of sdf paths
    Returns:
        mol_interx_fp: Dict of all interactions
    '''
    if isinstance(ligands_path, Path):
        ligand_pdb_paths = [_sdf2pdb_preprocessing(ligands_path)]
    elif len(ligands_path) > 1:
        ligand_pdb_paths = [_sdf2pdb_preprocessing(sdf) for sdf in ligands_path]
        ligand_protein_cpx_paths = [
            ligand_protein_complex(ligand_pdb, protein_path, protein_name, chains)[0]
            for ligand_pdb in ligand_pdb_paths
        ]
        fp_focused = interaction_fp_generator(
            ligand_protein_cpx_paths, output_file)
        return fp_focused
    else:
        raise ValueError("No sdf files found")
    mol_interx_fp = {}
    for ligand_pdb in ligand_pdb_paths:
        interaction_fp = []
        ligand_protein_cpx_chains = ligand_protein_complex(
            ligand_pdb, protein_path, protein_name, chains)
        for cpx in ligand_protein_cpx_chains:
            try:
                prepared_cpx = [
                    Structure.from_pdbfile(
                        str(cpx), ligand_name="HIT")]
                fp = InteractionFingerprint().calculate_fingerprint(
                    prepared_cpx,
                    labeled=True,
                    as_dataframe=True,
                    remove_non_interacting_residues=True,
                    remove_empty_interaction_types=True,
                    ensure_same_sequence=False,
                )
            except BaseException:
                print(f"Empty fingerprint for {cpx.stem}")
                continue
            if fp is None:
                print(f"Empty fingerprint for {cpx.stem}")
                continue
            chain = cpx.stem.split("_")[-1]
            fp = [f'{i}{chain}' for i in fp.index]
            interaction_fp.extend(fp)
            # for pdb in ligand_protein_cpx_chains:

            os.remove(str(cpx))
            os.remove(f'/tmp/{cpx.stem}_protonated.pdb')
            # os.remove(str(cpx))
        mol_interx_fp[ligand_pdb.stem] = interaction_fp
    # shutil.rmtree(ligands_path[0].parent)
    return mol_interx_fp


def interaction_fp_generator(complex_path:Path, output_path:Path) -> pd.DataFrame:
    """
    This function takes a path of complex pdb files and create a fingerprint 
    of the interactions
    Args:
        complex_path: path of the complex pdb files
        output_path: path of the output png file
    Returns:
        fp_focused: DataFrame of the interactions
    """
    
    structures = [Structure.from_pdbfile(str(pdb),ligand_name="HIT") for pdb in tqdm(complex_path)]

    fp = InteractionFingerprint().calculate_fingerprint(
        structures,
        labeled=True,
        as_dataframe=True,
        remove_non_interacting_residues=True,
        remove_empty_interaction_types=True,
        ensure_same_sequence=False,
    )

    if not fp.values.shape[0]:
        raise ValueError("Fingerprint is empty!")

    fp_focused = fp[fp.sum(axis=1) > len(complex_path) // 10]

    fig = (fingerprint_barplot(fp_focused))
    fig.write_image(output_path)

    return fp_focused

def split_sdf_path(sdf_path: Path) -> list:
    """
    This function takes a path of sdf file and split it into multiple sdf files 
    with the same name of the ligand

    Args :
        sdf_path: path of sdf file

    Returns :
        path of a directory contains splitted molecules of sdf file
    """

    ligands_path = []
    output_dir = Path(sdf_path).parent / Path(sdf_path).stem
    suppl = Chem.SDMolSupplier(str(sdf_path))

    output_dir.mkdir(exist_ok=True)

    if os.listdir(output_dir):

        if len(os.listdir(output_dir)) == len(suppl):
            print("SDF files are already splitted")
            return list(output_dir.glob(f"*.sdf"))

    skipped = 0
    for mol in tqdm(suppl, desc="SDF splitting"):
        if mol is None:
            skipped += 1
            continue

        sdf_path = output_dir / f"{mol.GetProp('_Name')}.sdf"
        writer = Chem.SDWriter(str(sdf_path))
        writer.write(mol)
        writer.close()
        ligands_path.append(sdf_path)
    return ligands_path


def _sdf2pdb_preprocessing(sdf_file:Path) -> Path:
    """
    This function takes a path of sdf file and convert it to pdb file
    Args:
        sdf_file: path of sdf file
    Returns:
        path of the converted pdb file
    """
    pdb_path = sdf_file.parent / f"{sdf_file.stem}.pdb"
    cmd.load(sdf_file)
    cmd.alter('resi 0', 'resi = 287')
    cmd.alter('resn UNK', 'resn = "HIT"')
    cmd.alter('HETATM', 'chain="E"')
    cmd.save(pdb_path)
    cmd.delete("all")
    return pdb_path

    # load protein and ligand and save as pdb file


def ligand_protein_complex(
        ligand_path:Path, 
        protein_path:Path, 
        protein_name: str, 
        chains: list
        ) -> list:
    """
    This function takes a path of ligand and protein and save them as a complex pdb file
    Args:
        ligand_path: path of ligand in pdb format
        protein_path: path of protein in pdb format
        protein_name: name of protein
        chains: list of chains to split
    Returns:
        list of paths of the complex pdb files
    """
    ligand_name = ligand_path.stem
    ligand_protein_dir = ligand_path.parent / "ligand_protein_complex"

    ligand_protein_cpx_chains = []
    ligand_protein_dir.mkdir(exist_ok=True)
    for i, chain_id in enumerate(chains):
        complex_name = f"{ligand_name}_{protein_name}_{chain_id}.pdb"
        pdb_output = ligand_protein_dir / complex_name

        if os.path.exists(pdb_output):
            ligand_protein_cpx_chains.append(pdb_output)
            continue

        if i == 0:
            cmd.load(protein_path)
            cmd.load(ligand_path, "LIG")
            cmd.alter("all", "q=1.0")

        cmd.select(f'chain_{chain_id}E', f'chain {chain_id}+E')
        cmd.save(pdb_output, f'chain_{chain_id}E')

        # open pdb file and remove line starts with TER and write it at after
        # line starts with HETATM.
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
        ligand_protein_cpx_chains.append(pdb_output)
    os.remove(ligand_path)
    cmd.delete("all")
    return ligand_protein_cpx_chains


def read_interactions_json(json_file:Path, output_file:Path) -> pd.DataFrame:
    """
    This function reads the interactions from a JSON file and convert it to a CSV file
    Args:
        json_file: path of the JSON file
        output_file: path of the output CSV file
    Returns:
        interactions_df: DataFrame of the interactions
    """
    if os.path.exists(output_file):
        print('Interactions are converted to CSV file.')
        df = pd.read_csv(output_file)
        return df
    with open(json_file, 'r') as file:
        interactions_dict = json.load(file)
    flattened = [(cpd, resi)
                 for resi, cpds in interactions_dict.items() for cpd in cpds]
    flattened_df = pd.DataFrame(flattened, columns=['Poses', 'Residues'])
    interactions_df = flattened_df.pivot_table(
        index='Poses', columns='Residues', aggfunc=len, fill_value=0)
    interactions_df.to_csv(output_file)
    return interactions_df


def indiviudal_interaction_fp_generator(
        sdfs_path: list[Path],
        protein_file: Path,
        protein_name: str,
        included_chains: list,
        output_dir: Path
        ) -> dict:
    """
    This function takes a list of sdf files and generate a fingerprint of the interactions
    Args:
        sdfs_path: list of sdf files
        protein_file: path of the protein in pdb format
        protein_name: name of the protein
        included_chains: list of chains to split
        output_dir: path of the output directory
    Returns:
        allposes_interaction_fp: Dict of all interactions
    """
    if output_dir.is_dir():
        print('Interactions for all poses are already executed')
        return output_dir
    print(f"Generating interactions for {len(sdfs_path)} poses\n\n")

    allposes_interaction_fp = {}
    for i, sdf in enumerate(sdfs_path):
        fp = plipify_fp_interaction(
            sdf, protein_file, protein_name, included_chains, output_dir)
        allposes_interaction_fp.update(fp)
        if i % 1000 == 0 and i != 0:
            _write_json(allposes_interaction_fp, str(output_dir))

    _write_json(allposes_interaction_fp, str(output_dir))
    return allposes_interaction_fp


def _write_json(allposes_interaction_fp: pd.DataFrame, output_path: str):
    """
    This function writes the interactions to a JSON file
    Args:
        allposes_interaction_fp: DataFrame of the interactions
        output_path: path of the output JSON file
    """
    residue_to_compounds = {}
    for compound, residues in allposes_interaction_fp.items():
        for residue in residues:
            if residue not in residue_to_compounds:
                residue_to_compounds[residue] = []
            residue_to_compounds[residue].append(compound)

    with open(output_path, "w") as json_file:
        json.dump(residue_to_compounds, json_file, indent=4)

    print(f"JSON file saved to {output_path}")

def interactions_aggregation(
        interactions_df: pd.DataFrame,
        important_interactions: list,
        id_column:str='ID'
        ) -> pd.DataFrame:
    """
    This function aggregates the interactions based on the important interactions
    Args:
        interactions_df: DataFrame of the interactions
        important_interactions: list of important interactions
        id_column: column name of the ID
    Returns:
        aggregated_df: DataFrame of the aggregated interactions
    """
    interactions_df['id'] = interactions_df[id_column].str.split('_').str[0]
    aggregated_df = interactions_df.groupby('id').sum()
    return aggregated_df[important_interactions].reset_index()


# def interx_plot(residue_df, imp_interactions):
#     if not imp_interactions:
#         imp_interactions = residue_df.columns
#     sns.set_theme(style="whitegrid")
#     plt.figure(figsize=(20, 10))
#     ax = sns.barplot(x=imp_interactions,
#                      y=residue_df.loc['frequency', imp_interactions])
#     ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
#     ax.set_title('Frequency of each residue in HIPS dataset')
#     ax.set_ylabel('Frequency')
#     ax.set_xlabel('Residue')
#     plt.tight_layout()
#     plt.show()

def actives_extraction(
        test_set_docked_path,
        merged_rescoring_path,
        docking_tool):
    """
    This function filters the actives from the docking poses based on a threshold 
    of the true value and the selected docking tool
    Args:
        test_set_docked_path: str, path to the sdf file of the docking poses
        merged_rescoring_path: str, path to the csv file of the merged rescoring results
        docking_tool: str, name of the docking tool used
    Returns:
        actives_path: str, path to the sdf file of the actives
    """
    docked_df = PandasTools.LoadSDF(str(test_set_docked_path))
    df_scores = pd.read_csv(str(merged_rescoring_path))
    actives_poses = df_scores[df_scores['activity_class'] == 1]
    if isinstance(docking_tool, str):
        actives_id = actives_poses[actives_poses['docking_tool']
                                   == docking_tool]['ID'].tolist()
    if isinstance(docking_tool, list):
        actives_id = actives_poses[actives_poses['docking_tool'].isin(
            docking_tool)]['ID'].tolist()
    print(f"Number of active compounds: {len(actives_id)}")
    df_filtered = docked_df[docked_df['ID'].isin(actives_id)]
    actives_path = test_set_docked_path.parent / "docked_actives.sdf"
    PandasTools.WriteSDF(
        df_filtered,
        str(actives_path),
        molColName='ROMol',
        idName='ID')
    return actives_path
