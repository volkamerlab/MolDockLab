import multiprocessing
import os

import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, PandasTools
from pathlib import Path

def minimize_and_select_most_stable(row: pd.Series, numConfs: int = 10) -> pd.DataFrame:
    """
    Minimize the energy of a molecule and select the most stable conformer
    Args:
        row (pandas.Series): Row of a pandas DataFrame with the SMILES string and the ID of the molecule
        numConfs (int): Number of conformers to generate
    Returns:
        mol (rdkit.Chem.rdchem.Mol): Molecule with the most stable conformer set as the active one
    """
    print(f"Processing Failed Gypsum-DL ID: {row.ID}")
    mol = Chem.MolFromSmiles(row.SMILES)
    if mol is None:
        print(f"Failed to parse ID: {row.ID}")
        return None
    mol = Chem.AddHs(mol)
    AllChem.EmbedMultipleConfs(mol, numConfs=numConfs)
    lowest_energy = float('inf')
    most_stable_conformer_id = None

    for conf_id in range(mol.GetNumConformers()):
        # For each conformer, get a force field
        ff = AllChem.MMFFGetMoleculeForceField(
            mol, AllChem.MMFFGetMoleculeProperties(mol), confId=conf_id)
        energy = ff.Minimize()
        energy = ff.CalcEnergy()
        if energy < lowest_energy:
            lowest_energy = energy
            most_stable_conformer_id = conf_id
    if most_stable_conformer_id is not None:
        # Set the most stable conformer as the active one for the molecule
        mol.SetProp('_MostStableConformerID', str(most_stable_conformer_id))

    new_row = pd.DataFrame({"ID": [row.ID], "Molecule": [mol]})
    return new_row


def run_gypsumdl(
        ligand_library: Path, 
        prepared_library_path: Path, 
        id_column : str ='ID'
        ) -> Path:
    """
    Run gypsum_dl to generate 3D conformations of ligands
    Args:
        ligand_library: Path to ligand's library
        prepared_library_path: Path to prepared library

    Return: 
        Path to output file
    """
    ncpus = multiprocessing.cpu_count() - 1
    gypsum_dl_command = (
        'python software/gypsum_dl-1.2.1/run_gypsum_dl.py'
        f' -s {ligand_library}'
        f' -o {prepared_library_path.parent}'
        ' --job_manager multiprocessing'
        f' -p {ncpus}'
        ' -m 1'
        ' -t 10'
        ' --min_ph 6.5'
        ' --max_ph 7.5'
        ' --pka_precision 1'
        ' --max_variants_per_compound 1'
    )

    if prepared_library_path.name not in os.listdir(str(prepared_library_path.parent)):
        os.system(gypsum_dl_command)

        # Clean output data (Remove the first row) and remove old one
        gypsum_df = PandasTools.LoadSDF(
            str(prepared_library_path.parent / 'gypsum_dl_success.sdf'),
            idName=id_column,
            molColName='Molecule',
            strictParsing=True
        )
        cleaned_df = gypsum_df.iloc[1:, :]
        cleaned_df = cleaned_df[['Molecule', id_column]]
        failed_file = prepared_library_path.parent / "gypsum_dl_failed.smi"
        if failed_file.exists():
            failed_cpds = pd.read_csv(
                failed_file,
                delimiter="\t",
                header=None,
                names=["SMILES","ID"])
            for _, row in failed_cpds.iterrows():
                failed_row = minimize_and_select_most_stable(row)
                if failed_row.Molecule is not None:
                    cleaned_df = pd.concat(
                        [cleaned_df, failed_row], ignore_index=True)

        PandasTools.WriteSDF(
            cleaned_df,
            str(prepared_library_path),
            idName=id_column,
            molColName='Molecule',
            properties=cleaned_df.columns
        )
        os.remove(str(prepared_library_path.parent / 'gypsum_dl_success.sdf'))
        

        if os.path.exists(str(failed_file)):
            os.remove(str(failed_file))
            return prepared_library_path
    else:
        print("Molecules are already prepared by Gypsum-DL")
