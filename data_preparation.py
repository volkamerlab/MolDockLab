from rdkit.Chem import PandasTools
import pandas as pd
import os
import csv
from rdkit import Chem
import multiprocessing
from rdkit import RDLogger
import re



def run_gypsumdl(ligand_library, prepared_library_path):
    """
    Run gypsum_dl to generate 3D conformations of ligands
    @Params:
    ligand_library: Path to ligand's library
    prepared_library_path: Path to prepared library
    
    @Return: Path to output file
    """
    ncpus = multiprocessing.cpu_count()
    gypsum_dl_command = (
    'python software/gypsum_dl-1.2.0/run_gypsum_dl.py '
    f'-s {ligand_library} '
    f'-o {prepared_library_path.parent} '
    '--job_manager multiprocessing -p '
    f'{ncpus} -m 1 -t 10 --skip_alternate_ring_conformations '
    '--skip_making_tautomers --skip_enumerate_double_bonds '
    '--max_variants_per_compound 1 '
    )

    if prepared_library_path.name not in os.listdir(str(prepared_library_path.parent)):
        os.system(gypsum_dl_command)
        # Clean output data (Remove the first row) and remove old one
        gypsum_df = PandasTools.LoadSDF(
            str(prepared_library_path.parent / 'gypsum_dl_success.sdf'),
            idName='ID',
            molColName='Molecule',
            strictParsing=True)

        cleaned_df = gypsum_df.iloc[1:, :]
        cleaned_df = cleaned_df[['Molecule', 'ID']]

        PandasTools.WriteSDF(
            cleaned_df,
            str(prepared_library_path),
            idName='ID',
            molColName='Molecule',
            properties=cleaned_df.columns)
        os.remove(str(prepared_library_path.parent / 'gypsum_dl_success.sdf'))
    else:
        print("Molecules are already prepared")
