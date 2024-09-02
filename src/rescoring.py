
import csv
import os
# from software.RTMScore.rtmscore_modified import *
import subprocess
import time
import shutil
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import pandas as pd
from rdkit.Chem import PandasTools

from src.preprocessing import plants_preprocessing
from src.utilities import (any_in_list, pdb_converter,
                       pocket_coordinates_generation, run_command, split_sdf)


def rescoring_function(
    rescoring_programs: list,
    protein_path: Path,
    docked_library_path: Path,
    ref_file: Path,
    ncpu: int
):
    """
    This function is the high-level function to deploy all scoring functions. It takes the following arguments:
    Args:
        rescoring_programs (list): The list of rescoring programs to be used
        protein_path (Path): The path to the protein file
        docked_library_path (Path): The path to the docked library
        ref_file (Path): The path to the reference ligand file
        ncpu (int): The number of CPUs to use
    Returns:
        Saved rescoring results in the rescoring_results folder as csv file, besides the individual rescoring results in the rescoring_results folder
    """
    rescoring_dict = {

        'smina_affinity': _smina_rescoring,
        'ad4': _ad4_rescoring,
        'linf9': _linf9_rescoring,
        'vinardo': _vinardo_rescoring,
        'chemplp': _chemplp_rescoring,
        'hyde': _hyde_rescoring,
        'vina_hydrophobic': _vina_hydrophobic_rescoring,
        'vina_intra_hydrophobic': _vina_intra_hydrophobic_rescoring,
        'rtmscore': _rtmscore_rescoring,
        'rfscore_v1': _rfscore_V1_rescoring,
        'rfscore_v2': _rfscore_V2_rescoring,
        'rfscore_v3': _rfscore_v3_rescoring,
        'cnnscore': _gnina_score_rescoring,
        'cnnaffinity': _gnina_affinity_rescoring,
        'scorch': _scorch_rescoring,
    }
    # Create folder for rescoring results
    results_folder = docked_library_path.parent / 'rescoring_results'
    results_folder.mkdir(exist_ok=True)
    num_cpus = ncpu

    # convert protein and ligand to mol2 and pdbqt format in case of CHEMPLP
    # and SCORCH respectively
    pdb_converter(protein_path, rescoring_programs)
    pdb_converter(ref_file, rescoring_programs)

    for program in rescoring_programs:
        program = program.lower()
        splitted_file_paths = split_sdf(docked_library_path, program, num_cpus)
        output_folder = results_folder / program
        output_folder.mkdir(exist_ok=True)

        print(f"\n\nNow rescoring with {program.upper()} ... ⌛⌛ ")
        if os.listdir(output_folder):
            print(f'{program} is already excuted')
            if f'{program}_rescoring.csv' in os.listdir(output_folder):
                print(f'{program} is already read')
                continue

        elif program in rescoring_dict.keys():
            # Run scoring functions in parellel

            print(f'Running {program} in parallel')
            # calculate the run time for each program
            start_time = time.time()

            with ProcessPoolExecutor(max_workers=num_cpus) as executor:
                _ = [
                    executor.submit(
                        run_command,
                        rescoring_dict[program](
                            protein_path,
                            file_path,
                            ref_file,
                            output_folder / f'{program}_{i}.sdf'
                        )
                    )
                    for i, file_path in enumerate(splitted_file_paths)
                ]
            end_time = time.time()    # End time

            duration = end_time - start_time
            print(f"\n\nThe {program} took {duration} seconds to run.")
        _read_rescoring_results(results_folder, program)

    _merge_rescoring_results(results_folder, rescoring_programs)
    shutil.rmtree(splitted_file_paths[0].parent)    
    _clean_rescoring_results(rescoring_programs, results_folder)

def _ad4_rescoring(
        protein_path: Path,
        docked_library_path: Path,
        ref_file: Path,
        output_path: Path,
) -> str:
    """
    This function for AD4 rescoring function and it takes the following arguments:
    Args:
        protein_path (Path): The path to the protein file
        docked_library_path (Path): The path to the docked library
        ref_file (Path): The path to the reference ligand file
        output_path (Path): The path to the output file
    Returns:
        The command to run the AD4 rescoring function
    """
    return (
        './software/gnina'
        f' --receptor {protein_path}'
        f' --ligand {str(docked_library_path)}'
        f' --out {str(output_path)}'
        f' --autobox_ligand {str(ref_file)}'
        ' --score_only'
        ' --scoring ad4_scoring --cnn_scoring none'
    )


def _smina_rescoring(
        protein_path: Path,
        docked_library_path: Path,
        ref_file: Path,
        output_path: Path,
) -> str:
    """
    This function for SMINA rescoring function and it takes the following arguments:
    Args:
        protein_path (Path): The path to the protein file
        docked_library_path (Path): The path to the docked library
        ref_file (Path): The path to the reference ligand file
        output_path (Path): The path to the output file
    Returns:
        The command to run the SMINA rescoring function
    """

    if any_in_list(['cnnaffinity', 'cnnscore'],
                   os.listdir((output_path.parent).parent)):
        print(f'{docked_library_path.name} is already excuted')
        return
    else:
        return (
            './software/gnina'
            f' --receptor {protein_path}'
            f' --ligand {str(docked_library_path)}'
            f' --out {str(output_path)}'
            f' --autobox_ligand {str(ref_file)}'
            ' --score_only'
            ' --cnn crossdock_default2018 --no_gpu'
        )


def _gnina_score_rescoring(
        protein_path: Path,
        docked_library_path: Path,
        ref_file: Path,
        output_path: Path,
) -> str:
    """
    This function for the score of GNINA rescoring function and it takes the following arguments:
    Args:
        protein_path (Path): The path to the protein file
        docked_library_path (Path): The path to the docked library
        ref_file (Path): The path to the reference ligand file
        output_path (Path): The path to the output file
    Returns:
        The command to run the the score of GNINA rescoring function
    """

    if any_in_list(['cnnaffinity', 'smina_affinity'],
                   os.listdir((output_path.parent).parent)):
        print(f'{output_path.name} is already excuted')
        return
    return (
        './software/gnina'
        f' --receptor {str(protein_path)}'
        f' --ligand {str(docked_library_path)}'
        f' --out {str(output_path)}'
        f' --autobox_ligand {str(ref_file)}'
        ' --score_only'
        ' --cnn crossdock_default2018 --no_gpu'
    )


def _gnina_affinity_rescoring(
        protein_path: Path,
        docked_library_path: Path,
        ref_file: Path,
        output_path: Path,
) -> str:
    """
    This function for GNINA rescoring function and it takes the following arguments:
    Args:
        protein_path (Path): The path to the protein file
        docked_library_path (Path): The path to the docked library
        ref_file (Path): The path to the reference ligand file
        output_path (Path): The path to the output file
    Returns:
        The command to run the GNINA rescoring function
    """
    if any_in_list(['cnnscore', 'smina_affinity'],
                   os.listdir((output_path.parent).parent)):
        print(f'{output_path.name} is already excuted')
        return
    return (
        './software/gnina'
        f' --receptor {str(protein_path)}'
        f' --ligand {str(docked_library_path)}'
        f' --out {str(output_path)}'
        f' --autobox_ligand {str(ref_file)}'
        ' --score_only'
        ' --cnn crossdock_default2018'
    )


def _vinardo_rescoring(
        protein_path: Path,
        docked_library_path: Path,
        ref_file: Path,
        output_path: Path,
) -> str:
    """
    This function for Vinardo rescoring function and it takes the following arguments:
    Args:
        protein_path (Path): The path to the protein file
        docked_library_path (Path): The path to the docked library
        ref_file (Path): The path to the reference ligand file
        output_path (Path): The path to the output file
    Returns:
        The command to run the Vinardo rescoring function
    """
    return (
        './software/gnina'
        f' --receptor {protein_path}'
        f' --ligand {str(docked_library_path)}'
        f' --out {str(output_path)}'
        f' --autobox_ligand {str(ref_file)}'
        ' --score_only'
        ' --scoring vinardo --cnn_scoring none'
    )


def _chemplp_rescoring(
        protein_path: Path,
        docked_library_path: Path,
        ref_file: Path,
        output_path: Path,
) -> str:
    """
    This function for CHEMPLP rescoring function and it takes the following arguments:
    Args:
        protein_path (Path): The path to the protein file
        docked_library_path (Path): The path to the docked library
        ref_file (Path): The path to the reference ligand file
        output_path (Path): The path to the output file
    Returns:
        The command to run the CHEMPLP rescoring function
    """
    plants_search_speed = 'speed1'
    ants = '20'

    protein_mol2, mols_library_mol2, ref_ligand_mol2 = plants_preprocessing(
        protein_path, docked_library_path, ref_file)
    center_x, center_y, center_z, radius = pocket_coordinates_generation(
        protein_mol2, ref_ligand_mol2, pocket_coordinates_path='bindingsite.def')
    print(f"Center of the pocket is: {center_x}, {center_y}, {center_z} with radius of {radius}")

    chemplp_config = [
        '# search algorithm\n',
        f'search_speed {plants_search_speed}\n',
        f'aco_ants {ants}\n',
        'flip_amide_bonds 0\n',
        'flip_planar_n 1\n',
        'force_flipped_bonds_planarity 0\n',
        'force_planar_bond_rotation 1\n',
        'rescore_mode simplex\n',
        'flip_ring_corners 0\n',
        '# scoring functions\n',
        '# Intermolecular (protein-ligand interaction scoring)\n',
        'scoring_function chemplp\n',
        'outside_binding_site_penalty 50.0\n',
        'enable_sulphur_acceptors 1\n',
        '# Intramolecular ligand scoring\n',
        'ligand_intra_score clash2\n',
        'chemplp_clash_include_14 1\n',
        'chemplp_clash_include_HH 0\n',
        '# input\n',
        f'protein_file {str(protein_mol2)}\n',
        f'ligand_file {str(mols_library_mol2)}\n',
        '# output\n',
        f'output_dir {str(output_path.parent / output_path.stem)}\n',
        '# write single mol2 files (e.g. for RMSD calculation)\n',
        'write_multi_mol2 1\n',
        '# binding site definition\n',
        f'bindingsite_center {str(center_x)} {str(center_y)} {str(center_z)}\n',
        f'bindingsite_radius {str(radius)}\n',
        '# cluster algorithm\n',
        'cluster_structures 10\n',
        'cluster_rmsd 2.0\n',
        '# write\n',
        'write_ranking_links 0\n',
        'write_protein_bindingsite 1\n',
        'write_protein_conformations 1\n',
        'write_protein_splitted 1\n',
        'write_merged_protein 0\n',
        '####\n']

    # Write config file
    chemplp_rescoring_config_path_config = docked_library_path.parent / f'{output_path.stem}.config'

    with chemplp_rescoring_config_path_config.open('w') as configwriter:
        configwriter.writelines(chemplp_config)

    # Run PLANTS docking
    return f'./software/PLANTS --mode rescore {str(chemplp_rescoring_config_path_config)}'


def _linf9_rescoring(
        protein_path: Path,
        docked_library_path: Path,
        ref_file: Path,
        output_path: Path,
) -> str:
    """
    This function for LinF9 rescoring function and it takes the following arguments:
    Args:
        protein_path (Path): The path to the protein file
        docked_library_path (Path): The path to the docked library
        ref_file (Path): The path to the reference ligand file
        output_path (Path): The path to the output file
    Returns:
        The command to run the LinF9 rescoring function
    """
    return (
        f'./software/smina.static'
        f' --receptor {str(protein_path)}'
        f' --ligand {str(docked_library_path)}'
        f' --out {str(output_path)}'
        f' --autobox_ligand {str(ref_file)}'
        ' --scoring Lin_F9 --score_only'
    )


def _rtmscore_rescoring(
        protein_path: Path,
        docked_library_path: Path,
        ref_file: Path,
        output_path: Path,
):
    """
    This function for RTMScore rescoring function and it takes the following arguments:
    Args:
        protein_path (Path): The path to the protein file
        docked_library_path (Path): The path to the docked library
        ref_file (Path): The path to the reference ligand file
        output_path (Path): The path to the output file
    Returns:
        The command to run the RTMScore rescoring function
    """
    RTMScore_pocket = str(protein_path).replace('.pdb', '_pocket.pdb')
    number_of_ligand = docked_library_path.stem.split('_')[-1]
    ref_file = str(ref_file).replace('.pdb', '.sdf')
    if not os.path.exists(RTMScore_pocket):
        print('Pocket is not found, generating the pocket first and rescore')
        return (
            f'python software/RTMScore/example/rtmscore.py'
            f' -p {str(protein_path)}'
            f' -l {str(docked_library_path)}'
            f' -rl {str(ref_file)}'
            f' -o {str(output_path.parent / f"rtmscore_{number_of_ligand}")}'
            f' -gen_pocket'
            f' -c 10.0'
            ' -m software/RTMScore/trained_models/rtmscore_model1.pth'
        )
    return (
        f'python software/RTMScore/example/rtmscore.py'
        f' -p {str(RTMScore_pocket)}'
        f' -l {str(docked_library_path)}'
        f' -o {str(output_path.parent / f"rtmscore_{number_of_ligand}")}'
        ' -m software/RTMScore/trained_models/rtmscore_model1.pth'
    )


def _scorch_rescoring(
        protein_path: Path,
        docked_library_path: Path,
        ref_file: Path,
        output_path: Path,
):
    """
    This function for SCORCH rescoring function and it takes the following arguments:
    Args:
        protein_path (Path): The path to the protein file
        docked_library_path (Path): The path to the docked library
        ref_file (Path): The path to the reference ligand file
        output_path (Path): The path to the output file
    Returns:
        The command to run the SCORCH rescoring function
    """
    protein_path_pdqbt = str(protein_path).replace('.pdb', '.pdbqt')
    docked_library_file_pdqbt = str(
        docked_library_path).replace('.sdf', '.pdbqt')
    ref_ligand_pdqbt = str(ref_file).replace('.pdb', '.pdbqt')
    print(
        f'python software/SCORCH/scorch.py '
        f' --receptor {str(protein_path_pdqbt)} '
        f' --ligand {docked_library_file_pdqbt}'
        f" --ref_lig {str(ref_ligand_pdqbt)}"
        f' --out {str(output_path.parent / output_path.stem)}.csv'
        ' --return_pose_scores'
        ' --threads 1'
    )
    return (
        f'python software/SCORCH/scorch.py '
        f' --receptor {str(protein_path_pdqbt)} '
        f' --ligand {docked_library_file_pdqbt}'
        f" --ref_lig {str(ref_ligand_pdqbt)}"
        f' --out {str(output_path.parent / output_path.stem)}.csv'
        ' --return_pose_scores'
        ' --threads 1'
    )


def _rfscore_V1_rescoring(
        protein_path: Path,
        docked_library_path: Path,
        ref_file: Path,
        output_path: Path,
):
    """
    This function for RFScore_ver1 rescoring function and it takes the following arguments:
    Args:
        protein_path (Path): The path to the protein file
        docked_library_path (Path): The path to the docked library
        ref_file (Path): The path to the reference ligand file
        output_path (Path): The path to the output file
    Returns:
        The command to run the SCORCH rescoring function
    """
    return (
        f"oddt_cli {str(docked_library_path)}"
        f' --receptor {str(protein_path)}'
        f" --score rfscore_v1"
        f" -O {str(output_path)}"
        " -n 1"
    )


def _hyde_rescoring(
        protein_path: Path,
        docked_library_path: Path,
        ref_file: Path,
        output_path: Path,
):
    """
    This function for HYDE rescoring function and it takes the following arguments:
    Args:
        protein_path (Path): The path to the protein file
        docked_library_path (Path): The path to the docked library
        ref_file (Path): The path to the reference ligand file
        output_path (Path): The path to the output file
    Returns:
        The command to run the HYDE rescoring function
    """
    if ref_file.suffix == '.pdb':
        ref_file = ref_file.parent / f'{ref_file.stem}.sdf'
        if not os.path.exists(ref_file):
            print('convert pdb to sdf')
            pdb_to_sdf = f'obabel {str(ref_file.parent / ref_file.stem)}.pdb -O {str(ref_file)}'
            subprocess.run(pdb_to_sdf, shell=True)
    return (
        "software/hydescorer-2.0.0/hydescorer"
        f" -i {str(docked_library_path)}"
        f" -o {str(output_path)}"
        f" -p {str(protein_path)}"
        f" -r {str(ref_file)}"
        " --thread-count 1"
    )

def _rfscore_V2_rescoring(
        protein_path: Path,
        docked_library_path: Path,
        ref_file: Path,
        output_path: Path,
):
    """
    This function for RFscorevs_V2 rescoring rescoring function and it takes the following arguments:
    Args:
        protein_path (Path): The path to the protein file
        docked_library_path (Path): The path to the docked library
        ref_file (Path): The path to the reference ligand file
        output_path (Path): The path to the output file
    Returns:
        The command to run the RFscorevs_V2 rescoring rescoring function
    """
    return (
        f"oddt_cli {str(docked_library_path)}"
        f' --receptor {str(protein_path)}'
        f" --score rfscore_v2"
        f" -O {str(output_path)}"
        " -n 1"
    )

def _rfscore_v3_rescoring(
        protein_path: Path,
        docked_library_path: Path,
        ref_file: Path,
        output_path: Path,
):
    """
    This function for RFscorevs_V3 rescoring rescoring function and it takes the following arguments:
    Args:
        protein_path (Path): The path to the protein file
        docked_library_path (Path): The path to the docked library
        ref_file (Path): The path to the reference ligand file
        output_path (Path): The path to the output file
    Returns:
        The command to run the RFscorevs_V3 rescoring rescoring function
    """
    return (
        f"oddt_cli {str(docked_library_path)}"
        f' --receptor {str(protein_path)}'
        f" --score rfscore_v3"
        f" -O {str(output_path)}"
        " -n 1"
    )

def _vina_hydrophobic_rescoring(
        protein_path: Path,
        docked_library_path: Path,
        ref_file: Path,
        output_path: Path,
):
    """
    This function for Vina Hydrophobic rescoring function and it takes the following arguments:
    Args:
        protein_path (Path): The path to the protein file
        docked_library_path (Path): The path to the docked library
        ref_file (Path): The path to the reference ligand file
        output_path (Path): The path to the output file
    Returns:
        The command to run the Vina Hydrophobic rescoring function
    """
    # @TODO : check if the poses are already rescored through RF-Score-V3, if yes, copy the file to the output path
    if ((output_path.parent).parent / 'rfscore_v3' / 'rfscore_v3_0.sdf').is_file():
        print(f'{docked_library_path.name} is already excuted')
        return
    else:
        return (
            f"oddt_cli {str(docked_library_path)}"
            f' --receptor {str(protein_path)}'
            f" --score rfscore_v3"
            f" -O {str(output_path)}"
            " -n 1"
        )

def _vina_intra_hydrophobic_rescoring(
        protein_path: Path,
        docked_library_path: Path,
        ref_file: Path,
        output_path: Path,
):
    """
    This function for Vina Intra Hydrophobic rescoring function and it takes the following arguments:
    Args:
        protein_path (Path): The path to the protein file
        docked_library_path (Path): The path to the docked library
        ref_file (Path): The path to the reference ligand file
        output_path (Path): The path to the output file
    Returns:
        The command to run the Vina Intra Hydrophobic rescoring function
    """
    if ((output_path.parent).parent / 'rfscore_v3' / 'rfscore_v3_0.sdf').is_file():
        print(f'{output_path.name} is already excuted')
        return
    else:
        return (
            f"oddt_cli {str(docked_library_path)}"
            f' --receptor {str(protein_path)}'
            f" --score rfscore_v3"
            f" -O {str(output_path)}"
            " -n 1"
        )

def _read_sdf_values_and_names(sdf_file_path):
    with open(sdf_file_path, 'r') as file:
        all_records = []
        capture_data = False
        current_values = {}
        molecule_name = ""
        for line in file:
            line = line.strip()
            if line == "$$$$":
                all_records.append((molecule_name, current_values))
                capture_data = False
                current_values = {}
                molecule_name = ""
            elif capture_data:
                if line.startswith("> <"):
                    parts = line.split("> <")
                    if len(parts) == 2:
                        key = parts[1].split(">")[0]
                        value = next(file).strip()
                        current_values[key] = value

                if line.startswith(">  <"):
                    parts = line.split(">  <")
                    if len(parts) == 2:
                        key = parts[1].split(">")[0]
                        value = next(file).strip()
                        current_values[key] = value

            elif line == "M  END":
                capture_data = True
            else:
                if not molecule_name:  
                    molecule_name = line
        df = pd.DataFrame.from_records(
            all_records, columns=[
                'ID', 'Properties'])
        df = pd.concat([df.drop(['Properties'], axis=1), df['Properties'].apply(pd.Series)], axis=1)
    return df


def _read_rescoring_results(
        rescoring_results_path,
        rescore_program
        ):
    """
    This function reads the rescoring results and saves them as csv files
    Args:
        rescoring_results_path (str): The path to the rescoring results
    Returns:
        Saved rescoring results in the rescoring directory as csv file
    """
    dfs = []
    print('\n\nReading rescoring results ⌛ ...\n\n')

    try:
        if f'{rescore_program}_rescoring.csv' in os.listdir(
                rescoring_results_path):
            print(f'{rescore_program} is already read')
            return

        if rescore_program in ['cnnscore', 'cnnaffinity', 'smina_affinity']:
            for path in ['cnnscore', 'cnnaffinity', 'smina_affinity']:
                rescore_path = rescoring_results_path / path
                if os.path.exists(rescore_path) and len(
                        os.listdir(rescore_path)) > 1:
                    break

            for sdf in rescore_path.glob('*.sdf'):

                df = _read_sdf_values_and_names(sdf)
                # chech if df is empty
                if df.empty:
                    continue
                if 'cnnscore' == rescore_program:
                    df = df[['ID', 'CNNscore']]

                elif 'cnnaffinity' == rescore_program:
                    df = df[['ID', 'CNNaffinity']]

                elif 'smina_affinity' == rescore_program:
                    df = df[['ID', 'minimizedAffinity']]
                    df.rename(
                        columns={
                            'minimizedAffinity': 'smina_affinity'},
                        inplace=True)
                dfs.append(df)

        if 'ad4' == rescore_program:
            for sdf in (rescoring_results_path / rescore_program).glob('*.sdf'):
                df = _read_sdf_values_and_names(sdf)[['ID', 'minimizedAffinity']]
                df.rename(columns={'minimizedAffinity': 'ad4'}, inplace=True)
                dfs.append(df)

        if 'linf9' == rescore_program:
            for sdf in (rescoring_results_path / rescore_program).glob('*.sdf'):
                df = _read_sdf_values_and_names(sdf)[['ID', 'minimizedAffinity']]
                df.rename(columns={'minimizedAffinity': 'LinF9'}, inplace=True)
                dfs.append(df)

        if 'vinardo' == rescore_program:
            for sdf in (rescoring_results_path / rescore_program).glob('*.sdf'):
                df = _read_sdf_values_and_names(sdf)[['ID', 'minimizedAffinity']]
                df.rename(columns={'minimizedAffinity': 'Vinardo'}, inplace=True)
                dfs.append(df)

        if 'rfscore_v1' == rescore_program:
            for sdf in (rescoring_results_path / rescore_program).glob('*.sdf'):
                df = _read_sdf_values_and_names(sdf)[['ID', 'rfscore_v1']]
                # df.rename(columns={'minimizedAffinity': 'Vinardo'}, inplace=True)
                dfs.append(df)

        if 'rfscore_v2' == rescore_program:
            for sdf in (rescoring_results_path / rescore_program).glob('*.sdf'):
                df = _read_sdf_values_and_names(sdf)[['ID', 'rfscore_v2']]
                # df.rename(columns={'minimizedAffinity': 'Vinardo'}, inplace=True)
                dfs.append(df)

        if 'rfscore_v3' == rescore_program:
            for sdf in (rescoring_results_path / rescore_program).glob('*.sdf'):
                df = _read_sdf_values_and_names(sdf)[['ID', 'rfscore_v3']]
                # df.rename(columns={'minimizedAffinity': 'Vinardo'}, inplace=True)
                dfs.append(df)

        if rescore_program in ['vina_hydrophobic', 'vina_intra_hydrophobic']:
            if os.path.exists(rescoring_results_path / 'rfscore_v3'):
                rescore_path = rescoring_results_path / 'rfscore_v3'
            else:
                rescore_path = rescoring_results_path / rescore_program

            for sdf in rescore_path.glob('*.sdf'):
                df = _read_sdf_values_and_names(sdf)

                if 'vina_hydrophobic' == rescore_program:
                    df = df[['ID', 'vina_hydrophobic']]

                elif 'vina_intra_hydrophobic' == rescore_program:
                    df = df[['ID', 'vina_intra_hydrophobic']]
                else:
                    df = df[['ID', 'vina_hydrophobic', 'vina_intra_hydrophobic']]
                # df.rename(columns={'minimizedAffinity': 'Vinardo'}, inplace=True)
                dfs.append(df)

        if 'rtmscore' == rescore_program:
            for csv_file in (
                    rescoring_results_path /
                    rescore_program).glob('*.csv'):
                df = pd.read_csv(csv_file)
                # split Pose ID column with - to remove last number
                df['id'] = df['id'].str.split('-').str[0]
                df.rename(columns={'id': 'ID', 'score': 'RTMScore'}, inplace=True)
                dfs.append(df)

        if 'chemplp' == rescore_program:
            for dir in (os.listdir(rescoring_results_path / rescore_program)):
                df = pd.read_csv(rescoring_results_path /
                                rescore_program /
                                dir /
                                'ranking.csv')[['LIGAND_ENTRY', 'TOTAL_SCORE']]
                # df = PandasTools.LoadSDF(str(sdf))[['ID', 'CHEMPLP']]
                df.rename(
                    columns={
                        'LIGAND_ENTRY': 'Pose ID',
                        'TOTAL_SCORE': 'CHEMPLP'},
                    inplace=True)
                df['Pose ID'] = df['Pose ID'].str.split('_').str[0:3].str.join('_')
                df.rename(columns={'Pose ID': 'ID'}, inplace=True)
                dfs.append(df)

        if 'scorch' == rescore_program:
            for csv_file in (
                    rescoring_results_path /
                    rescore_program).glob('*.csv'):
                df = pd.read_csv(csv_file)[['Ligand_ID', 'SCORCH_pose_score']]
                df['ID'] = PandasTools.LoadSDF(str(
                    rescoring_results_path.parent / 'sdf_split' / f"{df.loc[0, 'Ligand_ID']}.sdf"))['ID']
                df.rename(columns={'SCORCH_pose_score': 'SCORCH'}, inplace=True)
                dfs.append(df.drop(columns=['Ligand_ID']))

        if 'hyde' == rescore_program:
            for sdf in (rescoring_results_path / rescore_program).glob('*.sdf'):
                df = _read_sdf_values_and_names(
                    sdf)[['ID', 'BIOSOLVEIT.HYDE_ESTIMATED_AFFINITY_LOWER_BOUNDARY [nM]']]
                df.rename(
                    columns={
                        'BIOSOLVEIT.HYDE_ESTIMATED_AFFINITY_LOWER_BOUNDARY [nM]': 'HYDE'},
                    inplace=True)
                dfs.append(df)

        csv_file = pd.concat(dfs, ignore_index=True)
        csv_file.to_csv(
            rescoring_results_path /
            rescore_program /
            f'{rescore_program}_rescoring.csv',
            index=False)
        
    except Exception as e:
        print(f'❗❗Error in reading {rescore_program} results: {e}')
        print(f'{rescore_program} could be not excuted')




def _merge_rescoring_results(
        rescoring_results_path,
        rescoring_programs
):
    """
    This function is to merge the rescoring results. It takes the following arguments:
    Args:
        rescoring_results_path (str): The path to the rescoring results
        rescoring_programs (list): The list of rescoring programs to be used
    Returns:
        Merged rescoring results in csv file
    """
    all_rescoring_dfs = []
    for rescore_program in rescoring_programs:
        rescore_program = rescore_program.lower()
        if f'{rescore_program}_rescoring.csv' in os.listdir(
                rescoring_results_path / rescore_program):
            df = pd.read_csv(
                str(
                    rescoring_results_path /
                    rescore_program /
                    f'{rescore_program}_rescoring.csv')).drop_duplicates(
                subset="ID")
            all_rescoring_dfs.append(df)
        else:
            print(f'{rescore_program} is not excuted')
            return

    merged_df = all_rescoring_dfs[0]
    for df in all_rescoring_dfs[1:]:

        merged_df = pd.merge(merged_df, df, on='ID', how='outer')
    merged_df.drop_duplicates(subset='ID', inplace=True)
    merged_df.to_csv(
        rescoring_results_path.parent /
        'all_rescoring_results.csv',
        index=False)

    return merged_df

def _clean_rescoring_results(rescoring_programs, rescoring_results_path):
    """
    This function is to remove unncessary files and dirs. It takes the following arguments:
    Args:
        rescoring_programs (list): The list of rescoring programs to be used
        rescoring_results_path (str): The path to the rescoring results
    Returns:
        Cleaned rescoring results
    """
    for program in rescoring_programs:
        program = program.lower()
        for file in os.listdir(rescoring_results_path / program):
            if file != f'{program}_rescoring.csv':
                # if file then os.remove and if dir then shutil.rmtree
                if os.path.isdir(rescoring_results_path / program / file):
                    shutil.rmtree(rescoring_results_path / program / file)
                else:
                    os.remove(rescoring_results_path / program / file)