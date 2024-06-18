
import csv
import os
# from software.RTMScore.rtmscore_modified import *
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor

import pandas as pd
from rdkit.Chem import PandasTools

from preprocessing import plants_preprocessing
from utilities import (any_in_list, pdb_converter,
                       pocket_coordinates_generation, run_command, split_sdf)


def rescoring_function(
    rescore_programs,
    protein_file,
    docked_library_path,
    ref_file,
    ncpu
):
    """
    This function is the high-level function to deploy all scoring functions. It takes the following arguments:
    Args:
        rescore_programs (list): The list of rescoring programs to be used
        protein_file (str): The path to the protein file
        docked_library_path (str): The path to the docked library
        ref_file (str): The path to the reference ligand file
    Returns:
        Saved rescoring results in the rescoring_results folder as csv file, besides the individual rescoring results in the rescoring_results folder
    """
    rescoring_dict = {

        'smina_affinity': smina_rescoring,
        'ad4': ad4_rescoring,
        'linf9': linf9_rescoring,
        'vinardo': vinardo_rescoring,
        'chemplp': chemplp_rescoring,
        'hyde': hyde_rescoring,
        'vina_hydrophobic': vina_hydrophobic_rescoring,
        'vina_intra_hydrophobic': vina_intra_hydrophobic_rescoring,
        'rtmscore': rtmscore_rescoring,
        'rfscorevs_v1': rfscorevs_V1_rescoring,
        'rfscorevs_v2': rfscorevs_V2_rescoring,
        'rfscorevs_v3': rfscorevs_v3_rescoring,
        'cnnscore': gnina_score_rescoring,
        'cnnaffinity': gnina_affinity_rescoring,
        'scorch': scorch_rescoring,
    }
    # Create folder for rescoring results
    results_folder = docked_library_path.parent / 'rescoring_results'
    results_folder.mkdir(exist_ok=True)
    num_cpus = ncpu

    # convert protein and ligand to mol2 and pdbqt format in case of CHEMPLP
    # and SCORCH respectively
    pdb_converter(protein_file, rescore_programs)
    pdb_converter(ref_file, rescore_programs)

    for program in rescore_programs:

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
                            protein_file,
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
        read_rescoring_results(results_folder, program)

    merge_rescoring_results(results_folder, rescore_programs)


def ad4_rescoring(
        protein_file,
        docked_library_path,
        ref_file,
        output_path
) -> str:
    """
    This function for AD4 rescoring function and it takes the following arguments:
    Args:
        protein_file (str): The path to the protein file
        docked_library_path (str): The path to the docked library
        ref_file (str): The path to the reference ligand file
        output_path (str): The path to the output file
    Returns:
        The command to run the AD4 rescoring function
    """
    return (
        './software/gnina'
        f' --receptor {protein_file}'
        f' --ligand {str(docked_library_path)}'
        f' --out {str(output_path)}'
        f' --autobox_ligand {str(ref_file)}'
        ' --score_only'
        ' --scoring ad4_scoring --cnn_scoring none'
    )


def smina_rescoring(
        protein_file,
        docked_library_path,
        ref_file,
        output_path
) -> str:
    """
    This function for SMINA rescoring function and it takes the following arguments:
    Args:
        protein_file (str): The path to the protein file
        docked_library_path (str): The path to the docked library
        ref_file (str): The path to the reference ligand file
        output_path (str): The path to the output file
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
            f' --receptor {protein_file}'
            f' --ligand {str(docked_library_path)}'
            f' --out {str(output_path)}'
            f' --autobox_ligand {str(ref_file)}'
            ' --score_only'
            ' --cnn crossdock_default2018 --no_gpu'
        )


def gnina_score_rescoring(
        protein_file,
        docked_library_path,
        ref_file,
        output_path
) -> str:
    """
    This function for the score of GNINA rescoring function and it takes the following arguments:
    Args:
        protein_file (str): The path to the protein file
        docked_library_path (str): The path to the docked library
        ref_file (str): The path to the reference ligand file
        output_path (str): The path to the output file
    Returns:
        The command to run the the score of GNINA rescoring function
    """

    if any_in_list(['cnnaffinity', 'smina_affinity'],
                   os.listdir((output_path.parent).parent)):
        print(f'{output_path.name} is already excuted')
        return
    return (
        './software/gnina'
        f' --receptor {str(protein_file)}'
        f' --ligand {str(docked_library_path)}'
        f' --out {str(output_path)}'
        f' --autobox_ligand {str(ref_file)}'
        ' --score_only'
        ' --cnn crossdock_default2018 --no_gpu'
    )


def gnina_affinity_rescoring(
        protein_file,
        docked_library_path,
        ref_file,
        output_path
) -> str:
    """
    This function for GNINA rescoring function and it takes the following arguments:
    Args:
        protein_file (str): The path to the protein file
        docked_library_path (str): The path to the docked library
        ref_file (str): The path to the reference ligand file
        output_path (str): The path to the output file
    Returns:
        The command to run the GNINA rescoring function
    """
    if any_in_list(['cnnscore', 'smina_affinity'],
                   os.listdir((output_path.parent).parent)):
        print(f'{output_path.name} is already excuted')
        return
    return (
        './software/gnina'
        f' --receptor {str(protein_file)}'
        f' --ligand {str(docked_library_path)}'
        f' --out {str(output_path)}'
        f' --autobox_ligand {str(ref_file)}'
        ' --score_only'
        ' --cnn crossdock_default2018 --no_gpu'
    )


def vinardo_rescoring(
        protein_file,
        docked_library_path,
        ref_file,
        output_path
) -> str:
    """
    This function for Vinardo rescoring function and it takes the following arguments:
    Args:
        protein_file (str): The path to the protein file
        docked_library_path (str): The path to the docked library
        ref_file (str): The path to the reference ligand file
        output_path (str): The path to the output file
    Returns:
        The command to run the Vinardo rescoring function
    """
    return (
        './software/gnina'
        f' --receptor {protein_file}'
        f' --ligand {str(docked_library_path)}'
        f' --out {str(output_path)}'
        f' --autobox_ligand {str(ref_file)}'
        ' --score_only'
        ' --scoring vinardo --cnn_scoring none'
    )


def chemplp_rescoring(
        protein_file,
        docked_library_path,
        ref_file,
        output_path
) -> str:
    """
    This function for CHEMPLP rescoring function and it takes the following arguments:
    Args:
        protein_file (str): The path to the protein file
        docked_library_path (str): The path to the docked library
        ref_file (str): The path to the reference ligand file
        output_path (str): The path to the output file
    Returns:
        The command to run the CHEMPLP rescoring function
    """
    plants_search_speed = 'speed1'
    ants = '20'

    protein_mol2, mols_library_mol2, ref_ligand_mol2 = plants_preprocessing(
        protein_file, docked_library_path, ref_file)
    center_x, center_y, center_z, radius = pocket_coordinates_generation(
        protein_mol2, ref_ligand_mol2, pocket_coordinates_path='bindingsite.def')
    print(f"Center of the pocket is: {center_x}, {
          center_y}, {center_z} with radius of {radius}")

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
        f'protein_file {
            str(protein_mol2)}\n',
        f'ligand_file {
            str(mols_library_mol2)}\n',
        '# output\n',
        f'output_dir {
            str(
                output_path.parent / output_path.stem)}\n',
        '# write single mol2 files (e.g. for RMSD calculation)\n',
        'write_multi_mol2 1\n',
        '# binding site definition\n',
        f'bindingsite_center {
            str(center_x)} {
            str(center_y)} {
            str(center_z)}\n',
        f'bindingsite_radius {
            str(radius)}\n',
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
    chemplp_rescoring_config_path_config = docked_library_path.parent / \
        f'{output_path.stem}.config'

    with chemplp_rescoring_config_path_config.open('w') as configwriter:
        configwriter.writelines(chemplp_config)

    # Run PLANTS docking
    return f'./software/PLANTS --mode rescore {
        str(chemplp_rescoring_config_path_config)}'


def linf9_rescoring(
        protein_file,
        docked_library_path,
        ref_file,
        output_path
) -> str:
    """
    This function for LinF9 rescoring function and it takes the following arguments:
    Args:
        protein_file (str): The path to the protein file
        docked_library_path (str): The path to the docked library
        ref_file (str): The path to the reference ligand file
        output_path (str): The path to the output file
    Returns:
        The command to run the LinF9 rescoring function
    """
    return (
        f'./software/smina.static'
        f' --receptor {str(protein_file)}'
        f' --ligand {str(docked_library_path)}'
        f' --out {str(output_path)}'
        f' --autobox_ligand {str(ref_file)}'
        ' --scoring Lin_F9 --score_only'
    )


def rtmscore_rescoring(
        protein_file,
        docked_library_path,
        ref_file,
        output_path
):
    """
    This function for RTMScore rescoring function and it takes the following arguments:
    Args:
        protein_file (str): The path to the protein file
        docked_library_path (str): The path to the docked library
        ref_file (str): The path to the reference ligand file
        output_path (str): The path to the output file
    Returns:
        The command to run the RTMScore rescoring function
    """
    RTMScore_pocket = str(protein_file).replace('.pdb', '_pocket.pdb')
    number_of_ligand = docked_library_path.stem.split('_')[-1]
    ref_file = str(ref_file).replace('.pdb', '.sdf')
    if not os.path.exists(RTMScore_pocket):
        print('Pocket is not found, generating the pocket first and rescore')
        return (
            f'python software/RTMScore/rtmscore.py'
            f' -p {str(protein_file)}'
            f' -l {str(docked_library_path)}'
            f' -rl {str(ref_file)}'
            f' -o {str(output_path.parent / f"rtmscore_{number_of_ligand}")}'
            f' -gen_pocket'
            f' -c 10.0'
            ' -m software/RTMScore/trained_models/rtmscore_model1.pth'
        )
    return (
        f'python software/RTMScore/rtmscore.py'
        f' -p {str(RTMScore_pocket)}'
        f' -l {str(docked_library_path)}'
        f' -o {str(output_path.parent / f"rtmscore_{number_of_ligand}")}'
        ' -m software/RTMScore/trained_models/rtmscore_model1.pth'
    )


def scorch_rescoring(
        protein_file,
        docked_library_path,
        ref_file,
        output_path
):
    """
    This function for SCORCH rescoring function and it takes the following arguments:
    Args:
        protein_file (str): The path to the protein file
        docked_library_path (str): The path to the docked library
        ref_file (str): The path to the reference ligand file
        output_path (str): The path to the output file
    Returns:
        The command to run the SCORCH rescoring function
    """
    protein_file_pdqbt = str(protein_file).replace('.pdb', '.pdbqt')
    docked_library_file_pdqbt = str(
        docked_library_path).replace('.sdf', '.pdbqt')
    ref_ligand_pdqbt = str(ref_file).replace('.pdb', '.pdbqt')
    return (
        f'python software/SCORCH/scorch.py '
        f' --receptor {str(protein_file_pdqbt)} '
        f' --ligand {docked_library_file_pdqbt}'
        f" --ref_lig {str(ref_ligand_pdqbt)}"
        f' --out {str(output_path.parent / output_path.stem)}.csv'
        ' --return_pose_scores'
        ' --threads 1'
    )


def rfscorevs_V1_rescoring(
        protein_file,
        docked_library_path,
        ref_file,
        output_path
):
    """
    This function for RFScore_ver1 rescoring function and it takes the following arguments:
    Args:
        protein_file (str): The path to the protein file
        docked_library_path (str): The path to the docked library
        ref_file (str): The path to the reference ligand file
        output_path (str): The path to the output file
    Returns:
        The command to run the SCORCH rescoring function
    """
    return (
        f"oddt_cli {str(docked_library_path)}"
        f' --receptor {str(protein_file)}'
        f" --score rfscore_v1"
        f" -O {str(output_path)}"
        " -n 1"
    )


def hyde_rescoring(
        protein_file,
        docked_library_path,
        ref_file,
        output_path
):
    """
    This function for HYDE rescoring function and it takes the following arguments:
    Args:
        protein_file (str): The path to the protein file
        docked_library_path (str): The path to the docked library
        ref_file (str): The path to the reference ligand file
        output_path (str): The path to the output file
    Returns:
        The command to run the HYDE rescoring function
    """
    if ref_file.suffix == '.pdb':
        ref_file = ref_file.parent / f'{ref_file.stem}.sdf'
        if not os.path.exists(ref_file):
            print('convert pdb to sdf')
            pdb_to_sdf = f'obabel {
                str(ref_file.parent / ref_file.stem)}.pdb -O {str(ref_file)}'
            subprocess.run(pdb_to_sdf, shell=True)
    return (
        "software/hydescorer-2.0.0/hydescorer"
        f" -i {str(docked_library_path)}"
        f" -o {str(output_path)}"
        f" -p {str(protein_file)}"
        f" -r {str(ref_file)}"
        " --thread-count 1"
    )


def rfscorevs_V2_rescoring(
        protein_file,
        docked_library_path,
        ref_file,
        output_path
):
    """
    This function for RFscorevs_V2 rescoring rescoring function and it takes the following arguments:
    Args:
        protein_file (str): The path to the protein file
        docked_library_path (str): The path to the docked library
        ref_file (str): The path to the reference ligand file
        output_path (str): The path to the output file
    Returns:
        The command to run the RFscorevs_V2 rescoring rescoring function
    """
    return (
        f"oddt_cli {str(docked_library_path)}"
        f' --receptor {str(protein_file)}'
        f" --score rfscore_v2"
        f" -O {str(output_path)}"
        " -n 1"
    )


def rfscorevs_v3_rescoring(
        protein_file,
        docked_library_path,
        ref_file,
        output_path
):
    """
    This function for RFscorevs_V3 rescoring rescoring function and it takes the following arguments:
    Args:
        protein_file (str): The path to the protein file
        docked_library_path (str): The path to the docked library
        ref_file (str): The path to the reference ligand file
        output_path (str): The path to the output file
    Returns:
        The command to run the RFscorevs_V3 rescoring rescoring function
    """
    if any_in_list(['vina_hydrophobic', 'vina_intra_hydrophobic'],
                   os.listdir((output_path.parent).parent)):
        print(f'{docked_library_path.name} is already excuted')
        pass
    else:
        return (
            f"oddt_cli {str(docked_library_path)}"
            f' --receptor {str(protein_file)}'
            f" --score rfscore_v3"
            f" -O {str(output_path)}"
            " -n 1"
        )


def vina_hydrophobic_rescoring(
        protein_file,
        docked_library_path,
        ref_file,
        output_path
):
    """
    This function for Vina Hydrophobic rescoring function and it takes the following arguments:
    Args:
        protein_file (str): The path to the protein file
        docked_library_path (str): The path to the docked library
        ref_file (str): The path to the reference ligand file
        output_path (str): The path to the output file
    Returns:
        The command to run the Vina Hydrophobic rescoring function
    """
    # @TODO : check if the poses are already rescored through RF-Score-V3, if yes, copy the file to the output path
    if any_in_list(['rfscorevs_v3', 'vina_intra_hydrophobic'],
                   os.listdir((output_path.parent).parent)):
        print(f'{docked_library_path.name} is already excuted')
        return

    else:
        return (
            f"oddt_cli {str(docked_library_path)}"
            f' --receptor {str(protein_file)}'
            f" --score rfscore_v3"
            f" -O {str(output_path)}"
            " -n 1"
        )


def vina_intra_hydrophobic_rescoring(
        protein_file,
        docked_library_path,
        ref_file,
        output_path
):
    """
    This function for Vina Intra Hydrophobic rescoring function and it takes the following arguments:
    Args:
        protein_file (str): The path to the protein file
        docked_library_path (str): The path to the docked library
        ref_file (str): The path to the reference ligand file
        output_path (str): The path to the output file
    Returns:
        The command to run the Vina Intra Hydrophobic rescoring function
    """
    if any_in_list(['rfscorevs_v3', 'vina_hydrophobic'],
                   os.listdir((output_path.parent).parent)):
        print(f'{output_path.name} is already excuted')
        return
    else:
        return (
            f"oddt_cli {str(docked_library_path)}"
            f' --receptor {str(protein_file)}'
            f" --score rfscore_v3"
            f" -O {str(output_path)}"
            " -n 1"
        )


def read_sdf_values_and_names(sdf_file_path):
    with open(sdf_file_path, 'r') as file:
        all_records = []
        capture_data = False
        current_values = {}
        molecule_name = ""
        for line in file:
            line = line.strip()
            if line == "$$$$":
                # End of the current molecule record, reset for the next
                # molecule
                all_records.append((molecule_name, current_values))
                capture_data = False
                current_values = {}
                molecule_name = ""
            elif capture_data:
                if line.startswith("> <"):
                    # Assuming a simple key-value pair structure for data items
                    parts = line.split("> <")
                    if len(parts) == 2:
                        key = parts[1].split(">")[0]
                        # Read the next line for the value
                        value = next(file).strip()
                        current_values[key] = value

                if line.startswith(">  <"):
                    # Assuming a simple key-value pair structure for data items
                    parts = line.split(">  <")
                    if len(parts) == 2:
                        key = parts[1].split(">")[0]
                        # Read the next line for the value
                        value = next(file).strip()
                        current_values[key] = value

            elif line == "M  END":
                # Start capturing the data items after this line
                capture_data = True
            else:
                if not molecule_name:  # Capture the first non-empty line as the molecule name
                    molecule_name = line
        df = pd.DataFrame.from_records(
            all_records, columns=[
                'ID', 'Properties'])
        df = pd.concat([df.drop(['Properties'], axis=1),
                       df['Properties'].apply(pd.Series)], axis=1)
    return df


def read_rescoring_results(
        rescoring_results_path,
        rescore_program
):
    dfs = []
    print('\n\nReading rescoring results ⌛ ...\n\n')

    if f'{rescore_program}_rescoring.csv' in os.listdir(
            rescoring_results_path):
        print(f'{rescore_program} is already read')
        return

    # if 'cnnscore' == rescore_program:
    #     for sdf in (rescoring_results_path / rescore_program).glob('*.sdf'):
    #         df = PandasTools.LoadSDF(str(sdf))[['ID', 'CNNscore']]
    #         dfs.append(df)

    # if 'cnnaffinity' == rescore_program:
    #     for sdf in (rescoring_results_path / rescore_program).glob('*.sdf'):
    #         df = PandasTools.LoadSDF(str(sdf))[['ID', 'CNNaffinity']]
    #         dfs.append(df)

    # if 'smina_affinity' == rescore_program:
    #     for sdf in (rescoring_results_path / rescore_program).glob('*.sdf'):
    #         df = PandasTools.LoadSDF(str(sdf))[['ID', 'minimizedAffinity']]
    #         df.rename(columns={'minimizedAffinity': 'smina_affinity'}, inplace=True)
    #         dfs.append(df)

    if rescore_program in ['cnnscore', 'cnnaffinity', 'smina_affinity']:
        for path in ['cnnscore', 'cnnaffinity', 'smina_affinity']:
            rescore_path = rescoring_results_path / path
            if os.path.exists(rescore_path) and len(
                    os.listdir(rescore_path)) > 1:
                break

        for sdf in rescore_path.glob('*.sdf'):

            df = read_sdf_values_and_names(sdf)
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
            df = read_sdf_values_and_names(sdf)[['ID', 'minimizedAffinity']]
            df.rename(columns={'minimizedAffinity': 'ad4'}, inplace=True)
            dfs.append(df)

    if 'linf9' == rescore_program:
        for sdf in (rescoring_results_path / rescore_program).glob('*.sdf'):
            df = read_sdf_values_and_names(sdf)[['ID', 'minimizedAffinity']]
            df.rename(columns={'minimizedAffinity': 'LinF9'}, inplace=True)
            dfs.append(df)

    if 'vinardo' == rescore_program:
        for sdf in (rescoring_results_path / rescore_program).glob('*.sdf'):
            df = read_sdf_values_and_names(sdf)[['ID', 'minimizedAffinity']]
            df.rename(columns={'minimizedAffinity': 'Vinardo'}, inplace=True)
            dfs.append(df)

    if 'rfscorevs_v1' == rescore_program:
        for sdf in (rescoring_results_path / rescore_program).glob('*.sdf'):
            df = read_sdf_values_and_names(sdf)[['ID', 'rfscore_v1']]
            # df.rename(columns={'minimizedAffinity': 'Vinardo'}, inplace=True)
            dfs.append(df)

    if 'rfscorevs_v2' == rescore_program:
        for sdf in (rescoring_results_path / rescore_program).glob('*.sdf'):
            df = read_sdf_values_and_names(sdf)[['ID', 'rfscore_v2']]
            # df.rename(columns={'minimizedAffinity': 'Vinardo'}, inplace=True)
            dfs.append(df)

    if 'rfscorevs_v3' == rescore_program:
        for sdf in (rescoring_results_path / rescore_program).glob('*.sdf'):
            df = read_sdf_values_and_names(sdf)[['ID', 'rfscore_v3']]
            # df.rename(columns={'minimizedAffinity': 'Vinardo'}, inplace=True)
            dfs.append(df)

    if rescore_program in ['vina_hydrophobic', 'vina_intra_hydrophobic']:
        if os.path.exists(rescoring_results_path / 'rfscorevs_v3'):
            rescore_path = rescoring_results_path / 'rfscorevs_v3'
        else:
            rescore_path = rescoring_results_path / rescore_program

        for sdf in rescore_path.glob('*.sdf'):
            df = read_sdf_values_and_names(sdf)

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
            df = read_sdf_values_and_names(
                sdf)[['ID', 'BIOSOLVEIT.HYDE_ESTIMATED_AFFINITY_LOWER_BOUNDARY [nM]']]
            df.rename(
                columns={
                    'BIOSOLVEIT.HYDE_ESTIMATED_AFFINITY_LOWER_BOUNDARY [nM]': 'HYDE'},
                inplace=True)
            dfs.append(df)

    print(rescore_program, (rescoring_results_path / rescore_program))
    csv_file = pd.concat(dfs, ignore_index=True)
    csv_file.to_csv(
        rescoring_results_path /
        rescore_program /
        f'{rescore_program}_rescoring.csv',
        index=False)
    # return merged_df


def merge_rescoring_results(
        rescoring_results_path,
        rescore_programs
):
    all_rescoring_dfs = []
    for rescore_program in rescore_programs:
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
