import os
import shutil
import time
from pathlib import Path

import pandas as pd
from rdkit import Chem
from rdkit.Chem import PandasTools
from tqdm.auto import tqdm

from preprocessing import plants_preprocessing
from utilities import (extract_binding_pocket, pocket_coordinates_generation,
                       read_posebusters_data, run_command)


def docking(
    docking_methods,
    protein_file,
    current_library,
    ref_file,
    exhaustiveness,
    n_poses,
    OUTPUT,
    id_column='ID',
    time_calc=False,
    local_diffdock=False
):

    docking_dict = {
        'gnina': gnina_docking,
        'smina': smina_docking,
        'plants': plants_docking,
        'diffdock': diffdock_docking,
        'flexx': flexx_docking
    }

    OUTPUT.mkdir(parents=True, exist_ok=True)

    output_files = []
    allposes_df = pd.DataFrame()
    # Docking with different methods
    for docking_method in docking_methods:
        print(f"\n\nDocking with {docking_method.upper()} is running ...\n")
        docking_folder = OUTPUT / docking_method.lower()
        docking_folder.mkdir(parents=True, exist_ok=True)
        output_file = docking_folder / f'{docking_method.lower()}_poses.sdf'
        output_files.append(output_file)
        if output_file in os.listdir(docking_folder):
            print(
                f"{output_file} is already docked with {
                    docking_method.upper()}")
            continue

        docking_dict[docking_method.lower()](
            protein_file=protein_file,
            sdf_output=output_file,
            current_library=current_library,
            ref_file=ref_file,
            exhaustiveness=exhaustiveness,
            n_poses=n_poses,
            local_diffdock=local_diffdock
        )

    # Concatenate all poses in one SDF file
    print("\n\nConcatenating all poses in one SDF file ...\n")

    if 'allposes.sdf' in os.listdir(OUTPUT):
        print(
            f"Compounds are already docked and concatenated, CHECK {
                OUTPUT / 'allposes.sdf'}")
        return
    try:
        for output_file in output_files:
            if output_file.name not in os.listdir(output_file.parent):
                print(
                    f"{output_file} is not docked with any of the docking methods")
            else:
                output_df = PandasTools.LoadSDF(
                    str(output_file),
                    idName='ID',
                    molColName='Molecule',
                )
                allposes_df = pd.concat([allposes_df, output_df])

        PandasTools.WriteSDF(
            allposes_df,
            str(OUTPUT / 'allposes.sdf'),
            molColName='Molecule',
            idName='ID',
        )

    except Exception as e:
        print(
            f"ERROR:{e}\n Failed to concatenate all poses in one SDF file!\n")


def gnina_docking(
        protein_file,
        sdf_output,
        current_library,
        ref_file,
        exhaustiveness,
        n_poses,
        local_diffdock
):

    gnina_cmd = (
        f'./software/gnina -r {protein_file}'
        f' -l {current_library}'
        f' -o {sdf_output}'
        f' --autobox_ligand {str(ref_file)}'
        f' --seed 1637317264'
        f' --exhaustiveness {exhaustiveness}'
        f' --num_modes {str(n_poses)}'
        ' --cnn_scoring rescore'
        ' --cnn crossdock_default2018'
        ' --no_gpu'
    )
    start_time = time.time()
    if sdf_output.name not in os.listdir(sdf_output.parent):
        run_command(gnina_cmd)
        end_time = time.time()    # End time

        duration = end_time - start_time
        print(f"\n\nThe GNINA took {duration} seconds to run.")
    else:
        print(f"Compounds are already docked with GNINA v 1.0")

    df = PandasTools.LoadSDF(str(sdf_output))

    new_df = pd.DataFrame()
    if df is None:
        print("Invalid generated poses.")
        return None
    if df['ID'].str.split('_').str.len().max() <= 2:
        print('ID format of GNINA is incorrect, fixing it ...')

        for name, group in df.groupby('ID'):
            # Add number to ID for each row in the group
            group['newID'] = [
                f"{name}_gnina_{i}" for i in range(
                    1, len(group) + 1)]
            new_df = pd.concat([new_df, group])

        new_df.drop('ID', axis=1, inplace=True)
        new_df.rename(columns={'newID': 'ID'}, inplace=True)

        PandasTools.WriteSDF(
            new_df,
            str(sdf_output),
            idName='ID',
            molColName='ROMol',
            properties=list(
                new_df.columns),
            allNumeric=False)


def smina_docking(
        protein_file,
        sdf_output,
        current_library,
        ref_file,
        exhaustiveness,
        n_poses,
        local_diffdock
):

    smina_cmd = (
        f'./software/gnina -r {protein_file}'
        f' -l {current_library} -o {sdf_output}'
        f' --autobox_ligand {str(ref_file)}'
        ' --autobox_extend=1 --seed 1637317264'
        f' --exhaustiveness {exhaustiveness} --num_modes {str(n_poses)} --cnn_scoring=none'
    )
    if sdf_output.name not in os.listdir(sdf_output.parent):
        start_time = time.time()
        run_command(smina_cmd)
        end_time = time.time()    # End time

        duration = end_time - start_time
        print(f"\n\nThe SMINA took {duration} seconds to run.")
    else:
        print(f"Compounds are already docked with SMINA")

    # Rename ID column
    df = PandasTools.LoadSDF(str(sdf_output))
    print(df)
    if df is None:
        print("Invalid generated poses.")
        return None
    new_df = pd.DataFrame()

    # Iterate over each group
    if df['ID'].str.split('_').str.len().max() <= 2:
        print('ID format of SMINA is incorrect, fixing it ...')

        for name, group in df.groupby('ID'):
            # Add number to ID for each row in the group
            group['newID'] = [
                f"{name}_smina_{i}" for i in range(
                    1, len(group) + 1)]
            new_df = pd.concat([new_df, group])

        new_df.drop('ID', axis=1, inplace=True)
        new_df.rename(columns={'newID': 'ID'}, inplace=True)

        PandasTools.WriteSDF(
            new_df,
            str(sdf_output),
            idName='ID',
            molColName='ROMol',
            properties=list(
                new_df.columns),
            allNumeric=False)


def plants_docking(
        protein_file,
        sdf_output,
        current_library,
        ref_file,
        exhaustiveness,
        n_poses,
        local_diffdock
):
    '''
    Perform docking using the PLANTS software on a protein and a reference ligand, and return the path to the results.

    Args:
    protein_file (str): path to the protein file in PDB format
    ref_file (str): path to the reference ligand file in SDF format
    software (str): path to the software folder
    exhaustiveness (int): level of exhaustiveness for the docking search, ranges from 0-8
    n_poses (int): number of poses to be generated

    Returns:
    results_path (str): the path to the results file in SDF format
    '''
    # convert to structure, ligands, reference ligand to mol2
    print(current_library)
    protein_mol2, mols_library_mol2, ref_ligand_mol2 = plants_preprocessing(
        protein_file, 
        current_library, 
        ref_file
        )
    # get pocket coordinates
    center_x, center_y, center_z, radius = pocket_coordinates_generation(
        protein_mol2, 
        ref_ligand_mol2, 
        pocket_coordinates_path='bindingsite.def'
        )
    # print(f"Center of the pocket is: {center_x}, {center_y}, {center_z} with radius of {radius}")
    # Generate plants config file
    plants_docking_config_path = sdf_output.parent / 'config.config'
    plants_config = [
        '# search algorithm\n',
        'search_speed speed1\n',
        'aco_ants 20\n',
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
        f'output_dir ' + str(
            sdf_output.parent / 'temp') + '\n',
        '# write single mol2 files (e.g. for RMSD calculation)\n',
        'write_multi_mol2 1\n',
        '# binding site definition\n',
        f'bindingsite_center {str(center_x)} {str(center_y)} {str(center_z)}\n',
        f'bindingsite_radius {str(radius)}\n',
        '# cluster algorithm\n',
        'cluster_structures ' + str(n_poses) + '\n',
        'cluster_rmsd 2.0\n',
        '# write\n',
        'write_ranking_links 0\n',
        'write_protein_bindingsite 0\n',
        'write_protein_conformations 0\n',
        'write_protein_splitted 0\n',
        'write_merged_protein 0\n',
        '####\n']

    with plants_docking_config_path.open('w') as configwriter:
        configwriter.writelines(plants_config)
    # Run PLANTS docking

    plants_docking_command = f'software/PLANTS --mode screen {str(plants_docking_config_path)}'

    if sdf_output.name not in os.listdir(sdf_output.parent):
        start_time = time.time()
        run_command(plants_docking_command)
        end_time = time.time()    # End time

        duration = end_time - start_time
        print(f"\n\nThe PLANTS took {duration} seconds to run.")
    else:
        print(f"Compounds are already docked with PLANTS")
        return

    plants_docking_results_mol2 = sdf_output.parent / 'temp' / 'docked_ligands.mol2'
    plants_docking_results_sdf = plants_docking_results_mol2.with_suffix(
        '.sdf')
    # Convert PLANTS poses to sdf

    obabel_command = 'obabel -imol2 ' + \
        str(plants_docking_results_mol2) + ' -O ' + str(plants_docking_results_sdf)
    run_command(obabel_command)

    # plants_scoring_results = sdf_output.parent/ 'temp' / 'ranking.csv'
    # Fetch PLANTS poses

    plants_poses = PandasTools.LoadSDF(
        str(plants_docking_results_sdf),
        idName='ID',
        molColName='Molecule',
        includeFingerprints=False,
        embedProps=False,
        removeHs=False,
        strictParsing=True)
    if plants_poses is None:
        print("Invalid generated poses.")
        return None
    plants_poses['ID'] = plants_poses['ID'].str.split(
        '_').str[0] + '_plants_' + plants_poses['ID'].str.split('_').str[4]

    PandasTools.WriteSDF(plants_poses,
                         str(sdf_output),
                         molColName='Molecule',
                         idName='ID',
                         properties=list(plants_poses.columns))

    shutil.rmtree(sdf_output.parent / 'temp', ignore_errors=True)
    os.remove(str(sdf_output.parent / 'config.config'))


def diffdock_docking(
        protein_file,
        sdf_output,
        current_library,
        ref_file,
        exhaustiveness,
        n_poses,
        local_diffdock=False
):
    library_df = PandasTools.LoadSDF(str(current_library))
    molecule_id = library_df['ID'].tolist()
    ligands = [Chem.MolToSmiles(mol) for mol in library_df['ROMol'].tolist()]

    pocket_res_indices = f'{protein_file.stem}_pocket_residues'
    if pocket_res_indices not in os.listdir(
            protein_file.parent) and local_diffdock:
        extract_binding_pocket(
            protein_file,
            ref_file,
            protein_file.parent /
            pocket_res_indices)
    else:
        print(f"Binding pocket is already extracted")

    for id, smiles in tqdm(zip(molecule_id, ligands),
                           total=len(molecule_id),
                           desc='Local DiffDock is running ...'):
        diffdock_cmd = (
            f"python -m inference"
            f" --protein_path {str(protein_file)}"
            f" --ligand '{smiles}'"
            f" --complex_name results/local_diffdock/{id}"
            f" --out_dir {str(protein_file.parent)}"
            f" --inference_steps 20"
            f" --samples_per_complex {n_poses}"
            # f" --save_visualisation"
            f" --batch_size 8"
            f" --actual_steps 18"
            f" --no_final_step_noise"
        )
        if os.path.exists(sdf_output) or os.path.exists(
                sdf_output.parent / 'local_diffdock_poses.sdf'):
            print('Poses are already generated using DiffDock/Local DiffDock')
            break
        if local_diffdock:
            diffdock_cmd += f" --binding_site_residues {
                str(protein_file.parent / pocket_res_indices)}"

        if sdf_output.name in os.listdir(sdf_output.parent):
            print(f"Compounds are already docked with DiffDock")
            continue
        if id not in os.listdir(sdf_output.parent):
            os.chdir(os.getcwd() + '/software/DiffDock')
            start_time = time.time()
            run_command(diffdock_cmd)
            end_time = time.time()    # End time

            duration = end_time - start_time
            print(f"\n\nThe diffdock took {duration} seconds to run.")
            os.chdir(os.path.join(os.getcwd(), '..', '..'))
        else:
            print(f"Compound {id} is already docked with DiffDock")

    reading_diffdock_poses(sdf_output, n_poses, local_diffdock)
    # concatenate all poses in one dataframe


def reading_diffdock_poses(sdf_output, n_poses, local_diffdock):
    
    diffdock_type = 'diffdock'
    if local_diffdock:
        diffdock_type = 'localdiffdock'
    
    list_molecules = os.listdir(str(sdf_output.parent))
    df = pd.DataFrame(columns=['ID', 'molecules', 'confidence_score'])
    molecules = []
    ids = []
    confidence_scores = []
    for mol in list_molecules:
        try:
            for i in range(1, (n_poses) + 1):
                pose = [f for f in os.listdir(
                    str(sdf_output.parent / mol)) if f.startswith(f"rank{i}_")]
                if not pose:
                    print(f"No poses found for {mol}")
                    continue
                else:
                    pose = pose[0]
                supplier = Chem.SDMolSupplier(
                    str(sdf_output.parent / mol / pose), sanitize=False, removeHs=False)
                molecule = [mol for mol in supplier if mol is not None][0]
                if molecule:
                    ids.append(f"{mol}_{diffdock_type}_{i}")
                    molecules.append(molecule)
                    if 100 > i >= 10:
                        confidence_scores.append(pose[17:-4])
                    else:
                        confidence_scores.append(pose[16:-4])

        except Exception as e:
            print(f"ERROR: {e}\n")

    df['ID'] = ids
    df['molecules'] = molecules
    df['confidence_score'] = confidence_scores

    if sdf_output.name not in os.listdir(sdf_output.parent) and len(df) > 0:
        PandasTools.WriteSDF(
            df,
            str(sdf_output.parent / 'diffdock_poses.sdf'),
            idName='ID',
            molColName='molecules',
            properties=list(df.columns)
        )
        _ = [shutil.rmtree(sdf_output.parent / mol, ignore_errors=True) for mol in list_molecules]
    else:
        print(
            f"Compounds are already docked and concatenated, CHECK {sdf_output}")


def flexx_docking(
        protein_file,
        sdf_output,
        current_library,
        ref_file,
        exhaustiveness,
        n_poses,
        local_diffdock
):
    """
    The function docks the ligands with FlexX and returns the poses in SDF format
    Args:
        protein_file: protein file in PDB format
        sdf_output: path to the output file in SDF format
        current_library: library of ligands in SDF format
        ref_file: reference ligand in SDF format
        exhaustiveness: level of exhaustiveness for the docking search, ranges from 0-8
        n_poses: number of poses to be generated
    Returns:
        flexx_command: command to run FlexX
    """
    ref_file_sdf = ref_file.with_suffix('.sdf')
    if ref_file.suffix == '.pdb' and ref_file_sdf.name not in os.listdir(
            ref_file.parent):
        obabel_cmd = f"obabel -ipdb {str(ref_file)
                                     } -osdf -O {str(ref_file_sdf)}"
        run_command(obabel_cmd)

    else:
        print(f"Reference ligand is already in SDF format")

    flexx_cmd = (
        f"./software/flexx-6.0.0/flexx"
        f" --thread-count 8"
        f" -p {str(protein_file)}"
        f" --r {str(ref_file_sdf)}"
        f" -i {str(current_library)}"
        f" --max-nof-conf {str(n_poses)}"
        f" -o {str(sdf_output)}"

    )
    if sdf_output.name not in os.listdir(sdf_output.parent):
        try:
            start_time = time.time()
            run_command(flexx_cmd)
            end_time = time.time()    # End time

            duration = end_time - start_time
            print(f"\n\nThe flexx took {duration} seconds to run.")
        except Exception as e:
            print(f"ERROR: {e}")
            print("Please check if FlexX is installed and the path is correct and the license in the same directory of the software")
    else:
        print(f"Compounds are already docked with FlexX v 6.0")

    # Rename ID column
    df = PandasTools.LoadSDF(str(sdf_output))
    if df is None:
        print("Invalid generated poses.")
        return None
    if not (df['ID'].str.split('_').str[1] == 'flexx').all():

        print('ID format is incorrect, fixing it ...')
        df['ID'] = df['ID'].str.split('_').str[0] + '_flexx_' + df['ID'].str.split('_').str[-1]
        PandasTools.WriteSDF(
            df[['ID', 'ROMol']],
            str(sdf_output),
            idName='ID',
            molColName='ROMol',
        )
    else:
        print('ID format is correct')


def poses_checker(poses_path, protein_path, output_file):
    """
    The function checks if the poses are already scored with PoseBusters, if not, it runs the PoseBusters
    to check the quality of generated poses and returns the filtered dataframe

    Args:
        poses_path: path to the poses file in SDF format
        protein_path: path to the protein file in PDB format
        output_file: path to the output file in CSV format
    Returns:
        filtered_df: filtered dataframe with the poses and their scores
    """
    if os.path.exists(output_file):
        print('PoseBusters was executed')
    else:
        print('PoseBusters is running ...')
        posebusters_cmd = (
            f"bust {str(poses_path)} -p {str(protein_path)} --outfmt csv >> {output_file}"
        )
        run_command(posebusters_cmd)
    df = pd.read_csv(output_file)
    df.drop_duplicates(subset=['molecule'], inplace=True)
    filtered_df = read_posebusters_data(df)
    return filtered_df
