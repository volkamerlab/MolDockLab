import os
import shutil
import subprocess
import pandas as pd
from rdkit import Chem
from pathlib import Path
from tqdm.auto import tqdm
from rdkit.Chem import PandasTools
from utilities import get_ligand_coordinates_centers, extract_binding_pocket

def docking(docking_methods, 
            protein_file,
            current_library,
            ref_file,
            exhaustiveness,
            n_poses):
    
    docking_dict = { 
        'gnina': gnina_docking, 
        'smina': smina_docking, 
        'plants': plants_docking,
        'local_diffdock': local_diffdock_docking,
        'flexx' : flexx_docking
        }
    (protein_file.parent / 'results').mkdir(parents=True, exist_ok=True)

    # Get ligand coordinates
    print("Extracting ligand coordinates supports either SDF files or PDB files...")
    center_x, center_z, center_y = get_ligand_coordinates_centers(str(ref_file))
    
    output_files = []
    allposes_df = pd.DataFrame()

    # Docking with different methods
    for docking_method in docking_methods:    
        docking_folder = Path(protein_file).parent / 'results' / docking_method.lower()
        docking_folder.mkdir(parents=True, exist_ok=True)
        output_file = docking_folder / f'{docking_method.lower()}_poses.sdf'
        output_files.append(output_file)
        if output_file in os.listdir(docking_folder):
            print(f"{output_file} is already docked with {docking_method.upper()}")
            continue

        docking_dict[docking_method.lower()](
            protein_file,
            output_file,
            current_library,
            ref_file, center_x, center_z, center_y,
            exhaustiveness,
            n_poses
            )
        
    #Concatenate all poses in one SDF file    
    try:
        for output_file in output_files:
            if output_file.name not in os.listdir(output_file.parent):
                print(f"{output_file} is not docked with any of the docking methods")
            else:
                output_df = PandasTools.LoadSDF(
                    str(output_file),
                    idName='ID',
                    molColName='Molecule',
                    )
                allposes_df = pd.concat([allposes_df, output_df])

        PandasTools.WriteSDF(
                allposes_df,
                str(protein_file.parent / 'results' / 'allposes.sdf'),
                molColName='Molecule',
                idName='ID',
                )
        
    except Exception as e:
        print(f"ERROR:{e}\n Failed to concatenate all poses in one SDF file!\n")
        print('Please check column names in all poses to have the same columns (''ID'', ''Molecule'')')
        # display(allposes_df[['ID', 'Molecule']])        

def gnina_docking(
        protein_file,
        sdf_output,
        current_library,
        ref_file, center_x, center_z, center_y,
        exhaustiveness,
        n_poses
        ):

    gnina_cmd = (
    f'./software/gnina -r {protein_file}' 
    f' -l {current_library} -o {sdf_output} ' 
    f" --center_x {center_x}" 
    f" --center_y {center_y}" 
    f" --center_z {center_z}" 
    f" --size_x 10" 
    f" --size_y 10" 
    f" --size_z 10" 
    f' --seed 1637317264 --exhaustiveness {exhaustiveness}' 
    f' --num_modes {str(n_poses)} --cnn_scoring rescore' 
    ' --cnn crossdock_default2018 --no_gpu --log data/log.txt'
    )
    if sdf_output.name not in os.listdir(sdf_output.parent):
        subprocess.call(
            gnina_cmd, 
            shell=True,
            # stdout=DEVNULL,
            # stderr=STDOUT
            )
    else:
        print(f"Compounds are already docked with GNINA v 1.0")
        return
    # Rename ID column
    df = PandasTools.LoadSDF(str(sdf_output))[['ID', 'ROMol']]

    new_df = pd.DataFrame()

    # Iterate over each group

    if df['ID'].str.split('_').str.len().max() <= 2:
        print('ID format of GNINA is incorrect, fixing it ...')

        for name, group in df.groupby('ID'):
            # Add number to ID for each row in the group
            group['newID'] = [f"{name}_gnina_{i}" for i in range(1, len(group) + 1)]
            new_df = pd.concat([new_df, group])

        new_df.drop('ID', axis=1, inplace=True)
        new_df.rename(columns={'newID': 'ID'}, inplace=True)

        PandasTools.WriteSDF(new_df, str(sdf_output), idName='ID', molColName='ROMol', properties=list(new_df.columns), allNumeric=False)

def smina_docking(
        protein_file,
        sdf_output,
        current_library,
        ref_file, center_x, center_z, center_y,
        exhaustiveness,
        n_poses
        ):

    smina_cmd =( 
    f'./software/gnina -r {protein_file}' 
    f' -l {current_library} -o {sdf_output}' 
    f" --center_x {center_x}" 
    f" --center_y {center_y}" 
    f" --center_z {center_z}" 
    f" --size_x 5" 
    f" --size_y 5" 
    f" --size_z 5" 
    ' --autobox_extend=1 --seed 1637317264' 
    f' --exhaustiveness {exhaustiveness} --num_modes {str(n_poses)} --cnn_scoring=none'
    )
    if sdf_output.name not in os.listdir(sdf_output.parent):
        subprocess.call(
            smina_cmd, 
            shell=True,
            # stdout=DEVNULL,
            # stderr=STDOUT
            )
    else:
        print(f"Compounds are already docked with SMINA")
        return
    # Rename ID column
    df = PandasTools.LoadSDF(str(sdf_output))[['ID', 'ROMol']]

    new_df = pd.DataFrame()

    # Iterate over each group
    if df['ID'].str.split('_').str.len().max() <= 2:
        print('ID format of SMINA is incorrect, fixing it ...')

        for name, group in df.groupby('ID'):
            # Add number to ID for each row in the group
            group['newID'] = [f"{name}_gnina_{i}" for i in range(1, len(group) + 1)]
            new_df = pd.concat([new_df, group])

        new_df.drop('ID', axis=1, inplace=True)
        new_df.rename(columns={'newID': 'ID'}, inplace=True)

        PandasTools.WriteSDF(new_df, str(sdf_output), idName='ID', molColName='ROMol', properties=list(new_df.columns), allNumeric=False)

def plants_docking(
        protein_file,
        sdf_output,
        current_library,
        ref_file, center_x, center_z, center_y,
        exhaustiveness,
        n_poses
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


    # Convert protein file to .mol2 using open babel
    plants_protein_mol2 = protein_file.with_suffix(".mol2")
    

    if plants_protein_mol2.name not in os.listdir(plants_protein_mol2.parent):
        
        obabel_command = (
        f'obabel -ipdb {str(protein_file)}' 
        f' -O {str(plants_protein_mol2)}'
        )

        subprocess.call(
            obabel_command,
            shell=True,
            # stdout=DEVNULL,
            # stderr=STDOUT
            )
    else:
        print(f"Protein is already converted to mol2 format")
        
    # Convert prepared ligand file to .mol2 using open babel
    plants_library_mol2 = current_library.with_suffix(".mol2")

    if plants_library_mol2.name not in os.listdir(plants_library_mol2.parent):

        obabel_command = (
            f'obabel -isdf {str(current_library)}' 
            f' -O {str(plants_library_mol2)}'
        )
        subprocess.call(
            obabel_command,
            shell=True,
            # stdout=DEVNULL,
            # stderr=STDOUT
            )
    else:
        print(f"Library is already converted to mol2 format")

    
    # Generate plants config file
    plants_docking_config_path = sdf_output.parent / 'config.config'
    plants_config = ['# search algorithm\n',
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
                     f'protein_file {str(plants_protein_mol2)} \n',
                     f'ligand_file {str(plants_library_mol2)} \n',

                    '# output\n',
                    f'output_dir ' + str(sdf_output.parent / 'temp') + '\n',

                     '# write single mol2 files (e.g. for RMSD calculation)\n',
                     'write_multi_mol2 1\n',

                     '# binding site definition\n',
                     f'bindingsite_center {center_x} {center_y} {center_z} \n',
                     'bindingsite_radius 5 \n',

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
        subprocess.call(
            plants_docking_command,
            shell=True,
            # stdout=DEVNULL,
            # stderr=STDOUT
            )
    else:
        print(f"Compounds are already docked with PLANTS")
        return

    plants_docking_results_mol2 = sdf_output.parent/ 'temp' / 'docked_ligands.mol2'
    plants_docking_results_sdf = plants_docking_results_mol2.with_suffix('.sdf')
    # Convert PLANTS poses to sdf

    obabel_command = 'obabel -imol2 ' + \
        str(plants_docking_results_mol2) + ' -O ' + str(plants_docking_results_sdf)
    subprocess.call(
        obabel_command,
        shell=True,
        # stdout=DEVNULL,
        # stderr=STDOUT
        )

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
    # plants_scores = pd.read_csv(
    #     str(plants_scoring_results), usecols=[
    #         'LIGAND_ENTRY', 'TOTAL_SCORE'])
    # plants_scores = plants_scores.rename(
    #     columns={'LIGAND_ENTRY': 'ID', 'TOTAL_SCORE': 'CHEMPLP'})
    # plants_scores = plants_scores[['ID', 'CHEMPLP']]
    # plants_df = pd.merge(plants_scores, plants_poses, on='ID')

    plants_poses['ID'] = plants_poses['ID'].str.split('_').str[0] + '_plants_' + plants_poses['ID'].str.split('_').str[4]


    PandasTools.WriteSDF(plants_poses,
                            str(sdf_output),
                            molColName='Molecule',
                            idName='ID',
                            properties=list(plants_poses.columns))
    
    shutil.rmtree(sdf_output.parent / 'temp', ignore_errors=True)
    os.remove(str(sdf_output.parent / 'config.config'))


def local_diffdock_docking(
        protein_file,
        sdf_output,
        current_library,
        ref_file, center_x, center_z, center_y,
        exhaustiveness,
        n_poses
):
    library_df = PandasTools.LoadSDF(str(current_library))
    molecule_id = library_df['ID'].tolist()
    ligands = [Chem.MolToSmiles(mol) for mol in library_df['ROMol'].tolist()]
    # molecule_id = molecule_id[2:3]
    # ligands = ligands[2:3]

    pocket_res_indices = f'{protein_file.stem}_pocket_residues'
    if pocket_res_indices not in os.listdir(protein_file.parent):
        extract_binding_pocket (protein_file, ref_file, protein_file.parent / pocket_res_indices)
    else:
        print(f"Binding pocket is already extracted")
    os.chdir(str((protein_file.parent).parent / 'software' / 'DiffDock'))



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
            f" --binding_site_residues {str(protein_file.parent / pocket_res_indices)}"
            f" --no_final_step_noise"
        )

        if id not in os.listdir(sdf_output.parent) and sdf_output.name not in os.listdir(sdf_output.parent):

            subprocess.call(
                diffdock_cmd,
                shell=True,
                # stdout=DEVNULL,
                # stderr=STDOUT
                )
        else:
            print(f"Compound {id} is already docked with Local DiffDock")
    os.chdir(str((protein_file.parent).parent))

    # concatenate all poses in one dataframe
    list_molecules = os.listdir(str(sdf_output.parent))
    df = pd.DataFrame(columns=['ID', 'molecules', 'confidence_score'])
    molecules = []
    ids = []
    confidence_scores = []

    try:
        for mol in list_molecules:
            for i in range (1,(n_poses)+1):
                pose = [f for f in os.listdir(str(sdf_output.parent / mol)) if f.startswith(f"rank{i}_")][0]
                supplier = Chem.SDMolSupplier(str(sdf_output.parent / mol / pose), sanitize=False, removeHs=False)
                molecule = [mol for mol in supplier if mol is not None][0]
                if molecule:
                    ids.append(f"{mol}_localdiffdock_{i}")
                    molecules.append(molecule)
                    if i >= 10:
                        confidence_scores.append(pose[17:-4])
                    else:
                        confidence_scores.append(pose[16:-4])
            
            shutil.rmtree(sdf_output.parent / mol, ignore_errors=True)
    except Exception as e:
        print(f"ERROR: {e}\nOutput folders might be deleted or moved to another directory")
    
    df['ID'] = ids
    df['molecules'] = molecules
    df['confidence_score'] = confidence_scores

    if sdf_output.name not in os.listdir(sdf_output.parent) and len(df) > 0:
        PandasTools.WriteSDF(df, str(sdf_output.parent / 'local_diffdock_poses.sdf') , idName='ID', molColName='molecules', properties=list(df.columns))
    else:
        print(f"Compounds are already docked and concatenated, CHECK {sdf_output}")


def flexx_docking(
        protein_file,
        sdf_output,
        current_library,
        ref_file, center_x, center_z, center_y,
        exhaustiveness,
        n_poses
):
    ref_file_sdf = ref_file.with_suffix('.sdf')
    if ref_file.suffix == '.pdb' and ref_file_sdf.name not in os.listdir(ref_file.parent):
        obabel_cmd = f"obabel -ipdb {str(ref_file)} -osdf -O {str(ref_file_sdf)}"
        subprocess.call(obabel_cmd, shell=True)

    else:
        print(f"Reference ligand is already in SDF format")


    flexx_cmd = (
        f"./software/flexx-6.0.0/flexx"
        f" --thread-count 8"
        f" -p {str(protein_file)}"
        f" -r {str(ref_file_sdf)}"
        f" -i {str(current_library)}"
        f" --max-nof-conf {str(n_poses)}"
        f" -o {str(sdf_output)}"
        
    )
    print(sdf_output)
    if sdf_output.name not in os.listdir(sdf_output.parent):
        try:
            subprocess.call(
                flexx_cmd, 
                shell=True,
                # stdout=DEVNULL,
                # stderr=STDOUT
                )
            
        except Exception as e:
            print(f"ERROR: {e}\nPlease check if FlexX is installed and the path is correct and the license in the same directory of the software")
    else:
        print(f"Compounds are already docked with FlexX v 6.0")
    
    # Rename ID column
    df = PandasTools.LoadSDF(str(sdf_output))[['ID', 'ROMol']]

    if df['ID'].str.split('_').str.len().max() == 2:
        print('ID format is incorrect, fixing it ...')
        df['ID'] = df['ID'].str.split('_').str[0] + '_flexx_' + df['ID'].str.split('_').str[1]
        PandasTools.WriteSDF(df, str(sdf_output),idName='ID', molColName='ROMol', properties=list(df.columns), allNumeric=False)
    else:
        print('ID format is correct')