
from utilities import get_ligand_coordinates_centers, run_command, split_sdf, any_in_list, pdb_converter
from concurrent.futures import ProcessPoolExecutor
from software.RTMScore.rtmscore_modified import *
import subprocess
from rdkit.Chem import PandasTools
import pandas as pd
import os
import csv


                
def rescoring_function(
    rescore_programs,
    protein_file,
    docked_library_path,
    ref_file,
):
        docking_dict = { 
        'gnina_rescoring': gnina_rescoring, 
        'ad4': ad4_rescoring,  
        'linf9': linf9_rescoring, 
        'rtmscore': rtmscore_rescoring, 
        'vinardo': vinardo_rescoring, 
        'scorch': scorch_rescoring, 
        'chemplp': chemplp_rescoring, 
        'hyde': hyde_rescoring,
        'rfscorevs_v1': rfscorevs_V1_rescoring,
        'rfscorevs_v2': rfscorevs_V2_rescoring,
        'rfscorevs_v3': rfscorevs_v3_rescoring, 
        'vina_hydrophobic': vina_hydrophobic_rescoring, 
        'vina_intra_hydrophobic': vina_intra_hydrophobic_rescoring,
        }
        # Create folder for rescoring results
        results_folder = docked_library_path.parent / 'rescoring_results'
        results_folder.mkdir(exist_ok=True)
        num_cpus = os.cpu_count() - 2 if os.cpu_count() > 1 else 1
        # num_cpus = 1
        center_x, center_z, center_y = get_ligand_coordinates_centers(str(ref_file))

        # convert protein and ligand to mol2 and pdbqt format in case of CHEMPLP and SCORCH respectively
        pdb_converter(protein_file, rescore_programs)
                  
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

            elif program in docking_dict.keys():
                # Run scoring functions in parellel

                    print(f'Running {program} in parallel')
                    with ProcessPoolExecutor(max_workers=num_cpus) as executor:
                        futures = [
                            executor.submit(
                                run_command, 
                                docking_dict[program](
                                protein_file,
                                file_path,
                                ref_file, center_x, center_z, center_y,
                                output_folder / f'{program}_{i}.sdf'
                                )
                            )       
                    for i, file_path in enumerate(splitted_file_paths)
                    ]
                    
            read_rescoring_results(results_folder, program)
        
        merge_rescoring_results(results_folder,rescore_programs)


def ad4_rescoring(
        protein_file, 
        docked_library_path, 
        ref_file, center_x, center_y, center_z,
        output_path
        ) -> str:
    
    return (
        './software/gnina'
        f' --receptor {protein_file}'
        f' --ligand {str(docked_library_path)}' 
        f' --out {str(output_path)}'
        f' --autobox_ligand {str(ref_file)} --autobox_extend=1' 
        ' --score_only'
        ' --scoring ad4_scoring --cnn_scoring none'
    )

def gnina_rescoring(
        protein_file, 
        docked_library_path, 
        ref_file, center_x, center_y, center_z,
        output_path
        ) -> str:
    return (
        './software/gnina'
        f' --receptor {protein_file}'
        f' --ligand {str(docked_library_path)}' 
        f' --out {str(output_path)}'
        f' --autobox_ligand {str(ref_file)} --autobox_extend=1' 
        ' --score_only'
        ' --cnn crossdock_default2018 --no_gpu'
    )

def vinardo_rescoring(
        protein_file, 
        docked_library_path, 
        ref_file, center_x, center_y, center_z,
        output_path
        ) -> str:
    return (
        './software/gnina'
        f' --receptor {protein_file}'
        f' --ligand {str(docked_library_path)}' 
        f' --out {str(output_path)}'
        f' --autobox_ligand {str(ref_file)} --autobox_extend=1' 
        ' --score_only'
        ' --scoring vinardo --cnn_scoring none'
    )

def chemplp_rescoring(
        protein_file, 
        docked_library_path, 
        ref_file, center_x, center_y, center_z,
        output_path
        ) -> str:
        protein_file = protein_file.parent / f'{protein_file.stem}.mol2'

        plants_search_speed = 'speed1'
        ants = '20'

        chemplp_config = ['# search algorithm\n',
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
                          f'protein_file {str(protein_file)}\n',
                          f'ligand_file {str(docked_library_path)}\n',

                          '# output\n',
                          f'output_dir {str(output_path.parent /  output_path.stem)}\n',

                          '# write single mol2 files (e.g. for RMSD calculation)\n',
                          'write_multi_mol2 1\n',

                          '# binding site definition\n',
                          f'bindingsite_center {center_x} {center_y} {center_z}\n',
                          'bindingsite_radius 5\n',

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

def linf9_rescoring(
        protein_file, 
        docked_library_path, 
        ref_file, center_x, center_y, center_z,
        output_path
        ) -> str:

        return (
            f'./software/smina.static'
            f' --receptor {str(protein_file)}'
            f' --ligand {str(docked_library_path)}'
            f' --out {str(output_path)}'
            f' --autobox_ligand {str(ref_file)} --autobox_extend=1' 
            ' --scoring Lin_F9 --score_only'
        )    

def rtmscore_rescoring(
        protein_file, 
        docked_library_path, 
        ref_file, center_x, center_y, center_z,
        output_path
        ) -> str:
     
     # RTMScore needs the pocket not the protein
    print('Protein path is replaced with the pocket path by adding _pocket to the end of the file name.\n'
           ,'please check the name of the pocket path to make sure it is correct\n\n')
    print(protein_file)
    RTMScore_pocket = str(protein_file).replace('.pdb', '_pocket.pdb')
    print(RTMScore_pocket)
    number_of_ligand = docked_library_path.stem.split('_')[-1]

    try:
        return rtmscore(
                prot=str(RTMScore_pocket),
                lig=str(docked_library_path),
                output=str(output_path.parent / f'rtmscore_{number_of_ligand}.csv'),
                model='software/RTMScore/trained_models/rtmscore_model1.pth',
                ncpus= 1
                )
    except:
        print('RTMScore failed with the pocket, trying with the protein, it would take more time \n\n')
        return rtmscore(
                prot=str(protein_file),
                lig=str(docked_library_path),
                output=str(output_path.parent / f'rtmscore_{number_of_ligand}.csv'),
                model='software/RTMScore/trained_models/rtmscore_model1.pth',
                ncpus=  1
                )


def scorch_rescoring(
        protein_file, 
        docked_library_path, 
        ref_file, center_x, center_y, center_z,
        output_path
        ):
        return (
            f'python software/SCORCH/scorch.py '
            f' --receptor {str(protein_file)} '
            f' --ligand {docked_library_path}' 
            f" --ref_lig {str(ref_file)}"
            f' --out {str(output_path.parent / output_path.stem)}.csv'
            ' --return_pose_scores'
        )

def rfscorevs_V1_rescoring(
        protein_file, 
        docked_library_path, 
        ref_file, center_x, center_y, center_z,
        output_path
        ):
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
        ref_file, center_x, center_y, center_z,
        output_path
        ):
    return (
        "software/hydescorer-2.0.0/hydescorer"
        f" -i {str(docked_library_path)}"
        f" -o {str(output_path)}"
        f" -p {str(protein_file)}"
        f" -r {str(ref_file)}"
)

def rfscorevs_V2_rescoring(
        protein_file, 
        docked_library_path, 
        ref_file, center_x, center_y, center_z,
        output_path
        ):
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
        ref_file, center_x, center_y, center_z,
        output_path
        ):
    # if 'rfscorevs_v3' in os.listdir(docked_library_path.parent): 
                #    os.listdir(docked_library_path.parent
    if os.listdir(docked_library_path.parent / 'rfscorevs_v3'):
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

def vina_hydrophobic_rescoring(
        protein_file, 
        docked_library_path, 
        ref_file, center_x, center_y, center_z,
        output_path
        ):
    #@TODO : check if the poses are already rescored through RF-Score-V3, if yes, copy the file to the output path
    if any_in_list(['rfscorevs_v3', 'vina_hydrophobic', 'vina_intra_hydrophobic'], 
                   os.listdir(docked_library_path.parent)):
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

def vina_intra_hydrophobic_rescoring(
        protein_file, 
        docked_library_path, 
        ref_file, center_x, center_y, center_z,
        output_path
        ):
    if any_in_list(['rfscorevs_v3', 'vina_hydrophobic', 'vina_intra_hydrophobic'], 
                   os.listdir(docked_library_path.parent)):
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
    


def read_rescoring_results(
        rescoring_results_path,
        rescore_program
        ):
    dfs = []
    print('\n\nReading rescoring results ⌛ ...\n\n')

    if f'{rescore_program}_rescoring.csv' in os.listdir(rescoring_results_path):
        print(f'{rescore_program} is already read')
        return
    
    if 'gnina_rescoring' == rescore_program:
        for sdf in (rescoring_results_path / rescore_program).glob('*.sdf'):
            df = PandasTools.LoadSDF(str(sdf))[['ID', 'CNNscore', 'CNNaffinity', 'minimizedAffinity']]
            df.rename(columns={'minimizedAffinity': 'gnina_affinity'}, inplace=True)
            dfs.append(df)
  

    if 'ad4' == rescore_program:
        for sdf in (rescoring_results_path / rescore_program).glob('*.sdf'):
            df = PandasTools.LoadSDF(str(sdf))[['ID', 'minimizedAffinity']]
            df.rename(columns={'minimizedAffinity': 'ad4'}, inplace=True)
            dfs.append(df)
        
    if 'linf9' == rescore_program:
        for sdf in (rescoring_results_path / rescore_program).glob('*.sdf'):
            df = PandasTools.LoadSDF(str(sdf))[['ID', 'minimizedAffinity']]
            df.rename(columns={'minimizedAffinity': 'LinF9'}, inplace=True)
            dfs.append(df)

    if 'vinardo' == rescore_program:
        for sdf in (rescoring_results_path / rescore_program).glob('*.sdf'):
            df = PandasTools.LoadSDF(str(sdf))[['ID', 'minimizedAffinity']]
            df.rename(columns={'minimizedAffinity': 'Vinardo'}, inplace=True)
            dfs.append(df)

    if 'rfscorevs_v1' == rescore_program:
        for sdf in (rescoring_results_path / rescore_program).glob('*.sdf'):
            df = PandasTools.LoadSDF(str(sdf))[['ID', 'rfscore_v1']]
            # df.rename(columns={'minimizedAffinity': 'Vinardo'}, inplace=True)
            dfs.append(df)

    if 'rfscorevs_v2' == rescore_program:
        for sdf in (rescoring_results_path / rescore_program).glob('*.sdf'):
            df = PandasTools.LoadSDF(str(sdf))[['ID', 'rfscore_v2']]
            # df.rename(columns={'minimizedAffinity': 'Vinardo'}, inplace=True)
            dfs.append(df)

    if 'rfscorevs_v3' == rescore_program:
        for sdf in (rescoring_results_path / rescore_program).glob('*.sdf'):
            df = PandasTools.LoadSDF(str(sdf))[['ID', 'rfscore_v3']]
            # df.rename(columns={'minimizedAffinity': 'Vinardo'}, inplace=True)
            dfs.append(df)

    if rescore_program in ['vina_hydrophobic', 'vina_intra_hydrophobic']:
        if os.listdir(str(rescoring_results_path / 'rfscorevs_v3')):
            rescore_path = rescoring_results_path / 'rfscorevs_v3'
        else:
            rescore_path = rescoring_results_path / rescore_program

        for sdf in rescore_path.glob('*.sdf'):
            df = PandasTools.LoadSDF(str(sdf))

            if 'vina_hydrophobic' == rescore_program:
                df = df[['ID', 'vina_hydrophobic']]

            elif 'vina_intra_hydrophobic' == rescore_program:
                df = df[['ID', 'vina_intra_hydrophobic']]
            else:
                df = df[['ID', 'vina_hydrophobic', 'vina_intra_hydrophobic']]
            # df.rename(columns={'minimizedAffinity': 'Vinardo'}, inplace=True)
            dfs.append(df)

            

    if 'rtmscore' == rescore_program:
        for csv_file in (rescoring_results_path / rescore_program).glob('*.csv'):
            df = pd.read_csv(csv_file)
            # split Pose ID column with - to remove last number
            df['Pose ID'] = df['Pose ID'].str.split('-').str[0]
            df.rename(columns={'Pose ID': 'ID'}, inplace=True)
            dfs.append(df)

    if 'chemplp' == rescore_program:
        for file_number in range(len(os.listdir(rescoring_results_path / rescore_program))):
            df = pd.read_csv(rescoring_results_path / rescore_program / f'{rescore_program}_{file_number}' / 'ranking.csv')[['LIGAND_ENTRY', 'TOTAL_SCORE']]
            # df = PandasTools.LoadSDF(str(sdf))[['ID', 'CHEMPLP']]
            df.rename(columns={'LIGAND_ENTRY': 'Pose ID', 'TOTAL_SCORE': 'CHEMPLP'}, inplace=True)
            df['Pose ID'] = df['Pose ID'].str.split('_').str[0:3].str.join('_')
            df.rename(columns={'Pose ID': 'ID'}, inplace=True)
            dfs.append(df)
    if 'scorch' == rescore_program:
        for csv_file in (rescoring_results_path / rescore_program).glob('*.csv'):
            df = pd.read_csv(csv_file)[['Ligand_ID','SCORCH_pose_score']]

            df['ID'] = PandasTools.LoadSDF(str(rescoring_results_path.parent / 'sdf_split' / f"{df.loc[0,'Ligand_ID']}.sdf"))['ID']
            df.rename(columns={'SCORCH_pose_score': 'SCORCH'}, inplace=True)
            dfs.append(df.drop(columns=['Ligand_ID']))

    if 'hyde' == rescore_program:
        for sdf in (rescoring_results_path / rescore_program).glob('*.sdf'):
            
            df = PandasTools.LoadSDF(str(sdf))[['ID', 'BIOSOLVEIT.HYDE_ESTIMATED_AFFINITY_LOWER_BOUNDARY [nM]']]
            df.rename(columns={'BIOSOLVEIT.HYDE_ESTIMATED_AFFINITY_LOWER_BOUNDARY [nM]': 'HYDE'}, inplace=True)
            dfs.append(df)

    print(rescore_program, (rescoring_results_path / rescore_program))
    csv_file = pd.concat(dfs, ignore_index=True)
    csv_file.to_csv(rescoring_results_path / rescore_program / f'{rescore_program}_rescoring.csv', index=False)
    # return merged_df


def merge_rescoring_results(
        rescoring_results_path,
        rescore_programs
        ):
    all_rescoring_dfs = []
    for rescore_program in rescore_programs:
        if f'{rescore_program}_rescoring.csv' in os.listdir(rescoring_results_path / rescore_program):
            df = pd.read_csv(str(rescoring_results_path / rescore_program / f'{rescore_program}_rescoring.csv'))
            all_rescoring_dfs.append(df)
        else:
            print(f'{rescore_program} is not excuted')
            return

    merged_df = all_rescoring_dfs[0]
    for df in all_rescoring_dfs[1:]:
        
        merged_df = pd.merge(merged_df, df, on='ID', how='outer')
    merged_df.to_csv(rescoring_results_path.parent / 'all_rescoring_results.csv', index=False)

    return merged_df