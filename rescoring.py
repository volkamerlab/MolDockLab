import os
import warnings
from utility import rank_correlation, oddt_correlation

def rescoring_functions(docking_methods, scoring_methods, snapshot_ID, data_size, common_ID):

    if 'rescoring' not in os.listdir(f'data/{snapshot_ID}'):
        os.mkdir(f'data/{snapshot_ID}/rescoring/')
        print('creating rescoring directory ...')
    else:
        print('rescoring directory is already created.')
    protein_path = f'data/{snapshot_ID}/protein_protoss_noligand.pdb'
    ref_path = f'data/{snapshot_ID}/ref_ligand.pdb'
    for docking_method in docking_methods:
        print(f'{docking_method.upper()} is now rescored ... \n\n\n')
        docked_library = f"data/A/docked_{docking_method}_poses_A_{data_size}.sdf"
        

        for scoring_method in scoring_methods:
            print(f'{scoring_method.upper()} Running ... \n\n\n')

            rescoring_results_path = f"data/{snapshot_ID}/rescoring/docked_{docking_method}_{scoring_method}_results_{data_size}.sdf"

            if scoring_method == "rf-score-vs":
                rescoring_cmd = f'./software/rf-score-vs --receptor {protein_path} {docked_library} -o sdf -O {rescoring_results_path} -n 1'
            
            if scoring_method == "vinardo":
                rescoring_cmd =  f'./software/gnina -r {protein_path} -l {docked_library} --autobox_ligand {ref_path} -o {rescoring_results_path} --score_only --scoring vinardo --cnn_scoring none --no_gpu'

            if scoring_method == "ad4":
                rescoring_cmd =  f'./software/gnina -r {protein_path} -l {docked_library} --autobox_ligand {ref_path} -o {rescoring_results_path} --score_only --scoring ad4_scoring --cnn_scoring none --no_gpu'
            
            if scoring_method == "oddt":
                rescoring_cmd = f"oddt_cli {docked_library} --receptor data/{snapshot_ID}/protein_protoss_noligand.mol2 --score rfscore_v1_pdbbind2016 --score rfscore_v2_pdbbind2016 --score rfscore_v3_pdbbind2016 --score plecrf_pdbbind2016 -O {rescoring_results_path}"
            
            
            # Filter the UserWarning
            warnings.filterwarnings("ignore", category=UserWarning)
            os.system(rescoring_cmd + ' > /dev/null')
            warnings.filterwarnings("default", category=UserWarning)
           
           
            if scoring_method == "oddt":
                oddt_correlation(rescoring_results_path, common_ID)
            else:
                rank_correlation(rescoring_results_path, common_ID)

