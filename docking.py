import os
from data_preparation import prepare_diffdock_input

def run_diffdock(snapshot, data_size):
    protein_path = f'../../data/{snapshot}/protein_protoss.pdb'
    ligand_path = f'data/ligands/gypsum_dl_success_cleaned_{data_size}.sdf'
    output_path = 'data/ligands/ligands_protein_complex.csv'

    prepare_diffdock_input(protein_path, ligand_path, output_path)

    os.system('export PYTHONPATH=/home/hamza/Github/HitsForECFT/software/DiffDock:$PYTHONPATH')
    os.system('export PYTHONPATH=/home/hamza/Github/HitsForECFT/software/DiffDock/esm:$PYTHONPATH')

    os.chdir('software/DiffDock/')

    protein_ligand_file = "../../data/protein_ligands_complex.csv"
    results_path = '../../data/DiffDock/results/diffdock'

    diffdock_cmd = f'python -m inference --protein_ligand_csv {protein_ligand_file} --out_dir {results_path} --inference_steps 20 --samples_per_complex 5 --batch_size 10 --actual_steps 18 --no_final_step_noise'
    test_cmd = "python -m inference --protein_path ../../data/A/protein_protoss.pdb  --ligand_description 'CCCCC(NC(=O)CCC(=O)O)P(=O)(O)OC1=CC=CC=C1' --out_dir results/user_predictions_small --inference_steps 20 --samples_per_complex 40 --batch_size 1 --actual_steps 18"
    os.system(diffdock_cmd)
    os.chdir('../../')


def gnina_docking(snapshot_ID, sdf_name, current_library, ref_file):
    n_poses = '3'
    gnina_cmd = f'./software/gnina -r data/{snapshot_ID}/protein_protoss_noligand.pdb -l {current_library} --autobox_ligand {ref_file} -o data/{snapshot_ID}/{sdf_name} --seed 1637317264 --exhaustiveness 8 --num_modes '+str(n_poses)+' --cnn_scoring rescore --cnn crossdock_default2018 --no_gpu --log data/log.txt'
    if sdf_name not in os.listdir(f'data/{snapshot_ID}/'):
        os.system(gnina_cmd)
    else:
        print(f"Snapshot {snapshot_ID} is already docked with GNINA v 1.0")
    

def smina_docking(snapshot_ID, sdf_name, current_library, ref_file):
    n_poses = '3'
    smina_cmd = f'./software/gnina -r data/{snapshot_ID}/protein_protoss_noligand.pdb -l {current_library} --autobox_ligand {ref_file} -o data/{snapshot_ID}/{sdf_name} --autobox_extend=1 --seed 1637317264 --exhaustiveness 8 --num_modes {n_poses} --cnn_scoring=none'

    if sdf_name not in os.listdir(f'data/{snapshot_ID}/'):
        os.system(smina_cmd)
    else:
        print(f"Snapshot {snapshot_ID} is already docked with SMINA")