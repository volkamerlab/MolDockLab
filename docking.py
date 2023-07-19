import os


def gnina_docking(
        snapshot_ID,
        sdf_name,
        current_library,
        ref_file,
        exhaustiveness,
        n_poses):

    gnina_cmd = f'./software/gnina -r data/{snapshot_ID}/protein_protoss_noligand.pdb -l {current_library} --autobox_ligand {ref_file} -o data/{snapshot_ID}/{sdf_name} --seed 1637317264 --exhaustiveness {exhaustiveness} --num_modes ' + str(
        n_poses) + ' --cnn_scoring rescore --cnn crossdock_default2018 --no_gpu --log data/log.txt'
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
