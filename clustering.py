from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
import pandas as pd
from rdkit import DataStructs
from rdkit.ML.Cluster import Butina
import numpy as np
from rdkit import RDLogger

RDLogger.DisableLog('rdApp.*')  


# Reduce dimensionality step
def morgan_fp_generator(mol):
    fpg = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    return fpg.GetFingerprint(mol)

def tanimoto_distance_matrix(fp_list):
    similarity_matrix = []
    for i in range(0, len(fp_list)):
        similarity_matrix.append(DataStructs.BulkTanimotoSimilarity(fp_list[i], fp_list))
    return similarity_matrix


def butina_clustering(fingerprints, cutoff=0.6):
    # distance matrix has to be 1D
    distance_matrix = tanimoto_distance_matrix(fingerprints)
    one_d_dist_mat = [element for sublist in distance_matrix for element in sublist]
    lenn = len(fingerprints)
    clusters = Butina.ClusterData(one_d_dist_mat, lenn , cutoff, isDistData=True)
    clusters = sorted(clusters, key=len, reverse=True)
    return clusters


def choice_of_cutoff(docked_df):
    fp_list = docked_df.MorganFP.to_list()
    cutoff_dict = {}
    for cutoff in np.arange(0.1, 1.0, 0.1):
        clusters = butina_clustering(fp_list, cutoff=cutoff)
        cutoff_dict[cutoff] = len(clusters)
    total_mol_number = docked_df.shape[0]//3
    if max(cutoff_dict.values()) < total_mol_number:
        return max(cutoff_dict.keys())
    else:
        suggested_cutoff = [key for key, value in cutoff_dict if value > total_mol_number]
        return suggested_cutoff
