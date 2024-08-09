import random

import numpy as np
import pandas as pd
import torch
import tqdm as tqdm

from pathlib import Path
from scipy.optimize import minimize
from sklearn.preprocessing import StandardScaler


def scores_preprocessing(df: pd.DataFrame) -> tuple:
    """
    Preprocess the scores for the optimization
    Args:
        df: pd.DataFrame, the scores
    Returns:
        X: torch.Tensor of shape (n_ligands, n_docking_tools, n_scoring_tools, n_poses)
        y: torch.Tensor of shape (n_ligands)
        docking_cost: np.array of shape (n_docking_tools)
        scoring_cost: np.array of shape (n_scoring_tools)
        docking_tools: list of str, the docking tools
        scoring_tools: list of str, the scoring tools
    """

    cost_dict = {
        'localdiffdock': 407.5,
        'diffdock': 407.5,
        'flexx': 3.33,
        'smina': 99.9,
        'gnina': 105.8,
        'plants': 6.85,
        'cnnscore': 0.31,
        'cnnaffinity': 0.31,
        'smina_affinity': 0.31,
        'ad4': 0.28,
        'linf9': 0.24,
        'rtmscore': 0.41,
        'vinardo': 0.29,
        'scorch': 4.63,
        'hyde': 2.0,
        'chemplp': 0.121,
        'rfscore_v1': 0.682,
        'rfscore_v2': 0.687,
        'rfscore_v3': 0.69,
        'vina_hydrophobic': 0.69,
        'vina_intra_hydrophobic': 0.69
    }

    features = [col for col in df.columns if col not in [
            'docking_method',
            'pose',
            'ID',
            'id',
            'docking_tool',
            'activity_class'
            ]]
    df_copy = df.copy()
    df_copy[features] = df_copy[features].apply(pd.to_numeric, errors='coerce')
    scaler = StandardScaler()
    df_copy[features] = scaler.fit_transform(df_copy[features])
    df_copy.fillna(df_copy.min(), inplace=True)
    df_copy['pose'] = df_copy['ID'].apply(
        lambda x: x.split('_')[2]).astype(int) - 1

    docking_tools = list(df_copy['docking_tool'].unique())
    scoring_tools = [
        f for f in features if f not in [
            'true_value',
            'cpd_per_second']]
    poses = list(df_copy['pose'].unique())

    docking_cost = np.zeros(len(docking_tools))
    rescoring_cost = np.zeros(len(scoring_tools))

    ligands = list(df_copy['id'].unique())
    values = torch.zeros((len(ligands),len(docking_tools),len(scoring_tools),len(poses))) * torch.nan
    for _, row in tqdm.tqdm(df_copy.iterrows(), total=len(df_copy)):
        for rescoring_method in scoring_tools:
            lig_idx = ligands.index(row['id'])

            dck_idx = docking_tools.index(row['docking_tool'])
            if docking_cost[dck_idx] == 0:
                docking_cost[dck_idx] = cost_dict[row['docking_tool'].lower()]
            rsc_idx = scoring_tools.index(rescoring_method)
            if rescoring_cost[rsc_idx] == 0:
                rescoring_cost[rsc_idx] = cost_dict[rescoring_method.lower()]
            values[lig_idx, dck_idx, rsc_idx, int(
                row['pose'])] = row[rescoring_method]
    X = values.max(3)[0]
    y = torch.tensor([df_copy.set_index('id').loc[ligand,'true_value'].values[0] for ligand in ligands])
    return X, y, docking_cost, rescoring_cost, docking_tools, scoring_tools


def prediction(c_r: torch.Tensor, c_d:torch.Tensor, X: torch.Tensor) -> torch.Tensor:
    """
    Predict the binding affinity
    Args:
        c_r: torch.Tensor of shape (1, 1, 15), the weights for the rescoring function
        c_d: torch.Tensor of shape (1, 5, 1), the weights for the docking function
        X: torch.Tensor of shape (n_ligands, n_docking_tools, n_scoring_tools, n_poses)
    Returns:
        affinity: torch.Tensor of shape (n_ligands), the predicted binding affinity
    """
    return (X * c_r * c_d).nan_to_num().sum(1).sum(1)


def prepare_parameters(x: np.array) -> tuple:
    """
    Prepare the parameters for the optimization
    Args:
        x: np.array of shape (20), the weights for the scoring function
    Returns:
        c_r: torch.Tensor of shape (1, 1, 15), the weights for the rescoring function
        c_d: torch.Tensor of shape (1, 5, 1), the weights for the docking function
    """
    c = torch.tensor(x)

    c_r = c[:-5].reshape(1, 1, -1)
    c_d = c[-5:].reshape(1, -1, 1)

    return c_r, c_d


def loss(
        x_rand : np.array, 
        X: torch.Tensor, 
        y: torch.Tensor, 
        docking_cost: np.array, 
        scoring_cost: np.array, 
        reg : float=0.1, 
        verbose : bool=False
        ) -> float:
    """
    Loss function for the optimization
    Args:
        x_rand: np.array of shape (20), the weights for the scoring function
        X: torch.Tensor of shape (n_ligands, n_docking_tools, n_scoring_tools, n_poses)
        y: torch.Tensor of shape (n_ligands)
        docking_cost: np.array of shape (n_docking_tools)
        scoring_cost: np.array of shape (n_scoring_tools)
        reg: float, the regularization parameter
        verbose: bool, whether to print the loss
    Returns:
        loss: float, the loss
    """
    c = torch.tensor(x_rand)
    c_r, c_d = prepare_parameters(c)
    model_loss = ((prediction(c_r, c_d, X) - y) ** 2).mean().item()
    regularization = ((c_d * docking_cost).abs().sum() +
                      (c_r * scoring_cost).abs().sum())
    if verbose:
        print(f'model loss={model_loss}')

    return model_loss + (reg * regularization)


def optimize_score(
        X: torch.Tensor, 
        y : torch.Tensor, 
        docking_cost : np.array, 
        scoring_cost : np.array, 
        reg : float =0.3, 
        iter : int =500
        ) -> tuple:
    """
    Optimize the weights for the scoring function
    Args:
        X: torch.Tensor of shape (n_ligands, n_docking_tools, n_scoring_tools, n_poses)
        y: torch.Tensor of shape (n_ligands)
        docking_cost: np.array of shape (n_docking_tools)
        scoring_cost: np.array of shape (n_scoring_tools)
        reg: float, the regularization parameter
        iter: int, number of iterations for the optimization
    Returns:
        losses: list of float, the loss for each iteration
        weights: list of np.array, the weights for each iteration
    """

    losses = []
    weights = []
    random.seed(0)
    random_seeds = [random.randint(0, 1000000) for _ in range(500)]
    len_x0 = len(docking_cost) + len(scoring_cost)
    for i in tqdm.tqdm(range(iter)):
        np.random.seed(random_seeds[i])
        x0 = np.random.rand(len_x0)
        res = minimize(
            loss,
            x0,
            args=(
                X,
                y,
                docking_cost,
                scoring_cost,
                reg),
            method='Nelder-Mead')
        losses.append(res.fun)
        weights.append(res.x)
    return losses, weights


def score_pose_optimization(
        X: torch.Tensor, 
        y : torch.Tensor, 
        docking_cost : np.array, 
        scoring_cost : np.array, 
        weights_path : Path,
        alphas : list,
        iter : int =500) -> dict:
    """
    Optimize the weights for the scoring functions with list of different regularization parameters
    Args:
        X: torch.Tensor of shape (n_ligands, n_docking_tools, n_scoring_tools, n_poses)
        y: torch.Tensor of shape (n_ligands)
        docking_cost: np.array of shape (n_docking_tools)
        scoring_cost: np.array of shape (n_scoring_tools)
        weights_path: Path to save the optimized weights
        alphas: list of float, the regularization parameters
        iter: int, number of iterations for the optimization
    Returns:
        best_weights: dict of shape (n_alphas), the best weights for each alpha
    """

    if weights_path.exists():
        with open(str(weights_path), 'rb',) as file:
            min_weights = np.load(file, allow_pickle=True)
        best_weights = min_weights.item()
        return best_weights

    best_weights = {}
    for alpha in alphas:
        print(f'Optimization with Regularization: {alpha}')
        losses, weights = optimize_score(
            X, y, docking_cost, scoring_cost, reg=alpha, iter=iter)
        min_loss_idx = np.argmin(losses)
        best_weights[alpha] = weights[min_loss_idx]
        # print(alpha)
        # print(weights[min_loss_idx])
    np.save(str(weights_path), best_weights)
    return best_weights


def mapping_normalized_weights(
        best_weights : np.array, 
        scoring_tools : list, 
        docking_tools : list
        ) -> dict:
    """
    Map the normalized weights of the optimization function to the scoring functions and docking tools

    Args:
        best_weights: the best weights for a selected alpha
        scoring_tools: list of str, the scoring tools
        docking_tools: list of str, the docking tools
    Returns:
        normalized_weights: dict of shape (n_scoring_tools + n_docking_tools), the normalized weights for each tool
    """
    # Combine scoring and docking tools into one list
    all_tools = scoring_tools + docking_tools

    # Map weights to their corresponding tools
    mapped_weights = {
        tool: weight for tool,
        weight in zip(
            all_tools,
            best_weights)}

    # Normalize docking tools weights
    docking_weights = {tool: mapped_weights[tool] for tool in docking_tools}
    min_val_docking = min(docking_weights.values())
    max_val_docking = max(docking_weights.values())
    normalized_docking_weights = {
        tool: (weight - min_val_docking) / (max_val_docking - min_val_docking)
        for tool, weight in docking_weights.items()
    }

    # Normalize scoring tools weights
    scoring_weights = {tool: mapped_weights[tool] for tool in scoring_tools}
    min_val_scoring = min(scoring_weights.values())
    max_val_scoring = max(scoring_weights.values())
    normalized_scoring_weights = {
        tool: (weight - min_val_scoring) / (max_val_scoring - min_val_scoring)
        for tool, weight in scoring_weights.items()
    }

    # Combine normalized weights back into one dictionary
    normalized_weights = {
        **normalized_docking_weights,
        **normalized_scoring_weights}

    return normalized_weights
