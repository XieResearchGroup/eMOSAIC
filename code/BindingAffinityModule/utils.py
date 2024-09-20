from rdkit.Chem.Scaffolds import MurckoScaffold
from collections import defaultdict, Counter
from itertools import compress
from datetime import datetime
from math import ceil
import numpy as np
import random
import torch
import math
import os


def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def set_up_exp_folder(path):
    now = datetime.now()
    timestamp = now.strftime("%d-%m-%Y-%H-%M-%S")
    print('timestamp: ',timestamp)
    save_folder = path
    if os.path.exists(save_folder) == False:
            os.mkdir(save_folder)
    checkpoint_dir = '{}/exp{}/'.format(save_folder, timestamp)
    if os.path.exists(checkpoint_dir ) == False:
            os.mkdir(checkpoint_dir )
    return checkpoint_dir

def pad_or_truncate_tensor(tensor, threshold = 500):
    """
    function to pad or truncate the tensors
    """
    x = tensor.size(0)
    if x < threshold:
        # Pad the tensor with zeros
        padding = torch.zeros((threshold - x, 1280))
        padded_tensor = torch.cat((tensor, padding), dim=0)
        return padded_tensor
    elif x > threshold:
        truncated_tensor = tensor[:threshold]
        return truncated_tensor
    else:
        return tensor


def update_metrics(metrics_dict, prefix, metrics, exclude_keys=None):
    if exclude_keys is None:
        exclude_keys = set()
    for key, value in metrics.items():
        if key in exclude_keys:
            continue
        full_key = f"{prefix}_{key}"
        metrics_dict[full_key].append(value)

def random_pfam_split(
    df,
    frac_train=0.7,
    frac_valid=0.1,
    frac_test=0.2,
    seed=42,
):
    """
    Split dataframe based on Pfam families.
    :param df: pandas DataFrame with 'pfam' column
    :param frac_train:
    :param frac_valid:
    :param frac_test:
    :param seed;
    :return: train, valid, test slices of the input DataFrame
    """

    np.testing.assert_almost_equal(frac_train + frac_valid + frac_test, 1.0)

    rng = np.random.RandomState(seed)

    pfams = defaultdict(list)
    for idx, row in df.iterrows():
        pfam = row['pfam']
        pfams[pfam].append(idx)

    pfam_sets = rng.permutation(list(pfams.values()))

    n_total_valid = int(np.floor(frac_valid * len(df)))
    n_total_test = int(np.floor(frac_test * len(df)))

    train_idx = []
    valid_idx = []
    test_idx = []

    for pfam_set in pfam_sets:
        if len(valid_idx) + len(pfam_set) <= n_total_valid:
            valid_idx.extend(pfam_set)
        elif len(test_idx) + len(pfam_set) <= n_total_test:
            test_idx.extend(pfam_set)
        else:
            train_idx.extend(pfam_set)

    train_dataset = df.loc[train_idx]
    valid_dataset = df.loc[valid_idx]
    test_dataset = df.loc[test_idx]

    return train_dataset, valid_dataset, test_dataset


def generate_scaffold(smiles, include_chirality=False):
    """
    Obtain Bemis-Murcko scaffold from smiles
    :param smiles:
    :param include_chirality:
    :return: smiles of scaffold
    """
    scaffold = MurckoScaffold.MurckoScaffoldSmiles(
        smiles=smiles, includeChirality=include_chirality
    )
    return scaffold

def random_scaffold_split(
    dataset,
    smiles_list,
    task_idx=None,
    null_value=0,
    frac_train=0.7,
    frac_valid=0.1,
    frac_test=0.2,
    seed=42,
):
    """
    Adapted from https://github.com/pfnet-research/chainer-chemistry/blob/master/\
        chainer_chemistry/dataset/splitters/scaffold_splitter.py
    Split dataset by Bemis-Murcko scaffolds
    This function can also ignore examples containing null values for a
    selected task when splitting. Deterministic split
    :param dataset: pytorch geometric dataset obj
    :param smiles_list: list of smiles corresponding to the dataset obj
    :param task_idx: column idx of the data.y tensor. Will filter out
    examples with null value in specified task column of the data.y tensor
    prior to splitting. If None, then no filtering
    :param null_value: float that specifies null value in data.y to filter if
    task_idx is provided
    :param frac_train:
    :param frac_valid:
    :param frac_test:
    :param seed;
    :return: train, valid, test slices of the input dataset obj
    """

    np.testing.assert_almost_equal(frac_train + frac_valid + frac_test, 1.0)

    if task_idx is not None:
        y_task = np.array([data.y[task_idx].item() for data in dataset])
        non_null = y_task != null_value
        smiles_list = list(compress(enumerate(smiles_list), non_null))
    else:
        non_null = np.ones(len(dataset)) == 1
        smiles_list = list(compress(enumerate(smiles_list), non_null))

    rng = np.random.RandomState(seed)

    scaffolds = defaultdict(list)
    for ind, smiles in smiles_list:
        scaffold = generate_scaffold(smiles, include_chirality=True)
        scaffolds[scaffold].append(ind)

    scaffold_sets = rng.permutation(list(scaffolds.values()))

    n_total_valid = int(np.floor(frac_valid * len(dataset)))
    n_total_test = int(np.floor(frac_test * len(dataset)))

    train_idx = []
    valid_idx = []
    test_idx = []

    for scaffold_set in scaffold_sets:
        if len(valid_idx) + len(scaffold_set) <= n_total_valid:
            valid_idx.extend(scaffold_set)
        elif len(test_idx) + len(scaffold_set) <= n_total_test:
            test_idx.extend(scaffold_set)
        else:
            train_idx.extend(scaffold_set)

    train_dataset = dataset.iloc[train_idx]
    valid_dataset = dataset.iloc[valid_idx]
    test_dataset = dataset.iloc[test_idx]

    return train_dataset, valid_dataset, test_dataset

def calibrate_mean_var(matrix, m1, v1, m2, v2, clip_min=0.1, clip_max=10):
    if torch.sum(v1) < 1e-10:
        return matrix
    if (v1 == 0.).any():
        valid = (v1 != 0.)
        factor = torch.clamp(v2[valid] / v1[valid], clip_min, clip_max)
        matrix[:, valid] = (matrix[:, valid] - m1[valid]) * torch.sqrt(factor) + m2[valid]
        return matrix

    factor = torch.clamp(v2 / v1, clip_min, clip_max)
    return (matrix - m1) * torch.sqrt(factor) + m2
