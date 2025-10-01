import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg
import pandas as pd
import glob
import os

def fetch_blist(filepath):
    loaded_data = pd.read_csv(filepath, header=None)
    blist = loaded_data.to_numpy()[:,0]
    return blist

def eigensystem_from_blist(blist):
    diag = np.zeros(len(blist)+1)
    envals, envecs = linalg.eigh_tridiagonal(diag, blist)
    return envals, envecs

def save_eigensystem(envals, envecs, filepath):
    dirname, basename = os.path.split(filepath)

    new_prefix = "evals"
    new_basename = basename.replace('bn', new_prefix, 1)
    new_filepath = os.path.join(dirname,"processed" ,new_basename)

    np.savetxt(new_filepath, envals, delimiter=',')

    new_prefix = "evecs"
    new_basename = basename.replace('bn', new_prefix, 1)
    new_filepath = os.path.join(dirname,"processed" ,new_basename)

    np.savetxt(new_filepath, envecs, delimiter=',')
    return



def fetch_data(nspins, system_type, operator_label):
    allowed_systems = ['intg', 'nonintg']
    if system_type not in allowed_systems:
        raise ValueError(f"System type must be one of {allowed_systems}")
    allowed_operators = ['X1', 'Z1', 'X2', 'Z2', 'X3', 'Z3', '1s']
    if operator_label not in allowed_operators:
        raise ValueError(f"Operator label must be one of {allowed_operators}")
    allowed_nspins = [5, 6, 7]
    if nspins not in allowed_nspins:
        raise ValueError(f"Number of spins must be one of {allowed_nspins}")
    
    suffix = f"{system_type}_{nspins}_{operator_label}.csv"
    
    filepath = f"data/n={nspins}/bn_{suffix}"
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
    
    blist = pd.read_csv(filepath, header=None).to_numpy()[:,0]

    evals_filepath = f"data/n={nspins}/processed/evals_{suffix}"
    evecs_filepath = f"data/n={nspins}/processed/evecs_{suffix}"

    if not os.path.exists(evals_filepath):
        raise FileNotFoundError(f"File not found: {evals_filepath}")
    if not os.path.exists(evecs_filepath):
        raise FileNotFoundError(f"File not found: {evecs_filepath}")
    evals = pd.read_csv(evals_filepath, header=None).to_numpy()[:,0]
    evecs = pd.read_csv(evecs_filepath, header=None).to_numpy()

    return blist, evals, evecs

def fetch_all_operators(nspins, system_type):
    allowed_systems = ['intg', 'nonintg']
    if system_type not in allowed_systems:
        raise ValueError(f"System type must be one of {allowed_systems}")
    allowed_nspins = [5, 6, 7]
    if nspins not in allowed_nspins:
        raise ValueError(f"Number of spins must be one of {allowed_nspins}")


    directory = f'data/n={nspins}'

    # Get all CSV file paths in the directory
    csv_files = glob.glob(os.path.join(directory, "*.csv"))

    prefix = f"bn_{system_type}_{nspins}_"
    csv_files = [f for f in csv_files if os.path.basename(f).startswith(prefix)]
    if not csv_files:
        raise FileNotFoundError(f"No files found for nspins={nspins}, system_type={system_type}")
    
    operators = set()
    for filepath in csv_files:
        basename = os.path.basename(filepath)
        operator = basename.split('_')[3].split('.')[0]
        operators.add(operator)

    results = {}
    
    for operator in operators:
        blist, evals, evecs = fetch_data(nspins, system_type, operator)
        results[operator] = (blist, evals, evecs)
    
    return results

# blist, evals, evecs = fetch_data(6, 'intg', 'X2')

# results = fetch_all_operators(6, 'intg')