import numpy as np
import itertools
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core_engine import POMDPModel

def create_k6_attack_graph() -> POMDPModel:
    K = 6
    S = list(itertools.product([0, 1], repeat=K))
    A = [f"{act}_{i}" for act in ["scan", "patch", "exploit"] for i in range(K)] + ["wait"]
    O = [0, 1]
    
    num_s, num_a, num_o = len(S), len(A), len(O)
    T = np.zeros((num_a, num_s, num_s))
    Z = np.zeros((num_a, num_s, num_o))
    R = np.zeros((num_s, num_a))
    
    dependencies = {0: [], 1: [], 2: [0], 3: [0, 1], 4: [2, 3], 5: [2, 3]}
    
    for a_idx, action in enumerate(A):
        for s_idx, state in enumerate(S):
            # Default: state remains unchanged
            T[a_idx, s_idx, s_idx] = 1.0 
            
            # Observation Noise (Scan actions)
            if action.startswith("scan"):
                target = int(action.split("_")[1])
                is_vuln = state[target] == 1
                Z[a_idx, s_idx, 1] = 0.85 if is_vuln else 0.10
                Z[a_idx, s_idx, 0] = 0.15 if is_vuln else 0.90
            else:
                Z[a_idx, s_idx, :] = 0.5 # Uninformative for non-scans
                
            # TODO: Add specific exploit/patch transition logic and reward math here
            
    mu_0 = np.ones(num_s) / num_s
    return POMDPModel(S, A, O, T, Z, R, 0.95, mu_0)