import numpy as np
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core_engine import POMDPModel

def create_tiger_env() -> POMDPModel:
    S = ['left', 'right']
    A = ['listen', 'open-left', 'open-right']
    O = ['hear-left', 'hear-right']
    
    T = np.zeros((3, 2, 2))
    T[0] = [[1.0, 0.0], [0.0, 1.0]]  # Listen
    T[1] = [[0.5, 0.5], [0.5, 0.5]]  # Open-left
    T[2] = [[0.5, 0.5], [0.5, 0.5]]  # Open-right
    
    Z = np.zeros((3, 2, 2))
    Z[0] = [[0.85, 0.15], [0.15, 0.85]]
    Z[1] = [[0.5, 0.5], [0.5, 0.5]]
    Z[2] = [[0.5, 0.5], [0.5, 0.5]]
    
    R = np.zeros((2, 3))
    R[0] = [-1, -100, 10]
    R[1] = [-1, 10, -100]
    
    return POMDPModel(S, A, O, T, Z, R, 0.95, np.array([0.5, 0.5]))


class TigerEnvironment:
    def __init__(self):
        self.model = create_tiger_env()
        self.rng = np.random.default_rng()
        self.true_state = 0

    def reset(self, seed: int | None = None) -> int:
        self.rng = np.random.default_rng(seed)
        self.true_state = int(self.rng.integers(2))
        return self._listen()

    def step(self, action: str) -> tuple[int, float, bool, str]:
        if action == "listen":
            return self._listen(), -1.0, False, "listening"
        if action not in {"open-left", "open-right"}:
            return self._listen(), -1.0, False, "invalid action"

        won = (action == "open-right" and self.true_state == 0) or (
            action == "open-left" and self.true_state == 1
        )
        return self._listen(), 10.0 if won else -100.0, True, "treasure" if won else "eaten"

    def _listen(self) -> int:
        return self.true_state if self.rng.random() < 0.85 else 1 - self.true_state