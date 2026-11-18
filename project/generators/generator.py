import numpy as np
from abc import ABC, abstractmethod
class Generator:
    def __init__(self, env, _random=None):
        self._env = env
        self._random = _random if _random is not None else np.random.default_rng(42)

    def reset(self, **kwargs):
        if kwargs.get('seed'):
            self._random = np.random.default_rng(kwargs.get('seed'))

    def update(self):
        return NotImplementedError("The update function is not implemented, please make sure to implment it.")
    
    def best(self):
        return NotImplemented("The best function returns the current state of the generation (content).")
    
    def best_content(self):
        raise NotImplementedError("The best_content function returns the best content generated so far (used by evaluate.py).")

    def save(self, folderpath):
        raise NotImplementedError("The save function is not implemented, please make sure to implment it.")
    
    def load(self, folderpath):
        return NotImplementedError("The load function is not implemented, please make sure to implment it.")