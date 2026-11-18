import numpy as np
import generators.generator as base
from generators.state import State, evaluate_states

# random search baseline: every update samples a fresh batch of levels and keeps the best one seen.
# uses the same cascaded fitness as the GA so the two are comparable.
class Generator(base.Generator):
    def __init__(self, env, population=200, _random=None):
        base.Generator.__init__(self, env, _random)
        self.population = population
        self.control = None
        self.best_found = None

    def reset(self, **kwargs):
        base.Generator.reset(self, **kwargs)
        self.population = kwargs.get("population", self.population)
        self.control = kwargs.get("control") or self._env.control_space.sample()
        self.best_found = None
        self.last_mean = 0.0
        self.update()

    def update(self):
        states = [State(self._env, self.control, _random=self._random) for _ in range(self.population)]
        evaluate_states(self._env, states)
        self.last_mean = float(np.mean([s.fitness for s in states]))
        batch_best = max(states, key=lambda s: s.fitness)
        if self.best_found is None or batch_best.fitness > self.best_found.fitness:
            self.best_found = batch_best

    def best(self):
        return float(self.best_found.fitness)

    def best_content(self):
        return self.best_found.content

    def stats(self):
        b = self.best_found
        return {"best": float(b.fitness), "mean": self.last_mean, "quality": float(b.quality),
                "controlability": float(b.controlability), "diversity": float(b.diversity)}
