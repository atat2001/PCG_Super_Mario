import json
import os
from multiprocessing import Pool
import numpy as np
import pcg_benchmark
import generators.generator as base
from generators.state import State
from generators.genetic import mutate

# MAP-Elites (quality-diversity, lecture 83).
# The archive is a grid indexed by what the A* agent does in a level: enemies killed x jumps x coins collected.
# Those are exactly the benchmark's control parameters, so every cell is one control target and a single run
# fills in levels for many targets at once. Each cell keeps only its highest-quality level (its elite).
# New levels come from mutating (and sometimes crossing) elites picked uniformly from the filled cells, so
# diversity comes from the archive itself instead of from a term in the fitness.
BEHAVIOURS = ["enemies", "jumps", "coins"]

# the A* playthroughs of a batch are independent, so with workers > 1 they are spread over processes
_worker_env = None

def _init_worker(problem):
    global _worker_env
    _worker_env = pcg_benchmark.make(problem)

def _info(content):
    return _worker_env.info(content)

class Generator(base.Generator):
    def __init__(self, env, _random=None):
        base.Generator.__init__(self, env, _random)
        # one bin per possible control value (smb-small: enemies 0-6, jumps 0-9, coins 0-4)
        self.bins = [env.control_space._value[b]._max_value for b in BEHAVIOURS]
        self.init_size = 200
        self.batch = 100
        self.crossover = 0.5
        self.mutations = 10
        self.mutation_p = 0.2
        self.control = None
        self.archive = {}

    def reset(self, **kwargs):
        base.Generator.reset(self, **kwargs)
        self.init_size = kwargs.get("init_size", self.init_size)
        # evaluate.py/run.py pass --population to every generator: here it is the number of children per update
        self.batch = kwargs.get("population", kwargs.get("batch", self.batch))
        self.crossover = kwargs.get("crossover", self.crossover)
        self.mutations = kwargs.get("mutations", self.mutations)
        self.mutation_p = kwargs.get("mutation_p", self.mutation_p)
        self.control = kwargs.get("control") or self._env.control_space.sample()
        self.archive = {}
        self.iterations = 0
        self.inserted = 0
        self.evaluations = 0
        workers = kwargs.get("workers", 1)
        self.pool = Pool(workers, initializer=_init_worker, initargs=(self._env._name,)) if workers > 1 else None
        self.insert(self.evaluate([State(self._env, self.control, _random=self._random) for _ in range(self.init_size)]))

    def evaluate(self, states):
        contents = [s.content for s in states]
        if self.pool is not None:
            infos = self.pool.map(_info, contents, chunksize=max(1, len(contents) // (4 * self.pool._processes)))
        else:
            infos = self._env.info(contents)
        self.evaluations += len(states)
        _, quality, _ = self._env.quality(infos)
        for i, state in enumerate(states):
            state.info = infos[i]
            state.quality = float(quality[i])
        return states

    def cell(self, info):
        # levels outside the grid (e.g. more jumps than any target asks for) have no cell and are dropped
        key = tuple(int(info[b]) for b in BEHAVIOURS)
        if any(k < 0 or k >= n for k, n in zip(key, self.bins)):
            return None
        return key

    def insert(self, states):
        for state in states:
            key = self.cell(state.info)
            if key is None:
                continue
            if key not in self.archive or state.quality > self.archive[key].quality:
                self.archive[key] = state
                self.inserted += 1

    def update(self):
        elites = list(self.archive.values())
        children = []
        for _ in range(self.batch):
            parent = elites[self._random.integers(len(elites))]
            content = parent.content.copy()
            if len(elites) > 1 and self._random.random() < self.crossover:
                other = elites[self._random.integers(len(elites))]
                cut = self._random.integers(1, len(content))
                content = np.concatenate((content[:cut], other.content[cut:]))
            child = State(self._env, self.control, _random=self._random, content=content)
            mutate(child, n=self.mutations, p=self.mutation_p)
            children.append(child)
        self.insert(self.evaluate(children))
        self.iterations += 1

    def target(self):
        # the elite for the requested control; if that cell is still empty, the closest filled cell
        # (L1 distance in the grid), preferring playable levels
        key = tuple(int(self.control[b]) for b in BEHAVIOURS)
        if key in self.archive:
            return self.archive[key]
        def distance(k):
            return (self.archive[k].quality < 1, sum(abs(a - b) for a, b in zip(k, key)), -self.archive[k].quality)
        return self.archive[min(self.archive, key=distance)]

    def best(self):
        # cascaded like the GA: quality first, controllability for the target only counts once quality is 1.
        # (no diversity term: diversity is what the archive provides)
        elite = self.target()
        controlability = self._env.controlability(elite.info, self.control)[1]
        return (elite.quality + (controlability if elite.quality >= 1 else 0)) / 2

    def best_content(self):
        return self.target().content

    def coverage(self):
        playable = sum(1 for e in self.archive.values() if e.quality >= 1)
        return len(self.archive) / int(np.prod(self.bins)), playable / int(np.prod(self.bins))

    def stats(self):
        filled, playable = self.coverage()
        qualities = [e.quality for e in self.archive.values()]
        return {"best": self.best(), "mean": float(np.mean(qualities)), "coverage": filled,
                "playable_coverage": playable, "qd_score": float(np.sum(qualities)),
                "evaluations": self.evaluations}

    def save(self, folderpath):
        os.makedirs(folderpath, exist_ok=True)
        with open(f"{folderpath}/archive.json", "w") as f:
            json.dump({"bins": self.bins, "control": {k: int(v) for k, v in self.control.items()},
                       "stats": self.stats(),
                       "elites": [{"cell": list(k), "quality": e.quality, "content": [int(v) for v in e.content]}
                                  for k, e in self.archive.items()]}, f)
        self._env.render(self.target().content).save(f"{folderpath}/best.png")
