import json
import os
import numpy as np
import generators.generator as base
from generators.state import State, evaluate_states, state_fitness_softmax

class Generator(base.Generator):
    def __init__(self, env, population=200, _random=None):
        base.Generator.__init__(self, env, _random)
        self.population = population
        self.elitism = 0.1
        self.states = []
        self.control = None
        self.iterations = 0
        self.steps=None

    def reset(self, **kwargs):
        # the base reset replaces self._random when a seed is given, so it has to run
        # before the states are created
        base.Generator.reset(self, **kwargs)
        self.population = kwargs.get("population", self.population)
        self.elitism = kwargs.get("elitism", self.elitism)
        self.iterations = 0
        self.info_cache = {}
        self.cache_hits = 0
        self.cache_lookups = 0
        # every state in the population is judged against the same control
        # (the evaluation harness passes one in so every generator gets the same targets)
        self.control = kwargs.get("control") or self._env.control_space.sample()
        self.states = [State(self._env, self.control, _random=self._random) for _ in range(self.population)]
        self.steps = kwargs.get("steps", self.steps)
        self.evaluate_states()

    def cross_over(self, parent1, parent2):
        if self._random.random() > 0.5:
            parent1, parent2 = parent2, parent1

        cut = self._random.integers(1, parent1._width)

        new_content = np.concatenate((parent1.content[:cut], parent2.content[cut:]))
        return State(self._env, self.control, _random=self._random, content=new_content)

    def evaluate_states(self):
        hits, lookups = evaluate_states(self._env, self.states, cache=self.info_cache)
        self.cache_hits += hits
        self.cache_lookups += lookups

    def nr_elites(self):
        return int(round(self.population * self.elitism))

    def select_states(self):
        nr_states_selected = self.population - self.nr_elites()

        ## a temperatura deve ir de 1 para 0.01, independente dos steps.
        temperature_dropoff = 0.1/(1 + 99*(self.iterations/self.steps))

        # metemos o softmax com temperatura a diminuir com o tempo
        p = state_fitness_softmax(self.states, temperature=temperature_dropoff)

        parent_pairs_index = [self._random.choice(len(self.states), size=2, replace=False, p=p) for _ in range(nr_states_selected)]

        return [[self.states[parent_1], self.states[parent_2]] for [parent_1, parent_2] in parent_pairs_index]

    # um heap seria mais eficiente
    def select_elites(self):
        k = self.nr_elites()
        if k == 0:
            return []
        return sorted(self.states, key=lambda s: s.fitness)[-k:]

    def update(self):
        ## evaluates states on first itertions. After that it shouldnt change anything.
        self.evaluate_states()
        parent_pairs = self.select_states()
        childs = []
        for pair in parent_pairs:
            childs.append(self.cross_over(pair[0], pair[1]))

        percentage_steps_done = 1-((self.steps-self.iterations)/self.steps)

        nr_mutacoes = int(round(30 - percentage_steps_done* (30-5)))
        percentagem_mutacao = 0.3 - percentage_steps_done* (0.3-0.05)
        for child in childs:
            mutate(child, n=nr_mutacoes, p=percentagem_mutacao)

        self.iterations += 1
        self.states = self.select_elites() + childs
        self.evaluate_states()

    def best_state(self):
        return max(self.states, key=lambda s: s.fitness)

    def best(self):
        # the fitness is normalised by the weights, so it is in [0, 1] and early stopping (best >= 1) works
        return float(self.best_state().fitness)

    def best_content(self):
        return self.best_state().content

    def stats(self):
        # per-generation numbers for the fitness evolution plots
        fitnesses = np.array([s.fitness for s in self.states])
        best = self.best_state()
        return {"best": float(best.fitness), "mean": float(fitnesses.mean()),
                "quality": float(best.quality), "controlability": float(best.controlability),
                "diversity": float(best.diversity),
                "cache_hit_rate": self.cache_hits / max(1, self.cache_lookups),
                "evaluations": self.cache_lookups}

    def save(self, folderpath):
        os.makedirs(folderpath, exist_ok=True)
        state = self.best_state()
        with open(f"{folderpath}/best.json", "w") as f:
            json.dump({"content": [int(v) for v in state.content],
                       "control": {k: int(v) for k, v in self.control.items()},
                       "quality": float(state.quality),
                       "diversity": float(state.diversity),
                       "controlability": float(state.controlability),
                       "fitness": float(state.fitness)}, f)
        self._env.render(state.content).save(f"{folderpath}/best.png")


def mutate(state, n=3, p=0.2):
    ## tries to do n mutations, each at p probability
    hits = state._random.random(n) < p
    k = int(hits.sum())
    if k:
        index_to_mutate = state._random.integers(0, state._width, size=k)
        state.content[index_to_mutate] = np.asarray(state.env.content_space.sample())[index_to_mutate]
