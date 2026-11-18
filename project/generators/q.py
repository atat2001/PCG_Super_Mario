import json
import os
import numpy as np
import generators.generator as base
from generators.state import State

# Tabular Q-learning (lecture 8) inside a PCGRL-style loop (lecture 83).
# The agent builds a level left to right, one slice per step:
#   state  = (position bucket, last placed slice): where in the level we are and what is to the left
#   action = which of the 125 slices to place next
#   reward = shaped by the A*-free quality of the slices placed so far, plus the real evaluation at the end
# Only the placed prefix is scored. Scoring the whole level (prefix + the floor padding) made every action look
# bad or neutral: a flat level already has the maximum A*-free score, so no reward was ever positive.
#
# The rewards are potential-based: r_t = scale * (phi(s_t+1) - phi(s_t)), with phi = A*-free quality of the
# prefix, and on the last slice phi is replaced by the real fitness (A* run). With discount 1 the shaping terms
# cancel, so an episode's return is scale * final fitness: the shaping only gives earlier feedback, it does not
# change which levels are best.
#
# The control target is not in the state: with 7 x 10 x 5 = 350 targets the table would be 350x bigger.
# So the agent is trained for one target per run (reset(control=...)), which is also how the GA is evaluated.

class Generator(base.Generator):
    # evaluate.py saves the trained Q-table of every run (results/q/brain_<seed>/)
    saves_brain = True

    def __init__(self, env, batch=10, _random=None, discount=1.0, learning_rate=0.1, buckets=5,
                 epsilon_start=0.5, epsilon_end=0.05, reward_scale=10):
        base.Generator.__init__(self, env, _random)
        self.batch = batch
        self.discount = discount
        self.learning_rate = learning_rate
        self.buckets = buckets
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.reward_scale = reward_scale
        self.width = len(env.content_space._value)
        # the number of different slices that can be placed, "max" is excluded
        self.n_actions = env.content_space.range()["max"]
        self.q_table = {}
        self.control = None

    def reset(self, **kwargs):
        base.Generator.reset(self, **kwargs)
        for key in ["batch", "discount", "learning_rate", "buckets", "epsilon_start", "epsilon_end", "reward_scale"]:
            setattr(self, key, kwargs.get(key, getattr(self, key)))
        self.control = kwargs.get("control") or self._env.control_space.sample()
        self.steps = kwargs.get("steps", 1)
        self.iterations = 0
        self.q_table = {}
        self.episode_rewards = []
        self.last_explore_fitness = 0.0
        if kwargs.get("brain"):
            # use a trained table instead of starting from scratch
            self.load(kwargs["brain"])
        self.greedy = self.rollout(epsilon=0, learn=False)[0]
        self.best_greedy = self.greedy

    # ---- table ----

    def key(self, state):
        position = state.content_size_q_learning
        last = int(state.content[position - 1]) if position > 0 else -1
        return (position * self.buckets // self.width, last)

    def values(self, key):
        row = self.q_table.get(key)
        return np.zeros(self.n_actions) if row is None else row

    def choose(self, key, epsilon):
        if self._random.random() < epsilon:
            return int(self._random.integers(self.n_actions))
        values = self.values(key)
        # ties (e.g. a state never seen) are broken at random instead of always picking slice 0
        return int(self._random.choice(np.flatnonzero(values == values.max())))

    def update_q(self, key, action, reward, next_key, terminal):
        row = self.q_table.setdefault(key, np.zeros(self.n_actions))
        future = 0 if terminal else self.values(next_key).max()
        row[action] = (1 - self.learning_rate) * row[action] + self.learning_rate * (reward + self.discount * future)

    # ---- episodes ----

    def potential(self, state):
        if state.content_size_q_learning == 0:
            return 0.0
        return state.cheap_quality(length=state.content_size_q_learning)

    def final_fitness(self, states):
        # the real evaluation (A* playthrough), cascaded like the other generators: quality first, then control.
        # diversity is left out: it depends on the other levels of the batch, not on this agent's choices
        infos = self._env.info([s.content for s in states])
        _, quality, _ = self._env.quality(infos)
        _, controlability, _ = self._env.controlability(infos, self.control)
        for i, state in enumerate(states):
            state.info = infos[i]
            state.quality = float(quality[i])
            state.controlability = float(controlability[i])
            state.diversity = 0.0
            state.fitness = (state.quality + (state.controlability if state.quality >= 1 else 0)) / 2

    def rollout(self, epsilon, learn):
        # builds `batch` levels (1 when acting greedily) from scratch, one slice per step
        n = self.batch if learn else 1
        levels = [State(self._env, self.control, _random=self._random, content=np.zeros(self.width, dtype=int))
                  for _ in range(n)]
        returns = np.zeros(n)
        phis = np.zeros(n)
        for step in range(self.width):
            terminal = step == self.width - 1
            keys = [self.key(s) for s in levels]
            actions = [self.choose(k, epsilon) for k in keys]
            levels = [s.q_add_element(a) for s, a in zip(levels, actions)]
            if terminal:
                self.final_fitness(levels)
                new_phis = np.array([s.fitness for s in levels])
            else:
                new_phis = np.array([self.potential(s) for s in levels])
            rewards = self.reward_scale * (new_phis - phis)
            returns += rewards
            if learn:
                for k, a, r, s in zip(keys, actions, rewards, levels):
                    self.update_q(k, a, r, self.key(s), terminal)
            phis = new_phis
        return levels[0] if not learn else levels, returns

    def epsilon(self):
        progress = min(self.iterations / max(1, self.steps), 1)
        return self.epsilon_start - progress * (self.epsilon_start - self.epsilon_end)

    def update(self):
        # one training episode (a batch of levels built with epsilon-greedy), then one greedy level with the
        # current table: that greedy level is what the generator outputs
        levels, returns = self.rollout(self.epsilon(), learn=True)
        self.episode_rewards.append(float(returns.mean()))
        self.last_explore_fitness = float(np.mean([s.fitness for s in levels]))
        self.iterations += 1
        self.greedy = self.rollout(epsilon=0, learn=False)[0]
        # the greedy level still changes a lot between episodes while the table is learning, so the output is
        # the best greedy level seen so far (keeping the best snapshot of the policy)
        if self.greedy.fitness > self.best_greedy.fitness:
            self.best_greedy = self.greedy

    def best(self):
        return float(self.best_greedy.fitness)

    def best_content(self):
        return self.best_greedy.content

    def stats(self):
        return {"best": self.best(), "mean": self.last_explore_fitness, "greedy": float(self.greedy.fitness),
                "quality": self.best_greedy.quality, "controlability": self.best_greedy.controlability,
                "episode_reward": self.episode_rewards[-1] if self.episode_rewards else 0.0,
                "epsilon": self.epsilon(), "q_states": len(self.q_table)}

    # ---- the brain ----

    def save(self, folderpath):
        os.makedirs(folderpath, exist_ok=True)
        with open(f"{folderpath}/q_table.json", "w") as f:
            json.dump({"buckets": self.buckets, "discount": self.discount, "learning_rate": self.learning_rate,
                       "iterations": self.iterations,
                       "control": {k: int(v) for k, v in self.control.items()},
                       "q_table": {f"{b},{last}": [round(float(v), 5) for v in row]
                                   for (b, last), row in self.q_table.items()}}, f)
        with open(f"{folderpath}/best.json", "w") as f:
            json.dump({"content": [int(v) for v in self.best_greedy.content],
                       "control": {k: int(v) for k, v in self.control.items()},
                       "quality": self.best_greedy.quality, "controlability": self.best_greedy.controlability,
                       "fitness": self.best_greedy.fitness}, f)
        self._env.render(self.best_greedy.content).save(f"{folderpath}/best.png")

    def load(self, folderpath):
        with open(f"{folderpath}/q_table.json") as f:
            data = json.load(f)
        self.buckets = data["buckets"]
        self.discount = data["discount"]
        self.learning_rate = data["learning_rate"]
        self.iterations = data["iterations"]
        self.q_table = {tuple(int(v) for v in key.split(",")): np.array(row) for key, row in data["q_table"].items()}
