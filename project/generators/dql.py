import json
import os
import copy
import numpy as np
import generators.generator as base
from generators.state import State, get_input_size_dqn
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import deque

class DQN(nn.Module):
    def __init__(self, input_size, layer_nr, layer_size, output_size) -> None:
        super().__init__()
        self.layer_nr = layer_nr
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(input_size, layer_size))
        if layer_nr-2 > 0:
            for _ in range(layer_nr-2):
                self.layers.append(nn.Linear(layer_size, layer_size))
        self.layers.append(nn.Linear(layer_size, output_size))
        

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = F.relu(layer(x))
        return self.layers[-1](x)  ## Does not apply the relu to last


class ReplayBuffer():
    def __init__(self, max_len):
        self.max_len = max_len
        self.replay_buffer = deque(maxlen=max_len)

    def sample(self, n, rng):
        # rng.choice(a, size=None, replace=True, p=None)
        # choose over the indexes: numpy would try to turn the list of transitions into one ragged array
        size = min(n, len(self.replay_buffer))
        indexes = rng.choice(len(self.replay_buffer), size=size, replace=False)
        return [self.replay_buffer[i] for i in indexes]

    def append(self, transition):
        self.replay_buffer.append(transition)

    def extend(self, transitions):
        self.replay_buffer.extend(transitions)

    def __len__(self):
        return len(self.replay_buffer)


class Generator(base.Generator):
    # evaluate.py saves the trained Q-table of every run (results/q/brain_<seed>/)
    saves_brain = True

    def __init__(self, env, batch=10, _random=None, discount=0.99, learning_rate=0.001,
                 epsilon_start=1, epsilon_end=0.01, epsilon_decay = 0.995, reward_scale=10):
        base.Generator.__init__(self, env, _random)
        self.batch = batch
        self.discount = discount
        self.learning_rate = learning_rate

        self.epsilon_decay = epsilon_decay
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.reward_scale = reward_scale
        self.width = len(env.content_space._value)

        # the number of different slices that can be placed, "max" is excluded
        self.n_actions = env.content_space.range()["max"]
        self.control = None
        self.n = 3                  # length of the n-gram in the state

        self.minibatch = 64         # transitions per gradient step
        self.epochs = 1             # passes over the training data per rollout
        self.target_every = 10      # the target network is synced every N trainings
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.build()

    def build(self):
        # (re)creates everything that is learned, so every run starts from scratch.
        # called from reset as well: the learning rate and batch can change there
        input_size = get_input_size_dqn(self.n, self.n_actions)
        self.dqn = DQN(input_size=input_size, layer_nr=4, layer_size=256, output_size=self.n_actions).to(self.device)
        self.target = copy.deepcopy(self.dqn)
        self.optimizer = optim.Adam(self.dqn.parameters(), lr=self.learning_rate)
        self.buffer = ReplayBuffer(10 * self.batch * self.width)
        self.train_calls = 0

    def reset(self, **kwargs):
        base.Generator.reset(self, **kwargs)
        for key in ["batch", "discount", "learning_rate", "epsilon_start", "epsilon_end", "epsilon_decay",
                    "reward_scale", "minibatch", "epochs", "target_every"]:
            setattr(self, key, kwargs.get(key, getattr(self, key)))
        self.build()
        self.control = kwargs.get("control") or self._env.control_space.sample()
        self.steps = kwargs.get("steps", 1)
        self.iterations = 0
        self.episode_rewards = []
        self.last_explore_fitness = 0.0
        if kwargs.get("brain"):
            # use a trained network instead of starting from scratch
            self.load(kwargs["brain"])
        self.greedy = self.rollout(epsilon=0, learn=False)[0]
        self.best_greedy = self.greedy

    def get_action(self, state, epsilon):
        # state: the vector from State.dq_state. rollout runs this inside torch.no_grad()
        if self._random.random() < epsilon:
            return int(self._random.integers(self.n_actions))
        q_values = self.dqn(torch.as_tensor(state, dtype=torch.float32, device=self.device))    # (n_actions,)
        # ties (e.g. an untrained network) are broken at random instead of always picking slice 0
        q_values = q_values.cpu().numpy()
        return int(self._random.choice(np.flatnonzero(q_values == q_values.max())))

    def potential(self, state):
        if state.content_size_q_learning == 0:
            return 0.0
        return state.cheap_quality(length=state.content_size_q_learning)

    def train(self, current_transitions):
        # the new transitions plus the same amount of older ones from the buffer
        data = current_transitions + self.buffer.sample(len(current_transitions), self._random)

        # list of [s, a, r, s2] rows -> one tensor per field, built once
        s, a, r, s2 = zip(*data)
        s  = torch.as_tensor(np.array(s),  dtype=torch.float32, device=self.device)     # (N, input_size)
        s2 = torch.as_tensor(np.array(s2), dtype=torch.float32, device=self.device)     # (N, input_size)
        a  = torch.as_tensor(np.array(a),  dtype=torch.int64, device=self.device)       # (N,)
        r  = torch.as_tensor(np.array(r),  dtype=torch.float32, device=self.device)     # (N,)
        # a transition is final when its second state is the complete level: the position feature
        # (second to last of dq_state, content_size / width) is exactly 1.0 there
        done = (s2[:, -2] >= 1.0).float()                           # (N,)
        N = len(data)

        losses = []
        for _ in range(self.epochs):
            perm = torch.as_tensor(self._random.permutation(N), device=self.device)     # one shuffle for all the fields
            for i in range(0, N, self.minibatch):
                idx = perm[i:i + self.minibatch]

                # Q(s)[a]: only the action that was taken can be trained
                q = self.dqn(s[idx]).gather(1, a[idx].unsqueeze(1)).squeeze(1)      # (B,)
                with torch.no_grad():       # the target is a label, no gradients through it
                    y = r[idx] + self.discount * (1 - done[idx]) * self.target(s2[idx]).max(1).values

                loss = F.smooth_l1_loss(q, y)
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.dqn.parameters(), 10.0)
                self.optimizer.step()
                losses.append(loss.item())
        self.last_loss = float(np.mean(losses))

        # only now: the old data was sampled before the new one was added
        self.buffer.extend(current_transitions)
        self.train_calls += 1
        if self.train_calls % self.target_every == 0:
            self.target.load_state_dict(self.dqn.state_dict())

    def rollout(self, epsilon, learn):
        # builds `batch` levels (1 when acting greedily) from scratch, one slice per step
        n = self.batch if learn else 1
        levels = [State(self._env, self.control, _random=self._random, content=np.zeros(self.width, dtype=int))
                  for _ in range(n)]
        returns = np.zeros(n)
        phis = np.zeros(n)
        current_transitions = []
        with torch.no_grad():
            for step in range(self.width):
                terminal = step == self.width - 1
                states = [s.dq_state(self.n, self.n_actions) for s in levels]
                actions = [self.get_action(k, epsilon) for k in states]
                levels = [s.q_add_element(a) for s, a in zip(levels, actions)]
                if terminal:
                    self.final_fitness(levels)
                    new_phis = np.array([s.fitness for s in levels])
                else:
                    new_phis = np.array([self.potential(s) for s in levels])
                rewards = self.reward_scale * (new_phis - phis)
                returns += rewards
                if learn:
                    for k, a, r, s in zip(states, actions, rewards, levels):
                        current_transitions.append([k, a, r, s.dq_state(self.n, self.n_actions)])
                phis = new_phis
        if learn:
            self.train(current_transitions)
        return levels[0] if not learn else levels, returns


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

    def epsilon(self):
        return max(self.epsilon_end, self.epsilon_start * self.epsilon_decay ** self.iterations)

    def update(self):
        # one training episode (a batch of levels built with epsilon-greedy), then one greedy level with the
        # current network: the output is the best greedy level seen so far
        levels, returns = self.rollout(self.epsilon(), learn=True)
        self.episode_rewards.append(float(returns.mean()))
        self.last_explore_fitness = float(np.mean([s.fitness for s in levels]))
        self.iterations += 1
        self.greedy = self.rollout(epsilon=0, learn=False)[0]
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
                "epsilon": self.epsilon(), "buffer": len(self.buffer)}

    # ---- the brain ----

    def save(self, folderpath):
        os.makedirs(folderpath, exist_ok=True)
        torch.save({"dqn": self.dqn.state_dict(), "target": self.target.state_dict(),
                    "iterations": self.iterations, "train_calls": self.train_calls,
                    "n": self.n, "n_actions": self.n_actions}, f"{folderpath}/dqn.pt")
        with open(f"{folderpath}/best.json", "w") as f:
            json.dump({"content": [int(v) for v in self.best_greedy.content],
                       "control": {k: int(v) for k, v in self.control.items()},
                       "quality": self.best_greedy.quality, "controlability": self.best_greedy.controlability,
                       "fitness": self.best_greedy.fitness}, f)
        self._env.render(self.best_greedy.content).save(f"{folderpath}/best.png")

    def load(self, folderpath):
        data = torch.load(f"{folderpath}/dqn.pt", map_location=self.device)
        self.dqn.load_state_dict(data["dqn"])
        self.target.load_state_dict(data["target"])
        self.iterations = data["iterations"]
        self.train_calls = data["train_calls"]
