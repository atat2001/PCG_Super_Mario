# Procedural Level Generation with Evolutionary Search and Reinforcement Learning

Generating playable, varied and controllable **Super Mario Bros levels** with AI approaches, compared on one benchmark under identical conditions:

| Approach | Family | File |
|---|---|---|
| Random search | Baseline | `generators/random_search.py` |
| Genetic algorithm | Evolutionary optimisation | `generators/genetic.py` |
| Tabular Q-learning | Reinforcement learning | `generators/q.py` |
| Deep Q-learning (DQN) | Deep reinforcement learning (PyTorch) | `generators/dql.py` |

The point of the project is not only to get a generator working, but to **measure** which method works best, and why, with a fair evaluation harness.

## What this project demonstrates

- **Applied ML across paradigms:** evolutionary search, tabular RL and deep RL implemented on the same problem, behind one common interface.
- **Deep RL engineering in PyTorch:** replay buffer, target network, Huber loss, gradient clipping, epsilon-greedy exploration, minibatch training, CUDA when available, model checkpointing.
- **Reward design:** potential-based reward shaping that gives dense feedback without changing which levels are best.
- **Rigorous evaluation:** every generator gets the same seeds and the same control targets. Results are scored on a set of levels with quality, diversity and controllability, not on a single cherry-picked level.
- **Practical engineering:** multiprocessing for the expensive evaluation step, an evaluation cache, reproducible seeds, per-run saved results and plots.

## The problem

A level is a row of **slices** (columns of tiles), and the generator has to choose a slice for each position. A good level must be:

- **Playable (quality):** an A* agent playing the level must be able to finish it, and the level has to respect basic design rules (unbroken pipes, mostly flat ground, enemies standing on solid tiles).
- **Diverse:** levels differ from each other.
- **Controllable:** given a requested target (number of enemies killed, jumps, coins collected), the level must make the agent do about that.

The environment is the open-source [PCG Benchmark](pcg_benchmark) (`smb-small-v0`, vendored in this repo). Evaluating a level means running a real A* playthrough, which takes on the order of 0.5 s, so **sample efficiency matters**. A method that needs fewer evaluations wins.

## The algorithms

### Random search (baseline)
Every step samples a fresh batch of random levels and keeps the best one seen. It sets the floor that every other method has to beat.

### Genetic algorithm
A population of 200 levels is evolved over generations:
- **Selection:** parents are sampled with a softmax over fitness. The temperature falls over time, so early generations explore and late ones exploit.
- **Crossover:** a single cut point, taking the left part from one parent and the right part from the other.
- **Mutation:** random slices are replaced. The mutation count and rate decay linearly over the run (coarse changes first, fine tuning later).
- **Elitism:** the top 10% survive unchanged.
- **Fitness** is *cascaded*: quality comes first. Controllability only counts once the level is playable, and diversity only once both are satisfied. This stops the search from rewarding unplayable levels that happen to hit the target.
- A cache of A* results avoids replaying identical levels, since the population converges and many children are copies.

### Tabular Q-learning
Level generation is framed as a sequential decision problem (in the style of PCGRL):
- **Episode:** build the level left to right, one slice per step.
- **State:** the position bucket and the last slice placed.
- **Action:** which of the 125 slices to place next.
- **Reward:** potential-based shaping. After each step the reward is the scaled change in a cheap, A*-free quality score of the prefix placed so far, and on the last step the real A* fitness replaces it. With discount 1 the shaping terms cancel, so an episode's return equals the scaled final fitness. The shaping gives early feedback without changing which levels are best.
- **Exploration:** epsilon-greedy with a linearly decaying epsilon. The generator's output is the best greedy level seen so far.

The control target is not in the state (it would multiply the table by 350), so the agent is trained for one target per run.

### Deep Q-learning (DQN)
The same environment and reward as Q-learning, but the table is replaced by a neural network so the state can be much richer:
- **State:** a one-hot encoding of the last *n* slices, plus the relative position and the floor ratio, as a vector the network can generalise over.
- **Network:** a small MLP that outputs one Q-value per slice.
- **Stability techniques:** a **replay buffer** (each update trains on the new transitions plus the same number of older ones), a **target network** synced every few updates, the **Huber (smooth L1) loss** (outlier-robust, because bootstrapped targets are noisy), **gradient-norm clipping**, and Adam.
- **Terminal handling:** the bootstrap term is dropped on the last slice of a level.
- **Hardware:** runs on CUDA when available, otherwise CPU. Checkpoints (`dqn.pt`) restore on either.

A sanity test with a fixed set of terminal transitions drives the loss to about zero, which confirms the training loop is correct. With non-terminal bootstrapping on random states the loss can diverge, which is the known instability of bootstrapped Q-learning and the reason the target network and `target_every` are tunable.

## Results

16 runs per generator on `smb-small-v0`, up to 3000 steps, seeds 1 to 16, same seeds and control targets for every generator. Quality, diversity and controllability are means of the benchmark's continuous scores. The pass columns give the fraction of levels that fully pass each criterion.

| Generator | Quality | Diversity | Controllability | Diversity pass | Controllability pass | Mean time per run |
|---|---|---|---|---|---|---|
| Random search | 0.26 | 0.06 | 0.19 | 6% | 0% | 42 s |
| Genetic | 1.00 | 1.00 | 0.97 | 100% | 56% | 258 s |
| Q-learning | 1.00 | 0.93 | 0.69 | 50% | 6% | 979 s |

**Takeaways**
- Random search almost never produces a playable level. Every structured method beats it.
- The genetic algorithm reaches full quality and does well on diversity and control.
- Tabular Q-learning learns playable levels, but it is weakest on controllability, because the target is not part of its state.
- **The DQN generator is implemented and runs end to end, but it has not been benchmarked yet.** It is not in this table.

Plots (fitness curves, radar chart, episode reward) are written to `results/plots/`.

## Design decisions worth a conversation

- **Cascaded fitness** instead of a weighted sum, so one criterion cannot be traded off against playability.
- **Potential-based shaping** for RL, to speed up learning without introducing reward hacking.
- **Fair comparison:** the harness passes the same seed and control target to every generator, and diversity is scored over a *set* of levels, since it is meaningless for a single level.
- **Cost-aware design:** an A* result cache in the GA and an A*-free cheap quality score for RL shaping, all aimed at reducing the number of expensive evaluations.

## Limitations (stated honestly)

- Results are from one problem (`smb-small-v0`), 16 seeds. They show a clear ranking, but they are not a statistical study across problems.
- The RL agents see a short window of the level (last *n* slices, position, floor ratio). Richer state (progress features, the control target) is the obvious next step and is not done yet.
- DQN hyperparameters (network size, *n*, `target_every`, discount) have not been tuned or swept.

## Running it

```bash
cd project
pip install -r requirements.txt          # includes the vendored benchmark; DQN also needs torch

# compare generators: N runs each, same seeds and targets, scored as a set
python evaluate.py results -g genetic random_search q dql -n 16 -s 400 --seed 1
```

Extra `--key value` pairs are passed to every generator's `reset`, for example `--learning_rate 0.0005` or `--epsilon_decay 0.99`.

## Repository layout

```
project/
  evaluate.py            evaluation harness (quality, diversity, controllability over a set)
  run.py                 run a single generator
  analyse.py             plots and tables
  generators/
    generator.py         common interface: reset / update / best / save / load
    state.py             level wrapper, fitness, A*-free quality, RL state encoding
    random_search.py  genetic.py  q.py  dql.py
pcg_benchmark/           the benchmark environment (vendored)
```
