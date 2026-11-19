# Super Mario Level Generation with AI

This project generates **Super Mario Bros levels** automatically. It compares three AI algorithms: a genetic algorithm, Q-learning and Deep Q-learning (DQN).

A level is a row of **slices** (columns of tiles). The agent has to choose which slice goes in each position. A good level is:
- **playable:** an A* agent can finish it
- **varied:** it differs from other generated levels
- **controllable:** it matches a requested target (for example, a number of enemies, jumps and coins)

## Technologies

- **Python**
- **NumPy** for the numerical work
- **PyTorch** for the neural network (DQN), with CUDA support when a GPU is available
- **[PCG Benchmark](pcg_benchmark)** as the game environment. It plays each level with an A* agent and scores it.

## The algorithms

### 1. Genetic algorithm
Inspired by evolution. It keeps a **population** of levels and improves it over generations:
1. Score every level.
2. Pick the better levels as **parents**.
3. **Crossover:** build a child by taking the first part of one parent and the rest from another.
4. **Mutation:** randomly change a few slices of the child.
5. Keep the best levels unchanged (**elitism**) and repeat.

Over time the population fills up with better and better levels.

### 2. Q-learning
A **reinforcement learning** method. The AI builds a level one slice at a time, from left to right. Each choice is an **action**, and at the end it gets a **reward** based on how good the level is.

It stores a **Q-table**: for each situation (where it is in the level and the last slice placed), a score for each possible next slice. After each try it updates those scores, so choices that led to good levels get a higher score. It also tries random slices some of the time (**exploration**), so it can discover better options.

### 3. Deep Q-learning (DQN)
The same idea as Q-learning, but the table is replaced by a **neural network**. The network takes the last n (3) slices, the percentage of level that is floor (due to the quality requirement) and the percentage of level left as input and outputs a score for every possible slice.

To keep training stable it uses:
- a **replay buffer**, so it learns from a mix of new and older experience
- a **target network**, a slowly updated copy of the network that gives steadier training targets
- the **Huber loss** and **gradient clipping**, so a few extreme values don't wreck the training, as the last reward of the episode is much bigger.

## Running it

```bash
cd project
pip install -r requirements.txt       # DQN also needs torch

python evaluate.py results -g genetic q dql -n 16 -s 400 --seed 1
```

`-g` chooses the algorithms, `-n` the number of runs, and `-s` the number of steps per run. The results are written to the `results/` folder.

## Results

Mean scores over 16 levels per algorithm.

| Algorithm | Quality | Diversity | Controllability |
|---|---|---|---|
| Genetic algorithm | 1.000 | 1.000 | 0.973 |
| Q-learning | 1.000 | 0.934 | 0.685 |
| Deep Q-learning | 1.000 | 0.851 | 0.486 |

## Project layout

```
project/
  evaluate.py           runs the algorithms and scores the levels
  generators/
    genetic.py          genetic algorithm
    q.py                Q-learning
    dql.py              Deep Q-learning (DQN)
pcg_benchmark/          the game environment and scoring
```
