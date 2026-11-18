import numpy as np
from pcg_benchmark.probs.smb.problem import _convert2str, _calculate_hnoise, _caculate_fenemies
from pcg_benchmark.probs.utils import get_range_reward, get_num_tiles

class State:
    def __init__(self, env, control, _random=None, content=None):
        self.env = env
        self._random = _random if _random is not None else np.random.default_rng(42)
        self._width = len(env.content_space._value)
        self.content = np.asarray(content if content is not None else env.content_space.sample())
        self.control = control

        self.info = None
        self.quality = None
        self.diversity = None
        self.controlability = None
        self.fitness = None
        self.w_quality = 8
        self.w_diversity = 4
        self.w_controlability = 1
        self.content_size_q_learning = 0

        self.floor_count = 0

    def _fitness_state(self):
        if self.quality is None:
            print("Fitness: Quality not set.")
            return 0
        sum_ws = self.w_quality+self.w_diversity+self.w_controlability
        return (self.quality*self.w_quality + self.diversity*self.w_diversity + self.controlability*self.w_controlability)/sum_ws

    def fitness_state(self):
        if self.quality is None:
            print("Fitness: Quality not set.")
            return 0
        result = self.quality
        if self.quality >=1:
            result += self.controlability
        if self.quality >= 1 and self.controlability >= 1:
            result += self.diversity
        return result/3

    def cheap_quality(self, length=None):
        """
        The level has to have minimal horizontal changes
        The level has to have unbroken pipes (as pipes spawn on two slices)
        The level has 50% or more of just flat platform to comply with mario distribution of tiles
        The level shouldn't contain very few floating enemies (most enemies should be on a solid tile)
        Skipped: The level has to be beatable by the A* agent

        length: only score the first `length` slices (the part q-learning has already placed),
        otherwise the floor that fills the rest of the level is scored too
        """
        content = self.content if length is None else self.content[:length]
        problem = self.env._problem

        lvl = _convert2str(content, problem._slices)
        lvl_lines = lvl.split("\n")
        tube_issue = 0
        for l in lvl_lines:
            test_tube = 0
            for c in l:
                if c == 't':
                    test_tube += 1
                else:
                    if test_tube % 2 > 0:
                        tube_issue += 1
                    test_tube = 0
        allowed = np.where(problem._hnoise > 0, np.maximum(problem._hnoise, 2 / max(1, len(content) - 1)), 0)
        hnoise = np.array(_calculate_hnoise(content, problem._slices)) - allowed
        hnoise[hnoise < 0] = 0
        empty = get_num_tiles(np.array([content]), [0]) / len(content)
        fenemies = _caculate_fenemies(content, problem._slices)

        # the A* agent is not run, so the level never counts as completed and quality is at most 0.5
        info = {"tube": tube_issue, "empty": empty, "noise": hnoise, "fenemies": fenemies, "complete": 0.0}
        # same as fitness_state, as quality is always below 1
        return problem.quality(info)

    def get_ngram_representation(self,n):
        size = min(self.content_size_q_learning,n)
        ## in the begining we pretend it has only floor
        output = [0,]*n
        for i in range(size):
            ## we replace the last elements of n gram with the last elements in the content
            output[-1-i] = int(self.content[self.content_size_q_learning - 1 - i])
        return output

    def q_add_element(self, element):
        if self.content_size_q_learning >= self._width:
            raise IndexError("The content is already complete, no element can be added.")
        # the content is copied so that the parent state is not modified
        new_state = State(self.env, self.control, self._random, self.content.copy())
        new_state.content[self.content_size_q_learning] = element

        new_state.floor_count = self.floor_count
        if element == 0:
            new_state.floor_count +=1

        new_state.content_size_q_learning = self.content_size_q_learning + 1
        return new_state

    def q_entry(self,n):
        return str(self.get_ngram_representation(n))

    def get_new_ngram(self,n):
        #size = min(self.content_size_q_learning,n)
        ## in the begining we pretend it has only floor
        output = [-1,]*n
        for i in range(n):
            if self.content_size_q_learning - 1 - i < 0:
                continue
            ## we replace the last elements of n gram with the last elements in the content
            output[-1-i] = int(self.content[self.content_size_q_learning - 1 - i])
        return output

    def dq_state(self, n, n_slices=126):
        ngram = self.get_new_ngram(n)
        onehot = np.zeros((n, n_slices + 1), dtype=np.float32)   # +1 = "not placed yet"
        # (needs get_ngram_representation to pad with an index of its own, e.g. num_slices)
        for i, s in enumerate(ngram):
            onehot[i, s] = 1.0
        pos = self.content_size_q_learning / self._width
        floor_ratio = self.floor_count / max(1, self.content_size_q_learning)
        return np.concatenate([onehot.ravel(), [pos, floor_ratio]])


def get_input_size_dqn(n, n_slices=126):
    return n * (n_slices + 1) + 2

def evaluate_states(env, states, cache=None):
    # only the states without a fitness are evaluated, the others (elites) keep their scores
    # returns (cache hits, states evaluated) so the generators can report the cache hit rate
    new_states = [s for s in states if s.fitness is None]
    if not new_states:
        return 0, 0

    # the A* playthrough inside env.info costs ~0.5s on a playable level and the population converges,
    # so many children are copies of earlier levels: reuse their info instead of replaying them.
    # (info is deterministic, and quality/diversity/controlability accept infos instead of contents)
    if cache is None:
        cache = {}
    infos = []
    hits = 0
    for state in new_states:
        key = state.content.tobytes()
        if key not in cache:
            cache[key] = env.info(state.content)
        else:
            hits += 1
        infos.append(cache[key])

    _, quality, _ = env.quality(infos)
    _, diversity, _, _ = env.diversity(infos)
    _, controlability, _ = env.controlability(infos, [s.control for s in new_states])
    for i, state in enumerate(new_states):
        state.info = infos[i]
        state.quality = quality[i]
        state.diversity = diversity[i]
        state.controlability = controlability[i]
        state.fitness = state.fitness_state()
    return hits, len(new_states)

def cheap_evaluate(env, states):
    new_states = [s for s in states if s.fitness is None]
    if not new_states:
        return
    for state in new_states:
        state.info = None
        state.quality = state.cheap_quality()
        state.diversity = 0
        state.controlability = 0
        state.fitness = state.fitness_state()
    
def state_fitness_softmax(states, temperature=0.1):
    fitnesses = np.array([state.fitness for state in states])
    x = np.asarray(fitnesses, dtype=float) / temperature
    # subtract the max so exp doesn't overflow at low temperatures (same probabilities)
    exp_x = np.exp(x - x.max())
    return exp_x / exp_x.sum()


