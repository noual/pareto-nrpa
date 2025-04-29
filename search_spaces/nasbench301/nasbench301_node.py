import copy
import itertools
from collections import namedtuple
import random
from typing import Tuple
import sys

from pymoo.core.crossover import Crossover
from pymoo.core.evaluator import Evaluator
from pymoo.core.mutation import Mutation
from pymoo.core.population import Population
from pymoo.core.problem import ElementwiseProblem
from pymoo.core.sampling import Sampling
from pymoo.core.variable import get

from search_spaces.nasbench201.nasbench201_node import NASBench201Vertice, NASBench201Cell
from naslib.utils import get_dataset_api
from naslib.search_spaces.core import Metric

from naslib.search_spaces import NasBench301SearchSpace
from naslib.search_spaces.nasbench301.conversions import convert_genotype_to_compact, make_compact_mutable, \
    make_compact_immutable, convert_compact_to_genotype, convert_genotype_to_naslib

sys.path.append("~/projets/nas_ntk")
import numpy as np

from graphviz import Digraph
# from naslib2.search_spaces.nasbench301.conversions import convert_genotype_to_naslib, convert_genotype_to_compact, \
#     make_compact_immutable, convert_compact_to_genotype, make_compact_mutable
#
# from naslib2.search_spaces import NasBench301SearchSpace
#
# from naslib2.search_spaces.core import Metric
#
# from monet.search_spaces.nasbench201_node import NASBench201Cell, NASBench201Vertice


N_NODES = 6
ADJACENCY_MATRIX_SIZE = N_NODES ** 2
N_OPERATIONS = 7

class DARTSVertice(NASBench201Vertice):

    def __init__(self, id):
        super().__init__(id)
        self.id = id
        if id == 1:  # Il y a deux input cells
            self.actions = {}
        else:
            self.actions = {i: "none" for i in
                            range(id)}  # la valeur est l'opération qu'on joue entre deux représentations
        self.OPERATIONS = ["max_pool_3x3", "avg_pool_3x3", "skip_connect", "sep_conv_3x3",
                           "sep_conv_5x5", "dil_conv_3x3", "dil_conv_5x5"]  # Les labels qu'on peut assigner

    def get_n_predecessors(self):
        actions = [i for i in self.actions.values() if i != "none"]
        return len(actions)

    def get_action_tuples(self):
        n_predecessors = self.get_n_predecessors()
        if self.id > 1 and n_predecessors < 2:
            list_tuples = []
            for k, v in self.actions.items():
                if v == "none":
                    for op in self.OPERATIONS:
                        list_tuples.append((k, op))
            return list_tuples


    def is_complete(self):
        if self.id > 1:
            n_predecessors = self.get_n_predecessors()
            is_complete = (n_predecessors == 2)
            return is_complete
        else:
            return True


class DARTSCell(NASBench201Cell):

    def __init__(self, n_vertices=6, vertice_type=DARTSVertice):
        # Only 6 nodes because the output node is defined as the concatenation of all other nodes.
        super().__init__(n_vertices, vertice_type)

    def to_genotype(self):
        genotype = []
        for vertice in self.vertices[2:]:
            actions = {k: v for k, v in vertice.actions.items() if v != "none"}
            for k, v in actions.items():
                genotype.append((v, k))
        return genotype

    def get_action_tuples(self):
        list_tuples = []
        for i, v in enumerate(self.vertices[2:]):
            if not v.is_complete():
                actions = v.get_action_tuples()
                for action in actions:
                    list_tuples.append((v.id, *action))
        return list_tuples

    def adjacency_matrix(self):
        adjacency_matrix = np.zeros((self.n_vertices, self.n_vertices), dtype="int8")
        for i, vertice in enumerate(self.vertices):
            for j, operation in vertice.actions.items():
                if operation is None or operation == "none":
                    op_label = 0
                else:
                    op_label = self.OPERATIONS.index(operation)
                adjacency_matrix[j,i] = op_label
        return adjacency_matrix

    def plot(self, filename="cell"):
        genotype = self.to_genotype()
        g = Digraph(
                format='pdf',
                edge_attr=dict(fontsize='20', fontname="times"),
                node_attr=dict(style='filled, rounded', shape='rect', align='center', fontsize='20', height='0.5',
                               width='0.5', penwidth='2', fontname="times"),
                engine='dot')
        g.body.extend(['rankdir=LR'])

        g.node("c_{k-2}", fillcolor='darkseagreen2')
        g.node("c_{k-1}", fillcolor='darkseagreen2')
        assert len(genotype) % 2 == 0
        steps = len(genotype) // 2

        for i in range(steps):
            g.node(str(i), fillcolor='lightblue')

        for i in range(steps):
            for k in [2 * i, 2 * i + 1]:
                op, j = genotype[k]
                if j == 0:
                    u = "c_{k-2}"
                elif j == 1:
                    u = "c_{k-1}"
                else:
                    u = str(j - 2)
                v = str(i)
                g.edge(u, v, label=op, fillcolor="gray")
        g.node("c_{k}", fillcolor='palegoldenrod')
        for i in range(steps):
            g.edge(str(i), "c_{k}", fillcolor="gray")

        g.render(filename, view=True)

class DARTSState:

    def __init__(self, state: Tuple[DARTSCell, DARTSCell]):
        self.state = state
        N_NODES = state[0].n_vertices
        self.ADJACENCY_MATRIX_SIZE = N_NODES**2
        self.N_OPERATIONS = 7
        self.zobrist_table = None

    def calculate_zobrist_hash(self, zobrist_table):
        assert zobrist_table is not None, "Remember to pass zobrist_table to node constructor."
        hash = 0
        adjacency = np.vstack([self.state[0].adjacency_matrix(), self.state[1].adjacency_matrix()]).flatten()
        for i, element in enumerate(adjacency):
            hash ^= zobrist_table[i][element]
        return hash

    def initialize_zobrist_table(self):
        self.zobrist_table = []
        for _ in range(2):  # Une fois pour la normal cell et une fois pour la reduction cell
            for i in range(ADJACENCY_MATRIX_SIZE):
                adjacency_table = []
                for operation in range(N_OPERATIONS):
                    adjacency_table.append(random.randint(0, 2 ** 64))
                self.zobrist_table.append(adjacency_table)

    def get_action_tuples(self):
        list_normal = self.state[0].get_action_tuples()
        list_normal = [(0, *e) for e in list_normal]
        list_reduction = self.state[1].get_action_tuples()
        list_reduction = [(1, *e) for e in list_reduction]

        return list(itertools.chain.from_iterable([list_normal, list_reduction]))

    def play_action(self, cell, start_vertice, end_vertice, operation):
        self.state[cell].play_action(start_vertice, end_vertice, operation)

    def is_complete(self):
        return all([s.is_complete() for s in self.state])

    def mutate(self, mutation_rate=1):
        Genotype = namedtuple('Genotype', 'normal normal_concat reduce reduce_concat')
        genotype_config = Genotype(
            normal=self.state[0].to_genotype(),
            normal_concat=[2, 3, 4, 5],
            reduce=self.state[1].to_genotype(),
            reduce_concat=[2, 3, 4, 5]
        )
        parent_compact = convert_genotype_to_compact(genotype_config)
        parent_compact = make_compact_mutable(parent_compact)
        compact = copy.deepcopy(parent_compact)
        while True:
            for _ in range(int(mutation_rate)):
                cell = np.random.choice(2)
                pair = np.random.choice(8)
                num = np.random.choice(2)
                if num == 1:
                    compact[cell][pair][num] = np.random.choice(self.N_OPERATIONS)
                else:
                    inputs = pair // 2 + 2
                    choice = np.random.choice(inputs)
                    if pair % 2 == 0 and compact[cell][pair + 1][num] != choice:
                        compact[cell][pair][num] = choice
                    elif pair % 2 != 0 and compact[cell][pair - 1][num] != choice:
                        compact[cell][pair][num] = choice

            if make_compact_immutable(parent_compact) != make_compact_immutable(compact):
                break
        new_genotype = convert_compact_to_genotype(compact)
        for i, cell_type in enumerate(["normal", "reduce"]):
            cell = eval("new_genotype." + cell_type)
            for node_idx in range(4):
                ops = cell[2*node_idx:2*node_idx+2]
                for k in range(len(self.state[i].vertices[node_idx+2].actions)):
                    for op in ops:
                        if k == op[1]:
                            self.state[i].vertices[node_idx+2].actions[k] = op[0]
                            break
                        self.state[i].vertices[node_idx + 2].actions[k] = "none"


    def get_reward(self, api, metric=Metric.VAL_ACCURACY, dataset="cifar10", df=None):
        normal_cell_genotype = self.state[0].to_genotype()
        reduction_cell_genotype = self.state[1].to_genotype()
        Genotype = namedtuple('Genotype', 'normal normal_concat reduce reduce_concat')
        genotype_config = Genotype(
                normal=normal_cell_genotype,
                normal_concat=[2, 3, 4, 5],
                reduce=reduction_cell_genotype,
                reduce_concat=[2, 3, 4, 5]
        )

        candidate = NasBench301SearchSpace()
        convert_genotype_to_naslib(genotype_config, candidate)
        reward = candidate.query(dataset=dataset, metric=metric, dataset_api=api)
        return reward

    def get_n_parameters(self, df=None):
        normal_cell_genotype = self.state[0].to_genotype()
        reduction_cell_genotype = self.state[1].to_genotype()
        Genotype = namedtuple('Genotype', 'normal normal_concat reduce reduce_concat')
        genotype_config = Genotype(
            normal=normal_cell_genotype,
            normal_concat=[2, 3, 4, 5],
            reduce=reduction_cell_genotype,
            reduce_concat=[2, 3, 4, 5]
        )

        candidate = NasBench301SearchSpace()
        convert_genotype_to_naslib(genotype_config, candidate)
        candidate.parse()
        reward = sum(p.numel() for p in candidate.parameters() if p.requires_grad)
        return reward

    def get_multiobjective_reward(self, api=None, metric="val_accuracy", dataset="cifar10", df=None):
        accuracy = self.get_reward(api, df=df)
        n_parameters = self.get_n_parameters(df=df)
        return accuracy, n_parameters

    def sample_random(self):
        while not self.is_complete():
            actions = self.get_action_tuples()
            random_action = random.choice(actions)
            self.play_action(*random_action)


class NASBench301Problem(ElementwiseProblem):

    def __init__(self):

        super().__init__(n_var=1,
                         n_obj=2,
                         vtype=np.object_)
        self.df = None
        self.api = get_dataset_api("nasbench301", "cifar10")

    def _evaluate(self, x, out, *args, **kwargs):
        # print(x_)
        # print(f"Starting evaluate")
        if isinstance(x, np.ndarray):
            x = x[0]
        out["F"] = x.get_multiobjective_reward(api=self.api, metric="val_accuracy", dataset="cifar10", df=self.df)
        # print(f"Evaluate over")

class NASBench301Sampling(Sampling):

    def _do(self, problem, n_samples, **kwargs):
        print(f"Starting sampling")
        """
        Generate random samples for the NASBench101 problem.
        """
        samples = []
        for _ in range(n_samples):
            cell = DARTSState((DARTSCell(), DARTSCell()))
            cell.initialize_zobrist_table()
            cell.sample_random()
            samples.append(cell)
        print(f"Sampling over")
        return np.array(samples, dtype=object)

class NASBench301Mutation(Mutation):

    def do(self, problem, pop, inplace=True, **kwargs):

        # if not inplace copy the population first
        if not inplace:
            pop = copy.deepcopy(pop)

        n_mut = len(pop)

        # get the variables to be mutated
        X = pop.get("X")

        # retrieve the mutation variables
        Xp = self._do(problem, X, **kwargs)

        # the likelihood for a mutation on the individuals
        prob = get(self.prob, size=n_mut)
        mut = np.random.random(size=n_mut) <= prob

        # store the mutated individual back to the population
        pop[mut].set("X", Xp[mut])

        return pop

    def _do(self, problem, x, **kwargs):
        """
        Perform mutation on the given sample.
        """
        # print(f"Starting mutation")
        # print(x)
        for i in range(len(x)):
            cell = x[i][0]
            cell.mutate(mutation_rate=1)
        # print(f"Mutation over")
        return x

class NASBench301Crossover(Crossover):

    def __init__(self, prob=0.5):
        super().__init__(n_parents=2, n_offsprings=2)
        self.prob = prob  # probability to inherit gene from parent 1



    def do(self, problem, pop, parents=None, **kwargs):
        # if a parents with array with mating indices is provided -> transform the input first
        if parents is not None:
            pop = [pop[mating] for mating in parents]

        # get the dimensions necessary to create in and output
        n_parents, n_offsprings = self.n_parents, self.n_offsprings
        n_matings, n_var = len(pop), problem.n_var

        # get the actual values from each of the parents
        X = np.swapaxes(np.array([[parent.get("X") for parent in mating] for mating in pop]), 0, 1)
        if self.vtype is not None:
            X = X.astype(self.vtype)

        # # the array where the offsprings will be stored to
        # Xp = np.empty(shape=(n_offsprings, n_matings, n_var), dtype=X.dtype)
        #
        # # the probability of executing the crossover
        # prob = get(self.prob, size=n_matings)
        #
        # # a boolean mask when crossover is actually executed
        # cross = np.random.random(n_matings) < prob
        # cross = np.zeros_like(n_matings)
        #
        # # the design space from the parents used for the crossover
        # if np.any(cross):
        #     # we can not prefilter for cross first, because there might be other variables using the same shape as X
        #     Q = self._do(problem, X, **kwargs)
        #     assert Q.shape == (n_offsprings, n_matings, problem.n_var), "Shape is incorrect of crossover impl."
        #     Xp[:, cross] = Q[:, cross]
        #     print(Q[:, cross])
        #     print(Xp)
        # X = np.expand_dims(X, -1)
        # # now set the parents whenever NO crossover has been applied
        # for k in np.flatnonzero(~cross):
        #     if n_offsprings < n_parents:
        #         s = np.random.choice(np.arange(self.n_parents), size=n_offsprings, replace=False)
        #     elif n_offsprings == n_parents:
        #         s = np.arange(n_parents)
        #     else:
        #         s = []
        #         while len(s) < n_offsprings:
        #             s.extend(np.random.permutation(n_parents))
        #         s = s[:n_offsprings]
        #
        #     Xp[:, k] = np.copy(X[s, k])
        #
        # # flatten the array to become a 2d-array
        # Xp = Xp.reshape(-1, X.shape[-1])
        # print(Xp)
        # create a population object
        off = Population.new("X", X)

        return off

class NASBench301Evaluator(Evaluator):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _eval(self, problem, pop, evaluate_values_of, **kwargs):

        # get the design space value from the individuals
        X = pop.get("X", to_numpy=False)

        # call the problem to evaluate the solutions
        out = problem.evaluate(X, return_values_of=evaluate_values_of, return_as_dictionary=True, **kwargs)

        # for each of the attributes set it to the problem
        for key, val in out.items():
            if val is not None:
                pop.set(key, val)

        # finally set all the attributes to be evaluated for all individuals
        pop.apply(lambda ind: ind.evaluated.update(out.keys()))

if __name__ == '__main__':

    # Example usage
    cell1 = DARTSCell()
    cell2 = DARTSCell()
    # api = get_dataset_api("nasbench301", "cifar10")
    state = DARTSState((cell1, cell2))
    state.initialize_zobrist_table()
    state.sample_random()
