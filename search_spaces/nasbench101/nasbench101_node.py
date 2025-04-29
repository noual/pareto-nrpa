import copy
import math
import random

import numpy as np
import pandas as pd
from nasbench import api as ModelSpecAPI
from nasbench.lib.graph_util import hash_module
from pymoo.core.crossover import Crossover
from pymoo.core.evaluator import Evaluator
from pymoo.core.mutation import Mutation
from pymoo.core.population import Population

from pymoo.core.problem import ElementwiseProblem

import sys

from pymoo.core.sampling import Sampling
from pymoo.core.variable import get

sys.path.append("../..")

from search_spaces.nasbench201.nasbench201_node import NASBench201Cell


class NASBench101Vertice:

    def __init__(self, id):
        self.id = id
        self.label = "none"
        self.edges = {i: 0 for i in range(id)}  # 0 ou 1 : connexion avec les autres vertices
        self.OPERATIONS = ["none", 'conv3x3-bn-relu', 'conv1x1-bn-relu', 'maxpool3x3', "input", "output"]
        self.playable_operations = ['conv3x3-bn-relu', 'conv1x1-bn-relu', 'maxpool3x3']  # Les labels qu'on peut assigner

    def get_action_tuples(self):
        list_tuples = []
        if self.label == "none":
            for op in self.playable_operations:
                list_tuples.append(("set_label", op))
        for k, v in self.edges.items():
            if v == 0:
                list_tuples.append(("build_edge", k))
        return list_tuples

    def play_action(self, action_name, action):
        if action_name == "set_label":
            self.label = action
        elif action_name == "build_edge":
            k = action
            self.edges[k] = 1


class NASBench101Cell(NASBench201Cell):

    def __init__(self, n_vertices, vertice_type=NASBench101Vertice):
        super().__init__(n_vertices, vertice_type)

        self.vertices[0].play_action("set_label", "input")
        self.vertices[1].play_action("build_edge", 0)
        self.vertices[n_vertices - 1].play_action("set_label", "output")
        self.vertices[n_vertices - 1].play_action("build_edge", n_vertices-2)
        self.N_NODES = 7
        self.ADJACENCY_MATRIX_SIZE = self.N_NODES ** 2
        self.N_OPERATIONS = 6
        self.playable_operations = ['conv3x3-bn-relu', 'conv1x1-bn-relu', 'maxpool3x3']  # Les labels qu'on peut assigner


    # def get_hash(self):
    #     return convert_spec_to_tuple({"matrix": self.adjacency_matrix(), "ops": [v.label for v in self.vertices]})

    def hash_cell(self):
        pruned_matrix, pruned_operations = self.prune()  # Getting the reduced rank matrix and operations
        # Getting the labeling used in naslib2/utils/nb101_api
        labeling = [-1] + [self.vertices[0].playable_operations.index(op) for op in pruned_operations[1:-1]] + [-2]
        hash = hash_module(pruned_matrix, labeling)
        return hash

    def adjacency_matrix(self):
        adjacency_matrix = np.zeros((self.n_vertices, self.n_vertices), dtype="int8")
        for i, vertice in enumerate(self.vertices):
            for j, connexion in vertice.edges.items():
                if connexion == 1:
                    adjacency_matrix[j, i] = 1
        return adjacency_matrix

    def numbered_adjacency(self):
        adjacency_matrix = np.zeros((self.n_vertices, self.n_vertices), dtype="int8")
        for i, vertice in enumerate(self.vertices):
            for j, connexion in vertice.edges.items():
                if connexion == 1:
                    adjacency_matrix[j, i] = vertice.OPERATIONS.index(vertice.label)
        return adjacency_matrix

    def operations_and_adjacency(self):
        adjacency = self.adjacency_matrix()
        operations = []
        for v in self.vertices:
            operations.append(v.label)

        return adjacency, operations

    def play_action(self, vertice, id, operation):

        self.vertices[vertice].play_action(id, operation)
        super().play_action(vertice, id,  operation)

    def get_action_tuples(self):
        sum_edges = 0
        for v in self.vertices:
            n_edges = int(np.sum(list(v.edges.values())))
            sum_edges += n_edges
        list_tuples = []
        for i, v in enumerate(self.vertices):
            actions = v.get_action_tuples()
            if sum_edges >= 9:
                actions_dup = []
                for act in actions:
                    if act[0] == "set_label":
                        actions_dup.append(act)
                actions = actions_dup
            for action in actions:
                list_tuples.append((i, *action))
        return list_tuples


    def is_complete(self):
        is_complete = True
        sum_edges = 0
        for v in self.vertices:
            n_edges = int(np.sum(list(v.edges.values())))
            sum_edges += n_edges
        if sum_edges > 9:
            is_complete = False
        for v in self.vertices:
            if v.label == "none":
                is_complete = False
        return is_complete

    def initialize_zobrist_table(self):
        self.zobrist_table = []
        for i in range(self.ADJACENCY_MATRIX_SIZE):
            adjacency_table = []
            for operation in range(self.N_OPERATIONS):
                adjacency_table.append(random.randint(0, 2 ** 64))
            self.zobrist_table.append(adjacency_table)
        for i, v in enumerate(self.vertices):
            adjacency_table = []
            for operation in v.OPERATIONS:
                adjacency_table.append(random.randint(0, 2 ** 64))
            self.zobrist_table.append(adjacency_table)

    def calculate_zobrist_hash(self, zobrist_table):
        hash = 0
        adjacency = self.adjacency_matrix()
        for i, element in enumerate(adjacency.flatten()):
            hash ^= zobrist_table[i][element]
        for i, v in enumerate(self.vertices):
            op_index = v.OPERATIONS.index(v.label)
            hash ^= zobrist_table[adjacency.shape[0]**2+i][op_index]
        return hash

    def get_n_parameters(self, df):
        if self.prune() is None:  # INVALID SPEC
            return 0
        arch_hash = self.hash_cell()
        row = df.loc[df["arch_hash"] == arch_hash]
        reward = row["params"].item()
        return reward

    def get_reward(self, api, metric="val_accuracy", dataset="cifar10", df=None):

        if df is not None:
            assert metric == "val_accuracy", "Only VAL_ACCURACY is supported for now."
            assert dataset == "cifar10", "Only CIFAR-10 is supported for now."
            if metric == "val_accuracy" and dataset == "cifar10":
                metric_to_fetch = "cifar_10_val_accuracy"
            if self.prune() is None:  # INVALID SPEC
                return 0
            arch_hash = self.hash_cell()
            row = df.loc[df["arch_hash"] == arch_hash]
            reward = row[metric_to_fetch].item()
            return reward

        assert metric.name in ["VAL_ACCURACY"], f"Only VAL_ACCURACY is supported, not {metric.name}"
        adjacency, operations = self.operations_and_adjacency()
        model_spec = ModelSpecAPI.ModelSpec(
                # Adjacency matrix of the module
                matrix=adjacency,
                # Operations at the vertices of the module, matches order of matrix
                ops=operations)
        if not model_spec.valid_spec:
            # INVALID SPEC
            return 0
        if metric.name == "VAL_ACCURACY":
            reward = api.query(model_spec)["validation_accuracy"] * 100
        return reward

    def get_multiobjective_reward(self, api=None, metric="val_accuracy", dataset="cifar10", df=None):
            """
            Fetch the multi-objective reward for the cell using NASLib.

            Parameters:
            api: The NASLib API instance.
            metric (Metric): The metric to be used for evaluation.
            dataset (str): The dataset to be used for evaluation.
            df (pd.DataFrame): The DataFrame containing architecture scores.

            Returns:
            tuple: A tuple containing the rewards for the cell.
            """
            if df is not None:

                assert metric == "val_accuracy", "Only val_accuracy is supported for now."
                assert dataset == "cifar10", "Only CIFAR-10 is supported for now."
                if metric == "val_accuracy" and dataset == "cifar10":
                    metric_to_fetch = "cifar_10_val_accuracy"
                if self.prune() is None:  # INVALID SPEC
                    return (-1e10, 1e10)
                adjacency, operations = self.operations_and_adjacency()
                model_spec = ModelSpecAPI.ModelSpec(
                    # Adjacency matrix of the module
                    matrix=adjacency,
                    # Operations at the vertices of the module, matches order of matrix
                    ops=operations)
                if not model_spec.valid_spec:
                    # INVALID SPEC
                    return (-1e10, 1e10)
                try:
                    arch_hash = self.hash_cell()
                    row = df.loc[df["arch_hash"] == arch_hash]
                    reward = row[metric_to_fetch].item()
                    n_params = row["params"].item()
                    return (reward, n_params)
                except Exception as e:
                    print("Error in hash_cell:", e)
                    return (-1e10, 1e10)
            else:
                raise NotImplementedError("Please provide a DataFrame with architecture scores.")

    def prune(self):
        """Prune the extraneous parts of the graph.

        General procedure:
          1) Remove parts of graph not connected to input.
          2) Remove parts of graph not connected to output.
          3) Reorder the vertices so that they are consecutive after steps 1 and 2.

        These 3 steps can be combined by deleting the rows and columns of the
        vertices that are not reachable from both the input and output (in reverse).
        """
        adjacency_matrix, ops = self.operations_and_adjacency()
        num_vertices = np.shape(adjacency_matrix)[0]

        # DFS forward from input
        visited_from_input = set([0])
        frontier = [0]
        while frontier:
            top = frontier.pop()
            for v in range(top + 1, num_vertices):
                if adjacency_matrix[top, v] and v not in visited_from_input:
                    visited_from_input.add(v)
                    frontier.append(v)

        # DFS backward from output
        visited_from_output = set([num_vertices - 1])
        frontier = [num_vertices - 1]
        while frontier:
            top = frontier.pop()
            for v in range(0, top):
                if adjacency_matrix[v, top] and v not in visited_from_output:
                    visited_from_output.add(v)
                    frontier.append(v)

        # Any vertex that isn't connected to both input and output is extraneous to
        # the computation graph.
        extraneous = set(range(num_vertices)).difference(
            visited_from_input.intersection(visited_from_output))

        # If the non-extraneous graph is less than 2 vertices, the input is not
        # connected to the output and the spec is invalid.
        if len(extraneous) > num_vertices - 2:
            matrix = None
            ops = None
            valid_spec = False
            return

        adjacency_matrix = np.delete(adjacency_matrix, list(extraneous), axis=0)
        adjacency_matrix = np.delete(adjacency_matrix, list(extraneous), axis=1)
        for index in sorted(extraneous, reverse=True):
            del ops[index]
        return adjacency_matrix, ops

    def mutate(self, api, mutation_rate=1):
        original_matrix, original_ops = copy.deepcopy(self.operations_and_adjacency())
        new_matrix, new_ops = copy.deepcopy(self.operations_and_adjacency())
        edge_mutation_probability = mutation_rate / self.N_NODES
        for src in range(0, self.N_NODES-1):
            for dst in range(src+1, self.N_NODES):
                if random.random() < edge_mutation_probability:
                    new_matrix[src, dst] = 1 - new_matrix[src, dst]

        operation_mutation_probability = mutation_rate / self.N_OPERATIONS
        for ind in range(0, self.N_NODES-1):
            if random.random() < operation_mutation_probability:
                available = [op for op in self.playable_operations if op != new_ops[ind]]
                new_ops[ind] = np.random.choice(self.vertices[0].playable_operations)
        new_spec = ModelSpecAPI.ModelSpec(
            # Adjacency matrix of the module
            matrix=new_matrix,
            # Operations at the vertices of the module, matches order of matrix
            ops=new_ops)

        for i, vertice in enumerate(self.vertices):
            for j, connexion in vertice.edges.items():
                vertice.edges[j] = new_matrix[j, i]
            vertice.label = new_ops[i]
        # original_matrix, original_ops = copy.deepcopy(self.operations_and_adjacency())
        # while True:
        #     new_matrix, new_ops = copy.deepcopy(self.operations_and_adjacency())
        #     edge_mutation_probability = mutation_rate / self.N_NODES
        #     for src in range(0, self.N_NODES-1):
        #         for dst in range(src+1, self.N_NODES):
        #             if random.random() < edge_mutation_probability:
        #                 new_matrix[src, dst] = 1 - new_matrix[src, dst]
        #
        #     operation_mutation_probability = mutation_rate / self.N_OPERATIONS
        #     for ind in range(0, self.N_NODES-1):
        #         if random.random() < operation_mutation_probability:
        #             available = [op for op in self.playable_operations if op != new_ops[ind]]
        #             new_ops[ind] = np.random.choice(self.vertices[0].playable_operations)
        #     new_spec = ModelSpecAPI.ModelSpec(
        #         # Adjacency matrix of the module
        #         matrix=new_matrix,
        #         # Operations at the vertices of the module, matches order of matrix
        #         ops=new_ops)
        #     print(api.is_valid(new_spec))
        #     if not api.is_valid(new_spec):
        #         print(new_matrix)
        #         print(new_ops)
        #     if api.is_valid(new_spec):
        #         break
        # for i, vertice in enumerate(self.vertices):
        #     for j, connexion in vertice.edges.items():
        #         vertice.edges[j] = new_matrix[j, i]
        #     vertice.label = new_ops[i]

    def from_adj_and_labels(self, child_adj, child_labels):
        """
        Create a new NASBench101Cell from adjacency matrix and labels.
        """
        cell = NASBench101Cell(n_vertices=self.n_vertices)
        for i, vertice in enumerate(self.vertices):
            for j, connexion in vertice.edges.items():
                vertice.edges[j] = child_adj[j, i]
            vertice.label = child_labels[i]
        return cell


class NASBench101Problem(ElementwiseProblem):

    def __init__(self):

        super().__init__(n_var=1,
                         n_obj=2,
                         vtype=np.object_)
        self.df = pd.read_csv("../data/nas/nasbench101.csv")
        self.api = ModelSpecAPI.NASBench("/home/lam/projets/nas_ntk/naslib/data/nasbench_full.tfrecord")

    def _evaluate(self, x, out, *args, **kwargs):
        # print(x_)
        print(f"Starting evaluate")
        if isinstance(x, np.ndarray):
            x = x[0]
        out["F"] = x.get_multiobjective_reward(api=None, metric="val_accuracy", dataset="cifar10", df=self.df)
        print(f"Evaluate over")

class NASBench101Sampling(Sampling):

    def _do(self, problem, n_samples, **kwargs):
        print(f"Starting sampling")
        """
        Generate random samples for the NASBench101 problem.
        """
        samples = []
        for _ in range(n_samples):
            cell = NASBench101Cell(n_vertices=7)
            cell.initialize_zobrist_table()
            while not cell.is_complete():
                actions = cell.get_action_tuples()
                action = random.choice(actions)
                cell.play_action(*action)
            adjacency, operations = cell.operations_and_adjacency()
            model_spec = ModelSpecAPI.ModelSpec(
                # Adjacency matrix of the module
                matrix=adjacency,
                # Operations at the vertices of the module, matches order of matrix
                ops=operations)
            is_valid = problem.api.is_valid(model_spec)

            samples.append(cell)
        print(f"Sampling over")
        return np.array(samples, dtype=object)

class NASBench101Mutation(Mutation):
    def _do(self, problem, x, **kwargs):
        """
        Perform mutation on the given sample.
        """
        print(f"Starting mutation")
        for i in range(len(x)):
            cell = x[i][0]
            cell.mutate(api=problem.api, mutation_rate=1)
        print(f"Mutation over")
        return x

class NASBench101Crossover(Crossover):
    def __init__(self, prob=0.5):
        super().__init__(n_parents=2, n_offsprings=2)
        self.prob = prob  # probability to inherit gene from parent 1

    def _do(self, problem, X, **kwargs):
        print(f"Starting crossover")
        n_parents, n_matings = X.shape
        Y = np.full((self.n_offsprings, n_matings, 1), None, dtype=object)

        for k in range(n_matings):
            parent1 = X[0, k]
            parent2 = X[1, k]

            # Generate two children
            child1 = self.crossover(parent1, parent2, problem)
            child2 = self.crossover(parent2, parent1, problem)

            # Assign them properly
            Y[0, k, 0] = child1
            Y[1, k, 0] = child2
        print(f"Crossover over")
        return Y

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

        # the array where the offsprings will be stored to
        Xp = np.empty(shape=(n_offsprings, n_matings, n_var), dtype=X.dtype)

        # the probability of executing the crossover
        prob = get(self.prob, size=n_matings)

        # a boolean mask when crossover is actually executed
        cross = np.random.random(n_matings) < prob

        # the design space from the parents used for the crossover
        if np.any(cross):
            # we can not prefilter for cross first, because there might be other variables using the same shape as X
            Q = self._do(problem, X, **kwargs)
            assert Q.shape == (n_offsprings, n_matings, problem.n_var), "Shape is incorrect of crossover impl."
            Xp[:, cross] = Q[:, cross]

        # now set the parents whenever NO crossover has been applied
        for k in np.flatnonzero(~cross):
            if n_offsprings < n_parents:
                s = np.random.choice(np.arange(self.n_parents), size=n_offsprings, replace=False)
            elif n_offsprings == n_parents:
                s = np.arange(n_parents)
            else:
                s = []
                while len(s) < n_offsprings:
                    s.extend(np.random.permutation(n_parents))
                s = s[:n_offsprings]

            Xp[:, k] = np.expand_dims(np.copy(X[s, k]), -1)

        # flatten the array to become a 2d-array
        Xp = Xp.reshape(-1, X.shape[-1])
        Xp = np.asarray(Xp, dtype=object)

        # create a population object
        off = Population.new("X", Xp)

        return off

    def crossover(self, p1, p2, problem):

        p1_adj, _ = p1.operations_and_adjacency()
        p2_adj, _ = p2.operations_and_adjacency()

        child_adj = np.zeros_like(p1_adj)
        child_labels = []

        for i in range(p1_adj.shape[0]):
            for j in range(p1_adj.shape[1]):
                if np.random.rand() < self.prob:
                    child_adj[i, j] = p1_adj[i, j]
                else:
                    child_adj[i, j] = p2_adj[i, j]

            # Vertex label
            if np.random.rand() < self.prob:
                child_labels.append(p1.vertices[i].label)
            else:
                child_labels.append(p2.vertices[i].label)
        cell = NASBench101Cell(7).from_adj_and_labels(child_adj, child_labels)


        return cell

class NASBench101Evaluator(Evaluator):
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

    def encode_graph(adj_matrix, node_labels, k=5):
        n = len(node_labels)
        log_k = math.ceil(np.log2(k))
        print(log_k)

        # Encode adjacency (upper triangular, excluding diagonal)
        adj_bits = []
        for i in range(n):
            for j in range(i + 1, n):
                adj_bits.append(adj_matrix[i, j])

        # Encode node labels
        label_bits = []
        for label in node_labels:
            if not (1 <= label <= k):
                raise ValueError("Labels must be in the range 1 to k inclusive.")
            bin_str = format(label, f'0{log_k}b')
            label_bits.extend([int(b) for b in bin_str])

        return np.array(adj_bits + label_bits, dtype=int)



    df = pd.read_csv("../../data/nas/nasbench101.csv")
    cell = NASBench101Cell(n_vertices=7)
    cell.initialize_zobrist_table()
    print(cell.calculate_zobrist_hash(cell.zobrist_table))
    # Complete cell
    while not cell.is_complete():
        actions = cell.get_action_tuples()
        action = random.choice(actions)
        print(action)
        cell.play_action(*action)

    reward = cell.get_multiobjective_reward(api=None, df=df)

    problem = NASBench101Problem()