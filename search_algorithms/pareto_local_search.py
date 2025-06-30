import json

import numpy as np
from opt_einsum.paths import optimal
from pymoo.core.individual import Individual
from pymoo.core.population import Population
from pymoo.core.result import Result
from pymoo.indicators.hv import Hypervolume
from pymoo.operators.mutation.inversion import InversionMutation
from pymoo.util.dominator import Dominator
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from pymoo.util.ref_dirs import get_reference_directions
from yacs.config import CfgNode

from search_algorithms.mcts_agent import MCTSAgent
from search_algorithms.nsga2 import PermutationRandomSamplingWithBias
from search_spaces.tsptw.tsptw_node import TSPTSWProblem, TSPProblem


class ParetoLocalSearch(MCTSAgent):

    def __init__(self, config):
        super().__init__(config)
        self.sampler = PermutationRandomSamplingWithBias()
        self.fn_evaluations = 0
        self.hypervolume_history = []


    def adapt_search_space(self, search_space, dataset):
        super().adapt_search_space(search_space, dataset)
        self.search_space = search_space
        if self.root.state.zobrist_table is None: self.root.state.initialize_zobrist_table()
        if search_space == "tsptw_moo":
            self.problem = TSPTSWProblem(file=f"../data/tsptw/SolomonTSPTW/{dataset}.txt")
            with open(f"../data/tsptw/SolomonTSPTW/nadirs.json", "r") as f:
                nadirs = json.load(f)
            self.nadir = nadirs[dataset]
        elif search_space == "tsp":
            self.problem = TSPProblem(file=f"../data/tsptw/SolomonTSPTW/{dataset}.txt")
            with open(f"../data/tsptw/SolomonTSPTW/nadirs.json", "r") as f:
                nadirs = json.load(f)
            self.nadir = nadirs[dataset]

    def initialize(self):
        optimal_set = Population()
        for i in range(100):
            new_individual = self.sampler.do(self.problem, 1)[0]
            f = self.problem.evaluate(new_individual.get("X"))
            self.fn_evaluations += 1
            new_individual.set("F", f)
            new_individual.set("V",  False)
            optimal_set = Population.merge(optimal_set, new_individual)
        return optimal_set
    
    def get_neighbors(self, s):
        assert self.search_space in ["tsp", "tsptw_moo"]
        n = len(s.X)
        neighborhood = Population()

        # Generate neighbors using swap moves
        for i in range(1, n - 1):
            for j in range(i + 1, n):
                neighbor = s.X.copy()
                neighbor[i], neighbor[j] = neighbor[j], neighbor[i]
                ind = Individual(X=neighbor)
                if np.any(np.all(ind.get("X") == self.visited_solutions.get("X"), axis=1)):
                    continue
                f = self.problem.evaluate(ind.get("X"))
                self.fn_evaluations += 1
                ind.set("F", f)
                neighborhood = Population.merge(neighborhood, ind)

        # Generate neighbors using inversion moves
        for i in range(1, n - 1):
            for j in range(i + 1, n):
                neighbor = s.X.copy()
                neighbor[i:j + 1] = neighbor[i:j + 1][::-1]
                ind = Individual(X=neighbor)
                f = self.problem.evaluate(ind.get("X"), ind)
                self.fn_evaluations += 1
                ind.set("F", f)
                neighborhood = Population.merge(neighborhood, ind)

        return neighborhood

    def pls(self, optimal_set):
        k = 0
        self.visited_solutions = optimal_set.copy()
        optimal_prime = optimal_set.copy()
        while True:
            index = np.random.randint(len(optimal_prime))
            s = optimal_prime[index]
            # print(f"Selecting randomly solution {index}")
            neighborhood = self.get_neighbors(s)

            for neighbor in neighborhood:

                # print(f"Exploring Neighbor: {neighbor.get('F')}")
                # if not np.any(np.all(neighbor.get("X") == self.visited_solutions.get("X"), axis=1)):
                #     self.visited_solutions = Population.merge(self.visited_solutions, neighbor)

                if neighbor.get("F")[0] <= s.get("F")[0] or neighbor.get("F")[1] <= s.get("F")[1]:
                    # print(f"Neighbor is better: we add it to optimal")
                    neighbor.set("V", False)
                    # print(neighbor.get("X"))
                    # print(optimal_prime.get("X"))
                    if not np.any(np.all(neighbor.get("X") == optimal_prime.get("X"), axis=1)):
                        optimal_prime = Population.merge(optimal_prime, neighbor)
                    nds = NonDominatedSorting()
                    fronts = nds.do(optimal_prime.get("F"))
                    optimal_prime = optimal_prime[fronts[0]]
                    # print(optimal_prime.get("F"))
                    # print(f"New solution found {neighbor.get('F')}")

            s.set("V", True)

            optimal_set = optimal_prime[optimal_prime.get("V") == False]
            # print(f"The number of solutions still to visit: {len(optimal_set)}")
            # print(len(optimal_set))

            if len(optimal_set) == 0:
                break

            if self.fn_evaluations > 1000*k:
                """
                Video callback and hypervolume calculation
                """
                k += 1
                print(f"n_evaluations: {self.fn_evaluations}, k: {k}")
                approx_ideal = optimal_prime.get("F").min(axis=0)
                approx_nadir = optimal_prime.get("F").max(axis=0)
                # print("nadir")
                # print(self.nadir)
                metric = Hypervolume(ref_point=np.array(self.nadir),
                                     norm_ref_point=False,
                                     zero_to_one=False,
                                     ideal=approx_ideal,
                                     nadir=approx_nadir)
                # print(optimal_prime.get("F"))
                hv = metric.do(optimal_prime.get("F"))
                self.hypervolume_history.append(hv)

            if self.fn_evaluations >= self.n_iter:
                break

        return optimal_prime

    def result(self, optimal_set):
        print(f"FN EVALUATIONS: {self.fn_evaluations}")
        # Create a result object and store the final population data
        nds = NonDominatedSorting()
        fronts = nds.do(optimal_set.get("F"))
        optimal_set = optimal_set[fronts[0]]
        result = Result()
        if self.search_space == "nasbench101":
            result.X = np.ones_like(optimal_set.get("F"))
        else:
            result.X = optimal_set.get("X")
        result.F = optimal_set.get("F")
        result.P = optimal_set.get("P")
        return result

    def main_loop(self, app=None):
        optimal_set = self.initialize()
        result = self.pls(optimal_set)
        print(result.get("F"))
        return self.result(result)

class MultiRestartParetoLocalSearch(ParetoLocalSearch):
    """
    Multi-restart version of Pareto Local Search
    """
    def __init__(self, config):
        super().__init__(config)
        self.n_restarts = config.search.n_restarts
        self.n_iter = config.search.n_iter // self.n_restarts


    def deactivate(self, s: Individual, A: Population):
        """
        Algorithm 2 in Drugan paper.
        :param s:
        :param A:
        :return:
        """
        A_i = Population([s])
        dominator = Dominator()
        for s_prime in A:
            # print(f"Comparing {s.get('F')} with {s_prime.get('F')}")
            relation = dominator.get_relation(s.get("F"), s_prime.get("F"))
            if relation == 0:
                A_i = Population.merge(A_i, s_prime)
        return A_i

    def mpls(self, optimal_set):
        A = optimal_set
        k = 0
        while True:
            k += 1
            self.fn_evaluations = 0
            print(f"{k} / {self.n_restarts}, size of A: {len(A)}")
            s = self.sampler.do(self.problem, 1)[0]
            f = self.problem.evaluate(s.get("X"), s)
            self.fn_evaluations += 1
            s.set("F", f)
            A_prime = self.deactivate(s, A)
            # print(A_prime.get("F"))
            result = self.pls(A_prime)
            A = Population.merge(A, result)
            nds = NonDominatedSorting()
            fronts = nds.do(A.get("F"))
            indexes = []
            for f in fronts[0]:
                el = A[f]
                if el.get("F")[0] in [A[e].get("F")[0] for e in indexes]:
                    if el.get("F")[1] in [A[e].get("F")[1] for e in indexes]:
                        continue
                indexes.append(f)
            A = A[indexes]

            if k >= self.n_restarts:
                break
        return A


    def main_loop(self, app=None):
        optimal_set = self.initialize()
        result = self.mpls(optimal_set)
        print(result.get("F"))
        return self.result(result)



if __name__ == '__main__':
    config = CfgNode({"df_path": "none",
        "search": {"n_iter": 1000000, "population_size": 250, "sample_size": 25, "playouts_per_selection": 1, "n_restarts": 100},
        "disable_tqdm": "false", "seed": 0})
    pls = MultiRestartParetoLocalSearch(config)
    pls.adapt_search_space("tsptw_moo", "rc_205.3")
    r = pls.main_loop()
