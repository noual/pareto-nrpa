import copy
import json
import time
import random

import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from pymoo.core.callback import Callback
from pymoo.indicators.hv import Hypervolume
from pymoo.visualization.scatter import Scatter
from pyrecorder.recorder import Recorder
from pyrecorder.writers.streamer import Streamer
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier

from node import Node
from search_algorithms.mcts_agent import MCTSAgent
from utils.helpers import configure_seaborn

configure_seaborn()
from pymoo.core.individual import Individual
from pymoo.core.population import Population
from pymoo.core.problem import Problem
from pymoo.core.result import Result
from pymoo.operators.survival.rank_and_crowding import RankAndCrowding
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from tqdm import tqdm

# from monet.node import Node
# from monet.search_algorithms.nested import NRPA
# from naslib.search_spaces.core import Metric


class PolicyManager:

    def __init__(self, alpha):
        self.policies = {}
        self.weights = {}
        self.alpha = alpha

    @property
    def n_policies(self):
        return len(self.policies.keys())

    def get_policy(self, index):
        assert index in self.policies.keys(), f"{index} is not one of the policies"
        return self.policies[index]

    def update_policy(self, index, pol):
        self.policies[index] = pol

    def delete_policy(self, index):
        assert index in self.policies.keys(), f"{index} is not one of the policies"
        self.policies.pop(index)

    def update_weights(self, optimal_set):
        """
        Updating the weights for the policies. The weight depends on the number of points on the pareto front and on the crowding distance of such points.
        """
        crowding = RankAndCrowding()
        distances = crowding.do(problem=Problem(n_constr=0), pop=optimal_set)
        dist = np.where(distances.get("crowding") == np.inf, 1, distances.get("crowding"))
        # print(dist)
        pol_w = {k: 1 for k in self.policies.keys()}
        counts = pol_w.copy()
        # print(optimal_set.get("F"))
        for p, c in zip(optimal_set.get("P"), dist):
            # print(f"[ADAPT] Point with policy {p} and crowding distance {c}")
            # pol_w[p] += c
            counts[p] += 1
        weights = {k: v/counts[k] for k, v in pol_w.items()}
        # print(f"Weights = {weights}")
        # print(f"Counts = {counts}")
        for pol in self.policies.keys():
            self.weights[pol] = weights[pol]

    # def compute_updates(self, optimal_set):
    #     variances = {}
    #     means = {}
    #     for pol_index, pol in self.policies.items():
    #         solutions = Population([e for e in optimal_set if e.get(
    #             "P") == pol_index])  # Toutes les solutions qui ont été obtenues en suivant la politique pol
    #         if len(solutions) <= 1:  # There are no Pareto-efficient solutions given by pol, so we up the variance
    #             variances[pol_index] = 0.11 * np.ones((optimal_set[0].get("X").shape))
    #         if len(solutions) == 0:  # There are no Pareto-efficient solutions given by pol, so we keep the mean the same
    #             means[pol_index] = pol[:, 0]
    #         else:
    #             variances[pol_index] = np.var(solutions.get("X"), axis=0)
    #             means[pol_index] = np.mean(solutions.get("X"), axis=0)
    #     return means, variances

    def copy(self):
        pm = self.__class__(alpha=self.alpha)
        pm.weights = self.weights.copy()
        for k, v in self.policies.items():
            pm.update_policy(k, v.copy())
        return pm

    def adapt(self, optimal_set, algorithm):
        """
        Adapt policies based on the optimal set.
        """
        crowding = RankAndCrowding()
        distances = crowding.do(problem=Problem(n_constr=0), pop=optimal_set)
        dist = np.where(distances.get("crowding") == np.inf, 2, distances.get("crowding"))
        for elem, dis in zip(optimal_set, dist):
            # print(f"-=-=-=-=-=-=-=UPDATE-=-=-=-=-=-=-=-")
            # print(f"Policy P{elem.get('P')} ({self.weights[elem.get('P')]:.2f}) -> Element: {elem.get('F''')}")
            policy_index = elem.get("P")
            sequence = elem.get("X")
            policy = self.policies[policy_index]
            policy.copy()
            node_type = type(algorithm.root)
            node = node_type(state=copy.deepcopy(algorithm.root.state), move=None, parent=None, sequence=[])
            node.hash = node.calculate_zobrist_hash(algorithm.root.state.zobrist_table)
            pol_prime = policy.copy()
            for i, action in enumerate(sequence):
                best_code = algorithm._code(node, action)
                pol_prime[best_code] = pol_prime.get(best_code, 0) + (self.alpha*dis)
                z = 0
                o = {}
                available_moves = node.get_action_tuples()
                move_codes = [algorithm._code(node, m) for m in available_moves]
                for move, move_code in zip(available_moves, move_codes):
                    # print(f"[Adapt] {node.state.path[i], move[0]}")
                    o[move_code] = np.exp(policy.get(move_code, 0) + algorithm.b[(node.state.path[i], move[0])])
                    z += o[move_code]
                for move, move_code in zip(available_moves, move_codes):
                    pol_prime[move_code] = pol_prime.get(move_code, 0) -  (self.alpha*dis) * (o[move_code] / z)

                node.play_action(action)
                node.hash = node.calculate_zobrist_hash(algorithm.root.state.zobrist_table)

            self.update_policy(policy_index, pol_prime)
        self.update_weights(optimal_set)

    def adapt_one(self, optimal_set, algorithm):
        """
        Adapt policies based on the optimal set.
        """
        sequences_to_optimize = {}
        distances_to_optimize = {}
        crowding = RankAndCrowding()
        distances = crowding.do(problem=Problem(n_constr=0), pop=optimal_set)
        dist = np.where(distances.get("crowding") == np.inf, 2, distances.get("crowding"))
        for policy in range(self.n_policies):
            indexes = np.argwhere(optimal_set.get("P") == policy)
            if len(indexes) == 0:
                continue
            max_crowding = np.argmax(dist[indexes])
            sequences_to_optimize[policy] = optimal_set[max_crowding]
            distances_to_optimize[policy] = dist[indexes][max_crowding].squeeze()
        for policy_index, elem in sequences_to_optimize.items():
            # print(f"Policy P{elem.get('P')} ({self.weights[elem.get('P')]:.2f}) -> Element {i}: {elem.get('F''')}")
            sequence = elem.get("X")
            policy = self.policies[policy_index]
            policy.copy()
            node_type = type(algorithm.root)
            node = node_type(state=copy.deepcopy(algorithm.root.state), move=None, parent=None, sequence=[])
            node.hash = node.calculate_zobrist_hash(algorithm.root.state.zobrist_table)
            pol_prime = policy.copy()
            for i, action in enumerate(sequence):
                best_code = algorithm._code(node, action)
                pol_prime[best_code] = pol_prime.get(best_code, 0) + (self.alpha * distances_to_optimize[policy_index])
                z = 0
                o = {}
                available_moves = node.get_action_tuples()
                move_codes = [algorithm._code(node, m) for m in available_moves]
                for move, move_code in zip(available_moves, move_codes):
                    # print(f"[Adapt] {node.state.path[i], move[0]}")
                    o[move_code] = np.exp(policy.get(move_code, 0) + algorithm.b[(node.state.path[i], move[0])])
                    z += o[move_code]
                for move, move_code in zip(available_moves, move_codes):
                    pol_prime[move_code] = pol_prime.get(move_code, 0) - (self.alpha * distances_to_optimize[policy_index]) * (o[move_code] / z)

                node.play_action(action)
                node.hash = node.calculate_zobrist_hash(algorithm.root.state.zobrist_table)

            self.update_policy(policy_index, pol_prime)
        self.update_weights(optimal_set)



class ParetoNRPA(MCTSAgent):

    def __init__(self, config):
        super().__init__(config)
        self.level = config.search.level
        self.alpha = config.search.nrpa_alpha
        self.softmax_temp = config.search.softmax_temp
        self.lr_update = config.search.nrpa_lr_update
        self.pm = PolicyManager(alpha=self.alpha)
        self.b = {}
        self.search_space = None
        self.max_pareto_set = config.search.n_policies
        self.n_iter = int(np.ceil(np.power(self.n_iter, 1 / self.level)))
        self.hypervolume_history = []
        self.advancement = 0
        self.anytime_pareto_set = Population()  # Used to keep track of the pareto set at any iteration for metric calculations
        self._initialize()
        self.nadir = (None, None)
        if config.callback:
            self.callback = MyCallback()

    def softmax_temp_fn(self, x, tau, **kwargs):
        if "b" in kwargs:
            b = kwargs["b"]
        else:
            b = np.zeros(x.shape)
        e_x = np.exp((x / tau) + b)
        return e_x / e_x.sum()

    def adapt_search_space(self, search_space, dataset):
        super().adapt_search_space(search_space, dataset)
        self.search_space = search_space
        if self.root.state.zobrist_table is None: self.root.state.initialize_zobrist_table()

    def _code(self, node, move):

        if self.search_space == "tsptw_moo":
            code = node.state.path[-1] * node.state.travel_matrix.shape[0] + move[0]
            return code

        if move == None:
            ### SEULEMENT POUR LA RACINE DE L'ARBRE A PRIORI
            return node.hash

        state_code = node.hash
        code = str(state_code)
        # code = ""  # J'enlève le hashage de zobrist pour le moment # justepourvoir
        for i in range(len(move)):
            code = code + str(move[i])

        return code

    def _initialize(self):
        # Set initial policy
        for i in range(self.max_pareto_set):
            self.pm.update_policy(i, {})
            self.pm.weights[i] = 1

    def _playout(self, node: Node, policy):
        node_type = type(node)
        playout_node = node_type(state=copy.deepcopy(node.state), move=copy.deepcopy(node.move),
                                 parent=copy.deepcopy(node.parent), sequence=copy.deepcopy(node.sequence))
        sequence = playout_node.sequence
        playout_node.hash = playout_node.calculate_zobrist_hash(self.root.state.zobrist_table)
        joint_proba = 1
        while not playout_node.is_terminal():

            # # Vérifier si la policy a une valeur pour ce noeud
            # if self._code(playout_node, playout_node.move) not in policy:
            #     policy[self._code(playout_node, playout_node.move)] = 0

            available_actions = playout_node.get_action_tuples()
            probabilities = []
            for move in available_actions:
                if self._code(playout_node, move) not in policy:
                    policy[self._code(playout_node, move)] = 0

            policy_values = [policy[self._code(playout_node, move)] for move in
                             available_actions]  # Calcule la probabilité de sélectionner chaque action avec la policy
            b = None
            if hasattr(self, "b"):  # Bias term for GNRPA
                # print([(playout_node.state.path[-1], move[0]) for move in available_actions])
                b = [self.b[(playout_node.state.path[-1], move[0])] for move in
                     available_actions]  #TODO: change to other than tsptw
                # print(b)
            # probas_raw = self.softmax_temp_fn(np.array(policy_values), self.softmax_temp, b=np.zeros_like(np.array(policy_values)))
            # print(f"Raw probabilities: {probas_raw}")
            probabilities = self.softmax_temp_fn(np.array(policy_values), self.softmax_temp, b=b)
            # print(f"Probabilities: {probabilities}")
            # if len(self.best_reward) % 100 == 0:
            #     pprint(list(zip(available_actions, pplayout_node# Used because available_actions is not 1-dimensional
            # print(available_actions)
            # print(probabilities)
            try:
                action_index = random.choices(np.arange(len(available_actions)), weights=probabilities)[0]
            except Exception:
                print(policy_values)
                raise Exception
            joint_proba *= probabilities[action_index]

            action = available_actions[action_index]  # Used because available_actions is not 1-dimensional

            sequence.append(action)
            # print(f"With p={probabilities[action_index]:.4f}, tp={joint_proba:.4f} : Sampling sequence {sequence}")
            playout_node.play_action(action)
            playout_node.hash = playout_node.calculate_zobrist_hash(self.root.state.zobrist_table)

        # compact = convert_genotype_to_compact(genotype_config)
        # print(f"With p={joint_proba:.4f} : Sampling {sequence}")
        reward = playout_node.get_multiobjective_reward(self.api, metric=None, dataset="cifar10", df=self.df)
        reward = (-reward[0], -reward[1])  # Minimizing both objectives
        # print(f"With p={joint_proba:.4f} : Sampling {sequence}")
        del playout_node
        return reward, sequence

    def next(self, index, policy_manager):
        """
        Used for level 0 NRPA, this respects the framework of PyMoo and represents a single function evaluation

        @param index: The index of the current policy that we are following
        """
        # Generate a new random solution within the problem bounds
        # print(f"Playout with policy {index}")
        reward, sequence = self._playout(self.root, policy=policy_manager.get_policy(index))
        new_individual = Individual()
        new_individual.X = sequence

        # Evaluate solution
        new_individual.F = reward

        self.advancement += 1

        return new_individual, sequence

    def nrpa(self, node, level, policy_manager, set_):
        if level == 0:

            # Choose a random policy and perform a playout
            p = np.random.choice(list(self.pm.policies.keys()),
                                 )#p=self.softmax_temp_fn(np.array(list(policy_manager.weights.values())), 1))

            new_individual, sequence = self.next(p, policy_manager)

            new_individual.set("P", p)
            self.anytime_pareto_set = Population.merge(self.anytime_pareto_set, new_individual)

            if self.advancement % 1000 == 0:
                """
                Video callback and hypervolume calculation
                """
                # Anytime Pareto set
                nds = NonDominatedSorting()
                fronts = nds.do(self.anytime_pareto_set.get("F"))
                indexes = []
                for f in fronts[0]:
                    el = self.anytime_pareto_set[f]
                    if el.get("F")[0] in [self.anytime_pareto_set[e].get("F")[0] for e in indexes]:
                        if el.get("F")[1] in [self.anytime_pareto_set[e].get("F")[1] for e in indexes]:
                            continue
                    indexes.append(f)
                self.anytime_pareto_set = self.anytime_pareto_set[indexes]
                approx_ideal = self.anytime_pareto_set.get("F").min(axis=0)
                approx_nadir = self.anytime_pareto_set.get("F").max(axis=0)
                print("nadir")
                print(self.nadir)
                metric = Hypervolume(ref_point=np.array(self.nadir),
                                     norm_ref_point=False,
                                     zero_to_one=False,
                                     ideal=approx_ideal,
                                     nadir=approx_nadir)

                hv = metric.do(self.anytime_pareto_set.get("F"))
                self.hypervolume_history.append(hv)
                self.callback(self, self.anytime_pareto_set)
            individual = Population.merge(Population(), new_individual)
                # print(f"[{len(self.rewards)}/{self.n_iter ** self.level}] Best reward: {max(self.best_reward)}")
            self.pbar.update(1)
            return individual

        else:
            optimal_set = Population()
            for i in range(self.n_iter):
                if level == 1:  # Avoid useless policy copy for playout level 0
                    result = self.nrpa(node, level - 1, policy_manager, optimal_set)
                else:
                    pm_copy = policy_manager.copy()
                    result = self.nrpa(node, level - 1, pm_copy, optimal_set)


                # if level >= 2:
                #     print(
                #         f"NRPA search level {level - 1} ({self.n_iter} iterations) produced the following optimal set ({len(result)}): ")
                #     print([[round(float(s), 2) for s in e.get("F")] for e in result])
                #     print(f"Our current optimal set ({len(optimal_set)}) is :")
                #     print([[round(float(s), 2) for s in e.get("F")] for e in optimal_set])
                optimal_set = Population.merge(optimal_set, result)
                # Non-dominated sorting
                nds = NonDominatedSorting()
                fronts = nds.do(optimal_set.get("F"))
                indexes = []
                for f in fronts[0]:
                    el = optimal_set[f]
                    if el.get("F")[0] in [optimal_set[e].get("F")[0] for e in indexes]:
                        if el.get("F")[1] in [optimal_set[e].get("F")[1] for e in indexes]:
                            continue
                    indexes.append(f)


                #region Modifying the optimal set so that each policy is represented
                osi = optimal_set[indexes]  # Optimal set incomplete
                for p in policy_manager.policies.keys():
                    if len(osi[osi.get("P") == p]) == 0:
                        # print(f"Policy {p} has no point in the pareto set.")
                        finished = False
                        j = 1
                        while not finished:
                            if j  == len(fronts):
                                break
                            for f in fronts[j]:
                                if optimal_set[f].get("P") == p:
                                    # print(f"Adding element from front {j}")
                                    indexes.append(f)
                                    finished = True
                                    break
                            j += 1
                #endregion

                optimal_set = optimal_set[indexes]

                policy_manager.adapt(optimal_set, self)

                # if level == self.level:
                #     self.callback(self, optimal_set)
            # if level == 1:
            #     print(optimal_set)
            return optimal_set

    def result(self, optimal_set):

        # Create a result object and store the final population data
        nds = NonDominatedSorting()
        fronts = nds.do(optimal_set.get("F"))
        optimal_set = optimal_set[fronts[0]]
        result = Result()
        result.X = optimal_set.get("X")
        result.F = optimal_set.get("F")
        result.P = optimal_set.get("P")
        return result

    def main_loop(self, app=None):
        node = self.root
        pol = {}
        print(self.n_iter, self.level)
        self.pbar = tqdm(total=self.n_iter ** self.level, position=0, leave=True)
        t1 = time.time()
        optimal_set = self.nrpa(node, self.level, self.pm, self.alpha)
        t2 = time.time()
        self.pbar.close()
        # print(f"Sequence is {sequence} with score {reward}")
        # for action in sequence:
        #     node.play_action(action)
        return self.result(optimal_set)

class ParetoNRPAPolicyRepresentation(ParetoNRPA):

    def __init__(self, config):
        super().__init__(config)

    def nrpa(self, node, level, policy_manager, set_):
        if level == 0:

            # Choose a random policy and perform a playout
            p = np.random.choice(list(self.pm.policies.keys()),
                                 p=self.softmax_temp_fn(np.array(list(policy_manager.weights.values())), 1))

            new_individual, sequence = self.next(p, policy_manager)

            new_individual.set("P", p)
            # Anytime Pareto set
            self.anytime_pareto_set = Population.merge(self.anytime_pareto_set, new_individual)
            nds = NonDominatedSorting()
            front = nds.do(self.anytime_pareto_set.get("F"), only_non_dominated_front=True)
            self.anytime_pareto_set = self.anytime_pareto_set[front]
            if self.advancement % 1000 == 0:
                """
                Video callback and hypervolume calculation
                """
                approx_ideal = self.anytime_pareto_set.get("F").min(axis=0)
                approx_nadir = self.anytime_pareto_set.get("F").max(axis=0)
                metric = Hypervolume(ref_point=np.array([1000, 1000]),
                                     norm_ref_point=False,
                                     zero_to_one=False,
                                     ideal=approx_ideal,
                                     nadir=approx_nadir)

                hv = metric.do(self.anytime_pareto_set.get("F"))
                self.hypervolume_history.append(hv)
                self.callback(self, self.anytime_pareto_set)
            individual = Population.merge(Population(), new_individual)
            # print(f"[{len(self.rewards)}/{self.n_iter ** self.level}] Best reward: {max(self.best_reward)}")
            self.pbar.update(1)
            return individual

        else:
            optimal_set = Population()
            for i in range(self.n_iter):
                if level == 1:  # Avoid useless policy copy for playout level 0
                    result = self.nrpa(node, level - 1, policy_manager, optimal_set)
                else:
                    pm_copy = policy_manager.copy()
                    result = self.nrpa(node, level - 1, pm_copy, optimal_set)

                # if level >= 2:
                #     print(
                #         f"NRPA search level {level - 1} ({self.n_iter} iterations) produced the following optimal set ({len(result)}): ")
                #     print([[round(float(s), 2) for s in e.get("F")] for e in result])
                #     print(f"Our current optimal set ({len(optimal_set)}) is :")
                #     print([[round(float(s), 2) for s in e.get("F")] for e in optimal_set])
                optimal_set = Population.merge(optimal_set, result)
                # Non-dominated sorting
                nds = NonDominatedSorting()
                fronts = nds.do(optimal_set.get("F"))
                indexes = []
                for f in fronts[0]:
                    el = optimal_set[f]
                    if el.get("F")[0] in [optimal_set[e].get("F")[0] for e in indexes]:
                        if el.get("F")[1] in [optimal_set[e].get("F")[1] for e in indexes]:
                            continue
                    indexes.append(f)

                # region Modifying the optimal set so that each policy is represented
                osi = optimal_set[indexes]  # Optimal set incomplete
                for p in policy_manager.policies.keys():
                    if len(osi[osi.get("P") == p]) == 0:
                        print(f"Policy {p} has no point in the pareto set.")
                        finished = False
                        j = 1
                        while not finished:
                            if j  == len(fronts):
                                break
                            for f in fronts[j]:
                                if optimal_set[f].get("P") == p:
                                    print(f"Adding element from front {j}")
                                    indexes.append(f)
                                    finished = True
                                    break
                            j += 1
                # endregion

                optimal_set = optimal_set[indexes]

                policy_manager.adapt(optimal_set, self)

                # if level == self.level:
                #     self.callback(self, optimal_set)
            # if level == 1:
            #     print(optimal_set)
            return optimal_set

class ParetoRandomSearch(ParetoNRPA):

    def __init__(self, config):
        super().__init__(config)

    def random_search(self, node):
        optimal_set = Population()
        j = 0
        for i in range(self.n_iter**self.level):
            new_individual, sequence = self.next(0, self.pm)
            new_individual.P = 0
            j += 1
            optimal_set = Population.merge(optimal_set, new_individual)
            if j % 100 == 0:
                nds = NonDominatedSorting()
                fronts = nds.do(optimal_set.get("F"))
                optimal_set = optimal_set[fronts[0]]

                approx_ideal = optimal_set.get("F").min(axis=0)
                approx_nadir = optimal_set.get("F").max(axis=0)
                metric = Hypervolume(ref_point=np.array([1000, 1000]),
                                     norm_ref_point=False,
                                     zero_to_one=False,
                                     ideal=approx_ideal,
                                     nadir=approx_nadir)

                hv = metric.do(optimal_set.get("F"))
                self.hypervolume_history.append(hv)
                self.callback(self, optimal_set)
            self.pbar.update(1)
        # Non-dominated sorting
        print(j)
        nds = NonDominatedSorting()
        fronts = nds.do(optimal_set.get("F"))
        optimal_set = optimal_set[fronts[0]]
        return optimal_set


    def main_loop(self, app=None):
        node = self.root
        pol = {}
        print(self.n_iter, self.level)
        self.pbar = tqdm(total=self.n_iter ** self.level, position=0, leave=True)
        t1 = time.time()
        optimal_set = self.random_search(node)
        t2 = time.time()
        self.pbar.close()
        # print(f"Sequence is {sequence} with score {reward}")
        # for action in sequence:
        #     node.play_action(action)
        return optimal_set

class MyCallback(Callback):

    def __init__(self) -> None:
        super().__init__()
        self.rec = Recorder(Streamer(sleep=0.1))

    def __call__(self, algorithm, population):
        if not self.is_initialized:
            self.initialize(algorithm)
            self.is_initialized = True

        self.notify(algorithm, population)
        self.update(algorithm)

    def notify(self, algorithm, population):
        scatter1 = Scatter("Iter", {'pad': 30}, legend=True)
        scatter1.set_axis_style(color="grey", alpha=0.5)
        for i in range(len(np.unique(population.get("P")))):
            pop = population.get("F")[population.get("P") == i]
            scatter1.add(pop, label=f"Policy {i}")
        scatter1.do()
        #self.rec.record()
