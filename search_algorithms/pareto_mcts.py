import copy
import time
from typing import Tuple, Any

import numpy as np
from pymoo.core.individual import Individual
from pymoo.core.population import Population
from pymoo.core.result import Result
from pymoo.indicators.hv import Hypervolume
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from scipy.special import softmax
from tqdm import tqdm

from node import Node
from search_algorithms.mcts_agent import MCTSAgent
from search_algorithms.pareto_nrpa.pareto_nrpa import MyCallback


class Pareto_UCT(MCTSAgent):

    def __init__(self, config):
        super().__init__(config)
        self.advancement = 0
        self.hypervolume_history = []
        self.C = config.search.C
        self.optimal_set = Population()
        if config.callback:
            self.callback = MyCallback()

    def _score_node(self, child: Node, parent: Node, C=None):
        # Returns UCB score for a child node
        if len(child.results) == 0:  # Si le noeud n'a pas encore été visité !
            return -np.inf, -np.inf  #TODO:     adapt to n dimensions
        if C is None:
            C = self.C

        mu_i = np.mean(np.clip(child.results, 0, 1000), axis=0)

        result = mu_i - C * (np.sqrt(
                                    (4*np.log(len(parent.results) + np.log(2))) / (1*len(child.results))
                                    )
                            )
        # if parent == self.root:
        #     print("C = ", C)
        #     print(C * (np.sqrt(
        #         (4 * np.log(len(parent.results) + np.log(2))) / (2 * len(child.results))
        #     )
        #     ))
        #     print(f"{child.state.path} Mui = {mu_i}, const = {result-mu_i}")
        return tuple(result)

    def _selection(self, node: Node) -> Node:
        """
        Selects a candidate child node from the input node.
        """
        if not node.is_leaf():  # Tant que l'on a pas atteint une feuille de l'arbre
            # print(f"[SELECTION] Node with path {node.state.path}")
            children = node.get_children()
            population = Population()
            # print(f"[SELECTION] results : {[np.mean(c.results) for c in node.get_children()]}")
            # print(f"Mean children: {[np.clip(np.nan_to_num(np.mean(c.results)), 0, 1000) for c in node.get_children()]}")
            max_value = np.max([np.nan_to_num(np.mean(np.clip(c.results, 0, 1000))) for c in node.get_children()])
            min_value = np.min([np.nan_to_num(np.mean(np.clip(c.results, 0, 1000))) for c in node.get_children()])
            # print(f'Max value is {max_value}, min value is {min_value}')
            C_dif = max_value - min_value
            # print(f"[SELECTION] C_dif = {C_dif}")
            C = max([self.C, 2 * C_dif])
            # C = self.C
            for child in children:
                individual = Individual()
                individual.X = child
                individual.F = self._score_node(child, node, C)
                population = Population.merge(population, individual)

            nds = NonDominatedSorting()
            fronts = nds.do(population.get("F"))
            # for f in fronts[0]:
                # print(f"elem {population[f].X.state.path}: {population[f].F}")
            individual = population[fronts[0][np.random.randint(0, len(fronts[0]))]]
            # print(f"{population.get("F")}")
            # if node == self.root:
            #     print(f"[SELECTION] selecting {individual.X.state.path} with UCB score {individual.F} and C={C} and results {np.mean(individual.X.results)}")
            return self._selection(individual.X)
        return node

    def _expansion(self, node: Node) -> Node:
        """
        Unless L ends the game decisively (e.g. win/loss/draw) for either player,
        create one (or more) child nodes and choose node C from one of them.
        Child nodes are any valid moves from the game position defined by L.
        """
        if not node.is_terminal():
            """
            Si le noeud n'a pas encore été exploré : on le retourne directement
            """
            if len(node.results) == 0 and node.parent is not None:
                return node
            node_type = type(node)
            node.children = [node_type(copy.deepcopy(node.state),
                                       move=m,
                                       parent=node,
                                       sequence=node.sequence + [m])
                             for m in node.get_action_tuples()]

            for child in node.children:
                child.play_action(child.move)
            returned_node = node.children[np.random.randint(0, len(node.children))]
            return returned_node

        return node

    def softmax_temp_fn(self, x, tau, **kwargs):
        if "b" in kwargs:
            b = kwargs["b"]
        else:
            b = np.zeros(x.shape)
        e_x = np.exp((x / tau) + b)
        return e_x / e_x.sum()

    def _playout(self, node: Node):
        """
        Crée un playout aléatoire et renvoie l'accuracy sur le modèle entraîné
        :return:
        """
        node_type = type(node)
        playout_node = node_type(state=copy.deepcopy(node.state))
        sequence = copy.deepcopy(node.sequence)
        while not playout_node.is_terminal():
            available_actions = playout_node.get_action_tuples()
            # Get probabilities including bias
            probabilities = self.softmax_temp_fn(x = np.ones(len(available_actions)),
                                                 tau = 1,
                                                 b= [self.b[sequence[-1][0], a[0]] for a in available_actions])
            random_action = available_actions[np.random.choice(len(available_actions), p=probabilities)]
            playout_node.play_action(random_action)
        # print(f"We sample {playout_node.state.path}")
        # print(f"---")
        reward = playout_node.get_multiobjective_reward(None, None, None, None)
        reward = (-reward[0], -reward[1])  # Minimizing both objectives
        individual = Individual()
        individual.X = playout_node.state.path
        individual.F = reward
        individual.set("P", 0)
        self.advancement += 1
        self.optimal_set = Population.merge(self.optimal_set, individual)
        del playout_node
        return reward

    def _backpropagation(self, node: Node, result: float):
        """
        Backpropagates the result of a playout up the tree.
        """
        if node.parent is None:
            node.results.append(result)
            return "Done"
        node.results.append(result)  # Ajouter le résultat à la liste
        return self._backpropagation(node.parent, result)  # Fonction récursive


    def main_loop(self):
        """
        Corps de l'algorithme. Cherche le meilleur prochain coup jusqu'à avoir atteint un état terminal.
        :return: Le noeud représentant les meilleurs coups.
        """
        """Enregistrer les paramètres de la simulation dans le folder"""
        node = self.root
        print(self.n_iter)
        for i in tqdm(range(self.n_iter)):
            t0 = time.time()
            leaf_node = self._selection(self.root)
            t1 = time.time()
            expanded_node = self._expansion(leaf_node)
            t2 = time.time()
            for i_playout in range(self.playouts_per_selection):
                result = self._playout(expanded_node)
                t3 = time.time()
                _ = self._backpropagation(expanded_node, result)
                t4 = time.time()

            # Anytime Pareto set

            if self.advancement % 1000 == 0:
                """
                Video callback and hypervolume calculation
                """
                anytime_pareto_set = Population()
                nds = NonDominatedSorting()
                front = nds.do(self.optimal_set.get("F"), only_non_dominated_front=True)
                indexes = []
                for f in front:
                    el = self.optimal_set[f]
                    if el.get("F")[0] in [self.optimal_set[e].get("F")[0] for e in indexes]:
                        if el.get("F")[1] in [self.optimal_set[e].get("F")[1] for e in indexes]:
                            continue
                    indexes.append(f)
                anytime_pareto_set = self.optimal_set[indexes]
                print(f"Front: {anytime_pareto_set.get("F")}")
                self.optimal_set = anytime_pareto_set
                approx_ideal = anytime_pareto_set.get("F").min(axis=0)
                approx_nadir = anytime_pareto_set.get("F").max(axis=0)
                metric = Hypervolume(ref_point=np.array(self.nadir),
                                     norm_ref_point=False,
                                     zero_to_one=False,
                                     ideal=approx_ideal,
                                     nadir=approx_nadir)

                hv = metric.do(anytime_pareto_set.get("F"))
                self.hypervolume_history.append(hv)
                self.callback(self, anytime_pareto_set)
            tt = time.time()
            # print(f"Selection took {(t1 - t0):.3f} seconds, {100*(t1-t0)/(tt-t0):.3f}% of total time")
            # print(f"Expansion took {(t2 - t1):.3f} seconds, {100*(t2-t1)/(tt-t0):.3f}% of total time")
            # print(f"Playout took {(t3 - t2):.3f} seconds, {100*(t3-t2)/(tt-t0):.3f}% of total time")
            # print(f"Backpropagation took {(t4 - t3):.3f} seconds, {100*(t4-t3)/(tt-t0):.3f}% of total time")


        return self.result(self.optimal_set)

        # # Return most visited child
        # node = self.root
        # while len(node.children) > 0:
        #     node = node.get_children()[np.argmax([len(child.results) for child in node.get_children()])]
        # return node

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