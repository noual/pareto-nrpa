import copy

import numpy as np
from pymoo.core.population import Population
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from tqdm import tqdm

from node import Node
from search_algorithms.mcts_agent import MCTSAgent


class MO_MCTS(MCTSAgent):

    def __init__(self, config):
        super().__init__(config)
        self.C = config.search.C
        self.N_tree_walks = config.search.N_tree_walks
        self.RAVE = {}

    def r_bar(self, node: Node, action):
        # Returns the average reward of the child node corresponding to the action

    def mo_mcts(self):
        T = self.root
        initial_state = copy.deepcopy(self.root)
        P = Population()
        for i in range(self.N_tree_walks):
            r_u = self.tree_walk(T, P, initial_state)
            P = Population.merge(P, r_u)
            # If r_u is not dominated by any element in P, remove all elements in P that are dominated by r_u and add r_u to P
            nds = NonDominatedSorting()
            fronts = nds.do(P.get("F"))
            P = P[fronts[0]]

    def tree_walk(self, T, P, s: Node):
        if not s.is_leaf():
            pass
        else:
            A_s = s.get_action_tuples()  # Admissible actions not yet visited in s
            print(f"Admissible actions: {A_s}")
            actions = [self.R(a, P) for a in A_s]
            a_star = A_s[np.argmin(actions)]
            child = Node(state=copy.deepcopy(s.state))
            child.play_action(a_star)
            s.children.append(child)
            r_u = self.random_walk(child)


    def R(self, a: tuple, P: Population):
        # Equation 4 in the paper
        elem1 = self.projection(self.RAVE[str(a)], P)
        elem2 = self.RAVE[str(a)]
        return np.linalg.norm(elem1 - elem2)

    def projection(self, a, P: Population):
        # The projection is the unique intersection of the line (a, nadir) with P.
        # The nadir point is the point with the maximum value of each objective in P.
        nadir = np.max(P.get("F"), axis=0)
        values_P = P.get("F")
        # Find the projection of a on the line (a, nadir)
        intersection = np.array([a[i] + (nadir[i] - a[i]) * (values_P[:, i] - a[i]) / (nadir[i] - a[i]) for i in range(len(a))]).T
        return intersection

    def random_walk(self, s: Node):
        A_rnd = []  # Set of actions visited in the random phase
        node_type = type(s)
        playout_node = node_type(state=copy.deepcopy(s.state))

        while not playout_node.is_terminal():
            available_actions = playout_node.get_action_tuples()
            random_action = available_actions[np.random.randint(len(available_actions))]
            A_rnd.append(random_action)
            playout_node.play_action(random_action)

        reward = playout_node.get_multiobjective_reward(None, None, None, None)

        # Update RAVE(a) for a in A_rnd
        for a in A_rnd:
            pass

        del playout_node
        return reward


    def main_loop(self):
        optimal_set = self.mo_mcts()

    # def _score_node(self, child: Node, parent: Node, C=None) -> float:
    #     # Returns UCB score for a child node
    #     if len(child.results) == 0:  # Si le noeud n'a pas encore été visité !
    #         return np.inf
    #     if C is None:
    #         C = self.C
    #
    #     mu_i = np.mean(child.results)
    #     # print(f"[UCB] : move : {child.move}, mu_i = {mu_i}, autre param: {C * (np.sqrt(np.log(len(parent.results)) / len(child.results)))}")
    #     return mu_i + C * (np.sqrt(np.log(len(parent.results)) / len(child.results)))
    #
    # def _selection(self, node: Node) -> Node:
    #     """
    #     Selects a candidate child node from the input node.
    #     """
    #     if not node.is_leaf():  # Tant que l'on a pas atteint une feuille de l'arbre
    #
    #         scores = [self._score_node(child, node, self.C) for child in node.get_children()]
    #         candidate_id = np.random.choice(np.flatnonzero(scores == np.max(scores)))  # Argmax with random tie-breaks
    #         candidate = node.get_children()[candidate_id]
    #
    #         return self._selection(candidate)
    #
    #     return node
    #
    # def _expansion(self, node: Node) -> Node:
    #     """
    #     Unless L ends the game decisively (e.g. win/loss/draw) for either player,
    #     create one (or more) child nodes and choose node C from one of them.
    #     Child nodes are any valid moves from the game position defined by L.
    #     """
    #     if not node.is_terminal():
    #         """
    #         Si le noeud n'a pas encore été exploré : on le retourne directement
    #         """
    #         if len(node.results) == 0 and node.parent is not None:
    #             return node
    #         node_type = type(node)
    #         node.children = [node_type(copy.deepcopy(node.state),
    #                                    move=m,
    #                                    parent=node)
    #                          for m in node.get_action_tuples()]
    #
    #         for child in node.children:
    #             child.play_action(child.move)
    #         returned_node = node.children[np.random.randint(0, len(node.children))]
    #         return returned_node
    #
    #     return node
    #
    # def _playout(self, node: Node):
    #     """
    #     Crée un playout aléatoire et renvoie l'accuracy sur le modèle entraîné
    #     :return:
    #     """
    #     node_type = type(node)
    #     playout_node = node_type(state=copy.deepcopy(node.state))
    #
    #     while not playout_node.is_terminal():
    #         available_actions = playout_node.get_action_tuples()
    #         random_action = available_actions[np.random.randint(len(available_actions))]
    #         playout_node.play_action(random_action)
    #
    #     reward = playout_node.get_multiobjective_reward(None, None, None, None)
    #
    #     del playout_node
    #     return reward
    #
    # def _backpropagation(self, node: Node, result: float):
    #     """
    #     Backpropagates the result of a playout up the tree.
    #     """
    #     if node.parent is None:
    #         node.results.append(result)
    #         return "Done"
    #     node.results.append(result)  # Ajouter le résultat à la liste
    #     return self._backpropagation(node.parent, result)  # Fonction récursive
    #
    # def next_best_move(self, all_rewards=None, best_reward=None) -> Node:
    #     """
    #     Body of UCT
    #     """
    #     best_reward_value = np.max(best_reward) if len(best_reward) > 0 else 0
    #     for i in tqdm(range(self.n_iter), disable=self.disable_tqdm):
    #
    #         leaf_node = self._selection(self.root)
    #         expanded_node = self._expansion(leaf_node)
    #
    #         for i_playout in range(self.playouts_per_selection):
    #             result = self._playout(expanded_node)
    #             _ = self._backpropagation(expanded_node, result)
    #             all_rewards.append(result)
    #             if result > best_reward_value:
    #                 best_reward_value = result
    #             best_reward.append(best_reward_value)
    #
    #     best_move_id = np.argmax([np.mean(child.results) for child in self.root.get_children()])
    #     best_move = self.root.get_children()[best_move_id]
    #     # print(f"[BODY] Selecting best move {best_move.move} with mean result {np.mean(best_move.results)}")
    #
    #     return best_move, all_rewards, best_reward
    #
    # def main_loop(self):
    #     """
    #     Corps de l'algorithme. Cherche le meilleur prochain coup jusqu'à avoir atteint un état terminal.
    #     :return: Le noeud représentant les meilleurs coups.
    #     """
    #     """Enregistrer les paramètres de la simul ation dans le folder"""
    #     # if self.save_folder is not None:
    #     #     shutil.copyfile(self.params_path, f"runs/{self.save_folder}/{self.__class__.__name__}-params.json")
    #     node = self.root
    #     self.all_rewards = []
    #     self.best_reward = []
    #     self.best_reward_value = 0
    #
    #     while not node.is_terminal():
    #         best_move, self.all_rewards, self.best_reward = self.next_best_move(self.all_rewards, self.best_reward)
    #         print(best_move.move)
    #
    #         node.play_action(best_move.move)
    #         root_type = type(self.root)
    #         self.root = best_move
    #         # print(len(best_move.children))
    #         # print([(len(e.results), np.mean(e.results)) for e in best_move.children])
    #         # self.root = root_type(copy.deepcopy(node.state))