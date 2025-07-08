import os
import random
import time

import numpy as np
from pymoo.core.problem import ElementwiseProblem
from pymoo.operators.sampling.rnd import PermutationRandomSampling

class TSPState:
    """
    Represents the state for the Traveling Salesman Problem with Time Windows (TSPTW).
    """

    def __init__(self,  n_cities, path = [0], visited={0}):
        """
        Initialize the TSPTW state.

        :param path: List of city IDs representing the current path, starting with the depot (0).
        :param current_time: Current time at the end of the path (arrival time at the last city).
        :param visited: Set of visited required cities (excluding the depot).
        :param cities_data: Dictionary mapping city IDs to their time windows and service times.
                            Format: {city_id: {'time_window': (e, l), 'service_time': s}, ...}
        :param travel_matrix: 2D list or dictionary of dictionaries containing travel times between cities.
                              Format: travel_matrix[from][to] = travel_time
        """
        self.path = path
        self.visited = visited.copy()
        self.travel_matrix = np.load(f"../../data/tsp/tsp_primary_cost_{n_cities}.npy")
        self.secondary_matrix = np.load(f"../../data/tsp/tsp_secondary_cost_{n_cities}.npy")
        self.zobrist_table = None

    def is_complete(self):
        """
        Check if the state represents a complete TSPTW solution.

        :return: True if the path starts and ends at the depot (0) and all required cities are visited.
        """
        # for (city, time) in self.visit_times:
        #     print(f"{CITIES[city]} visited at {time} ({self.cities_data[city]['time_window']})")
        #     if time > self.cities_data[city]['time_window'][1]:
        #         return True # Penalize late arrivals
        if len(self.path) < 2:
            return False
        if self.path[0] != 0 or self.path[-1] != 0:
            return False
        required_cities = set(range(self.travel_matrix.shape[0])) - {0}
        visited_in_path = set(self.path[1:-1])
        return visited_in_path == required_cities

    def get_action_tuples(self):
        """
        Generate valid actions (next cities to visit) from the current state.

        :return: List of valid actions as tuples. Each action is a tuple containing the next city ID.
        """
        actions = [None for _ in range(self.travel_matrix.shape[0])]
        actions = []
        current_city = self.path[-1]
        required_cities = set(range(self.travel_matrix.shape[0])) - {0}
        unvisited = required_cities - self.visited

        # Check if all required cities are visited and the last city is not the depot
        if not unvisited and current_city != 0:
                actions.append((0,))
                return actions

        for v in range(self.travel_matrix.shape[0]):
            if v in unvisited:
                    actions.append((v,))
        return actions

    def play_action(self, city):
        """
        Apply an action to the current state and return the new state.

        :param action: A tuple containing the next city ID to visit.
        :return: New TSPTWState instance after applying the action.
        """
        # print(f"Current time: {self.current_time}")
        j = city
        new_path = self.path.copy()
        new_path.append(j)

        new_visited = self.visited.copy()
        if j != 0:
            new_visited.add(j)

        self.path = new_path
        self.visited = new_visited


    def initialize_zobrist_table(self):
        self.zobrist_table = []
        for i in range(self.travel_matrix.shape[0]):
                self.zobrist_table.append(random.randint(0, 2 ** 32))

    def calculate_zobrist_hash(self, zobrist_table):
        """
        Calculate the Zobrist hash for the current state using a provided Zobrist table.

        :param zobrist_table: 2D list or dictionary where zobrist_table[from][to] is a random number.
        :return: Zobrist hash value.
        """
        h = 0
        for city in self.path:
            h ^= zobrist_table[city]
        return h

    def get_reward(self, api, metric, dataset, df):
        """
        Calculate the reward for the current state. Assumes the goal is to minimize total travel time.

        :return: Negative of the total travel time if the state is complete, otherwise 0.
        """
        assert self.is_complete(), "Path is not complete."
        n_violations = 0
        score = 0
        for i in range(len(self.path) - 1):
            from_city = self.path[i]
            to_city = self.path[i+1]
            score -= self.travel_matrix[from_city][to_city]

        # Reward is negative of the total time (to be maximized)
        # print(f"We have {n_violations} violations so the reward is {-self.current_time + 1e6*n_violations}.")
        return score - 1e6*n_violations

    def get_multiobjective_reward(self, api, metric, dataset, df):
        """
        Calculate the reward for the current state. Assumes the goal is to minimize total travel time.

        :return: Negative of the total travel time if the state is complete, otherwise 0.
        """
        assert self.is_complete(), "Path is not complete."
        n_violations = 0
        score = 0
        for i in range(len(self.path) - 1):
            from_city = self.path[i]
            to_city = self.path[i+1]
            score += self.travel_matrix[from_city][to_city]

        secondary_score = 0
        for i in range(len(self.path) - 1):
            from_city = self.path[i]
            to_city = self.path[i+1]
            secondary_score += self.secondary_matrix[from_city][to_city]
        # Reward is negative of the total time (to be maximized)
        # print(f"We have {n_violations} violations so the reward is {-self.current_time + 1e6*n_violations}.")

        return score, secondary_score

class TSPProblem(ElementwiseProblem):

    def __init__(self, n_cities):
        self.b = {}
        self.travel_matrix = np.load(f"../../data/tsp/tsp_primary_cost_{n_cities}.npy")
        self.secondary_matrix = np.load(f"../../data/tsp/tsp_secondary_cost_{n_cities}.npy")

        distances = self.travel_matrix
        max_ = np.max(distances)
        min_ = np.min(distances)

        for i in range(distances.shape[0]):
            for j in range(distances.shape[1]):
                self.b[(i, j)] = self.b.get((i, j), 0) - 10 * (distances[i, j] - min_) / (max_ - min_)
                self.b[(j, i)] = self.b.get((j, i), 0) - 10 * (distances[j, i] - min_) / (max_ - min_)

        distances = self.secondary_matrix
        max_ = np.max(distances)
        min_ = np.min(distances)

        for i in range(distances.shape[0]):
            for j in range(distances.shape[1]):
                self.b[(i, j)] = self.b.get((i, j), 0) - 10 * (distances[i, j] - min_) / (max_ - min_)
                self.b[(j, i)] = self.b.get((j, i), 0) - 10 * (distances[j, i] - min_) / (max_ - min_)
                # print(f"{i} -> {j}: {self.b[(i, j)]}")

        super().__init__(n_var=n_cities,
                         n_obj=2,
                         xl=0,
                         xu=n_cities-1,
                         vtype=int)


    def _evaluate(self, x, out, *args, **kwargs):
        x_ = np.zeros(x.shape[0]+1, dtype=int)
        x_[:-1] = x
        # print(x_)
        out["F"] = self.get_multiobjective_reward(x_)

    def get_multiobjective_reward(self, x):

        score = 0
        for i in range(len(x) - 1):
            from_city = x[i]
            to_city = x[i + 1]
            score += self.travel_matrix[from_city][to_city]

        secondary_score = 0
        for i in range(len(x) - 1):
            from_city = x[i]
            to_city = x[i + 1]
            secondary_score += self.secondary_matrix[from_city][to_city]
        # Reward is negative of the total time (to be maximized)
        # print(f"We have {n_violations} violations so the reward is {-self.current_time + 1e6*n_violations}.")
        reward = (score, secondary_score)
        # print(reward)
        return reward

if __name__ == '__main__':
    node = TSPState(n_cities=100)
    while not node.is_complete():
        actions = node.get_action_tuples()
        action = random.choice(actions)[0]
        node.play_action(action)
    reward = node.get_multiobjective_reward(None, None, None, None)
    print(reward)