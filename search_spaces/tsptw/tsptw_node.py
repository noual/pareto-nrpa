import io
import pickle
import random
import re

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from pymoo.core.problem import Problem, ElementwiseProblem
from pymoo.core.repair import Repair


def parse_file(path, kind="euclidean"):
    if kind == "euclidean":
        with open(path, 'r', encoding="utf-8") as f:
            s = f.read()
        n_cities, text = s.split("\n", 1)
        new_text = ""
        for line in text.split("\n"):
            new_text += re.sub("\s+", ";", line.strip()) + "\n"
        n_cities = int(n_cities)
        df = pd.read_csv(io.StringIO(new_text), sep=';', lineterminator='\n', header=None)
        # print(df)
        data = {}
        matrix = np.zeros((n_cities, n_cities))
        for i in range(n_cities):
            for j in range(n_cities):
                matrix[i, j] = np.sqrt((df.iloc[i, 1] - df.iloc[j, 1]) ** 2 + (df.iloc[i, 2] - df.iloc[j, 2]) ** 2) + df.iloc[i, 6]
        # print(matrix)
        for i in range(n_cities):
            data[i] = {'time_window': (int(df.iloc[i, 4]), int(df.iloc[i, 5])), 'service_time': int(0)}
        return matrix, data

    with open(path, 'r', encoding="utf-8") as f:
        lines = f.readlines()
    n_cities = int(lines[0])
    data = {}
    matrix = np.zeros((n_cities, n_cities))
    for k in range(n_cities):
        l = lines[k+1].rstrip()
        for i, val in enumerate(l.split(" ")):
            matrix[k, i] = float(val)
    for j in range(n_cities):
        l = lines[j+n_cities+1].rstrip()
        for i, char in enumerate(l):
            if char == " ":
                entry = l[:i]
                break
        l_reverse = l[::-1]
        print(l_reverse)
        for i, char in enumerate(l_reverse):
            if char == " ":
                out = l_reverse[:i][::-1]
                break
        print(f"City {j}: {entry}, {out}")
        data[j] = {'time_window': (int(entry), int(out)), 'service_time': int(0)}
    return matrix, data

def visualize(problem, x, fig=None, ax=None, show=True, label=True):
    with open(f"../data/tsptw/SecondaryCost/coordinates_{problem}.pickle", "rb") as f:
        sc = pickle.load(f)

    with open(f"../data/tsptw/SolomonTSPTW/{problem}.txt", 'r', encoding="utf-8") as f:
        s = f.read()
    n_cities, text = s.split("\n", 1)
    new_text = ""
    for line in text.split("\n"):
        new_text += re.sub("\s+", ";", line.strip()) + "\n"
    n_cities = int(n_cities)
    df = pd.read_csv(io.StringIO(new_text), sep=';', lineterminator='\n', header=None)
    with plt.style.context('ggplot'):

        if fig is None or ax is None:
            fig, ax = plt.subplots(1, 2, figsize=(12, 5))

        # plot cities using scatter plot
        ax[0].scatter(df[1], df[2], s=250)
        if label:
            # annotate cities
            for i in range(n_cities):
                ax[0].annotate(str(i), xy=(df[1].iloc[i], df[2].iloc[i]), fontsize=10, ha="center", va="center", color="white")

        # plot the line on the path
        for i in range(len(x)):
            current = x[i]
            next_ = x[(i + 1) % len(x)]
            ax[0].plot((df[1].iloc[current], df[1].iloc[next_]), (df[2].iloc[current], df[2].iloc[next_]), 'r--')

        # plot cities using scatter plot

        ax[1].scatter([x[0] for x in sc.values()], [x[1] for x in sc.values()], s=250, color="b")
        if label:
            # annotate cities
            for i in range(n_cities):

                ax[1].annotate(str(i), xy=(sc[i][0], sc[i][1]), fontsize=10, ha="center", va="center",
                               color="white")

        # plot the line on the path
        for i in range(len(x)):
            current = x[i]
            next_ = x[(i + 1) % len(x)]
            ax[1].plot((sc[current][0], sc[next_][0]), (sc[current][1], sc[next_][1]), 'b--')
        #
        # fig.suptitle("Route length: %.4f" % problem.get_route_length(x))
        ax[0].set_title("Primary cost")
        ax[1].set_title("Secondary cost")

        if show:
            plt.show()
            # fig.show()


class TSPTWState:
    """
    Represents the state for the Traveling Salesman Problem with Time Windows (TSPTW).
    """

    def __init__(self, path = [0], current_time=0, visited={0}, visit_times=[(0,0)], file=None, multiobjective=False):
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
        self.current_time = current_time
        self.visited = visited.copy()
        self.visit_times = visit_times.copy()
        if file is not None:
            m, d = parse_file(file)
            self.cities_data = d
            self.travel_matrix = m
        if multiobjective:
            file_root = file.split("/")[-1].split(".txt")[0]
            self.cost_matrix = np.load(f"../data/tsptw/SecondaryCost/{file_root}.npy")
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
        required_cities = set(self.cities_data.keys()) - {0}
        visited_in_path = set(self.path[1:-1])
        return visited_in_path == required_cities

    def get_action_tuples(self):
        """
        Generate valid actions (next cities to visit) from the current state.

        :return: List of valid actions as tuples. Each action is a tuple containing the next city ID.
        """
        actions = [None for _ in range(self.travel_matrix.shape[0])]
        nb = 0
        current_city = self.path[-1]
        required_cities = set(self.cities_data.keys()) - {0}
        unvisited = required_cities - self.visited
        # Check if all required cities are visited and the last city is not the depot
        if not unvisited and current_city != 0:
            return [(0,)]
        for i in range(self.travel_matrix.shape[0]):
            if i in unvisited:
                actions[nb] = (i,)
                if self.current_time + self.cities_data[current_city]['service_time'] + \
                        self.travel_matrix[current_city][i] > self.cities_data[i]['time_window'][1]:
                    nb += 1
        if nb == 0:
            for i in range(self.travel_matrix.shape[0]):
                if i in unvisited:
                    actions[nb] = (i,)
                    trop_tard = False
                    for j in range(1, self.travel_matrix.shape[0]):
                        if j != i:
                            if j in unvisited:
                                if (
                                        (self.current_time <= self.cities_data[j]['time_window'][1]) and
                                        (self.current_time + self.travel_matrix[current_city][j] +
                                         self.cities_data[current_city]['service_time'] <=
                                         self.cities_data[j]['time_window'][1]) and
                                        (max(self.current_time + self.travel_matrix[current_city][i] +
                                             self.cities_data[i]['service_time'],
                                             self.cities_data[i]['time_window'][0]) >
                                         self.cities_data[j]['time_window'][1])
                                ):
                                    trop_tard = True
                                    break
                    if not trop_tard:
                        nb += 1
        if nb == 0:
            # print(f"Len actions is 0")
            for v in range(self.travel_matrix.shape[0]):
                if v in unvisited:
                    actions[nb] = (v,)
                    nb += 1
        return actions[:nb]
        # actions = []
        # current_city = self.path[-1]
        # required_cities = set(self.cities_data.keys()) - {0}
        # unvisited = required_cities - self.visited
        # # print(f"Unvisited: {unvisited}")
        #
        # # Check if all required cities are visited and the last city is not the depot
        # if not unvisited and current_city != 0:
        #         actions.append((0,))
        #         return actions
        #
        # for v in range(self.travel_matrix.shape[0]):
        #     if v in unvisited:
        #         if self.current_time + self.cities_data[current_city]['service_time'] + self.travel_matrix[current_city][v] > self.cities_data[v]['time_window'][1]:
        #             # print(f"We add ")
        #             actions.append((v,))
        # # if len(actions) == 0:
        # #     for v in range(self.travel_matrix.shape[0]):
        # #         if v in unvisited:
        # #             impossible_move = False
        # #             for k in range(self.travel_matrix.shape[0]):
        # #                 if v in unvisited:
        # #                     if (
        # #                             (self.current_time <= self.cities_data[k]['time_window'][1]) and
        # #                             (self.current_time + self.travel_matrix[current_city][k] + self.cities_data[k]['service_time'] <= self.cities_data[k]['time_window'][1]) and
        # #                             ((self.current_time + self.travel_matrix[current_city][v] > self.cities_data[k]['time_window'][1]) or
        # #                              (self.cities_data[v]['time_window'][0] > self.cities_data[k]['time_window'][1]))
        # #                     ):
        # #                         impossible_move = True
        # #                         # print(f"{current_city} - > {v} IMPOSSIBLE MOVE")
        # #                         break
        # #             if not impossible_move:
        # #                 actions.append((v,))
        # if len(actions) == 0:
        #     # print(f"Len actions is 0")
        #     for v in range(self.travel_matrix.shape[0]):
        #         if v in unvisited:
        #             actions.append((v,))
        # return actions

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
        # print(f"New path: {new_path} = {[CITIES[i] for i in new_path]}")
        current_city = self.path[-1]
        # print("Current city:", CITIES[current_city], f"({current_city})")
        e_i, l_i = self.cities_data[j]['time_window']
        # print(f"e_i = {e_i}, l_i = {l_i}")
        s_i = self.cities_data[current_city]['service_time']
        departure_time_i = self.current_time + s_i
        # print(f"Departure time: {departure_time_i}")
        travel_time = self.travel_matrix[current_city][j]
        # print(f"Travel time: {travel_time}")
        arrival_time_j = departure_time_i + travel_time
        # print(f"Arrival time in city {CITIES[j]} ({j}): {arrival_time_j}")
        wait_time = max(0, e_i - arrival_time_j)
        # print(f"Wait time: {wait_time}")
        arrival_time_j += wait_time
        # print(f"We leave the city {CITIES[current_city]} ({current_city}) at {departure_time_i}h and arrive in city {CITIES[j]} ({j}) at {arrival_time_j - wait_time}h."
        #       f"The opening times for city {CITIES[j]} are {e_i}h to {l_i}h."
        #       f" We have to wait {wait_time}h before starting the service so the real arrival time is {arrival_time_j}h.")
        new_visited = self.visited.copy()
        if j != 0:
            new_visited.add(j)
        self.visit_times.append((j, arrival_time_j))
        self.path = new_path
        self.visited = new_visited
        self.current_time = arrival_time_j

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
        # print(f"Score: {score}, current time: {self.current_time}")
        for i, (city, time) in enumerate(self.visit_times):
            if time > self.cities_data[city]['time_window'][1]:
                n_violations +=1
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
            score -= self.travel_matrix[from_city][to_city]
        # print(f"Score: {score}, current time: {self.current_time}")
        for i, (city, time) in enumerate(self.visit_times):
            if time > self.cities_data[city]['time_window'][1]:
                n_violations +=1

        secondary_score = 0
        for i in range(len(self.path) - 1):
            from_city = self.path[i]
            to_city = self.path[i+1]
            secondary_score -= self.cost_matrix[from_city][to_city]
        # Reward is negative of the total time (to be maximized)
        # print(f"We have {n_violations} violations so the reward is {-self.current_time + 1e6*n_violations}.")
        return score - 1e6*n_violations, secondary_score - 1e6*n_violations

class TSPTSWProblem(ElementwiseProblem):

    def __init__(self, file):
        matrix, data = parse_file(file)
        n_cities = matrix.shape[0]
        self.travel_matrix = matrix
        self.cities_data = data
        file_root = file.split("/")[-1].split(".txt")[0]
        self.cost_matrix = np.load(f"../data/tsptw/SecondaryCost/{file_root}.npy")
        self.b = {}
        distances = self.travel_matrix
        max_ = np.max(distances)
        min_ = np.min(distances)
        for i in range(distances.shape[0]):
            for j in range(distances.shape[1]):
                self.b[(i, j)] = -10 * (distances[i, j] - min_) / (max_ - min_)
                print(f"{i} -> {j}: {self.b[(i, j)]}")

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
        # self.get_reward2(x)
        visit_times = self.calculate_visit_times(x)
        n_violations = 0
        score = 0
        for i in range(len(x) - 1):
            from_city = x[i]
            to_city = x[i + 1]
            score += self.travel_matrix[from_city][to_city]
        # print(f"Score: {score}, current time: {self.current_time}")
        for i, (city, time) in enumerate(visit_times):
            if time > self.cities_data[city]['time_window'][1]:
                n_violations += 1

        secondary_score = 0
        for i in range(len(x) - 1):
            from_city = x[i]
            to_city = x[i + 1]
            secondary_score += self.cost_matrix[from_city][to_city]
        # Reward is negative of the total time (to be maximized)
        # print(f"We have {n_violations} violations so the reward is {-self.current_time + 1e6*n_violations}.")
        reward = (score + 1e6 * n_violations, secondary_score + 1e6 * n_violations)
        # print(reward)
        return reward

    def calculate_visit_times(self, x):

        visit_times = [(0,0)]
        for i, city in enumerate(x[:-1]):
            city = x[i+1]
            previous_city = x[i]
            # print(f"Visiting city {previous_city} -> {city}.")
            # Get travel time from city to previous city
            travel_time = self.travel_matrix[previous_city][city]
            # Service_time
            service_time = self.cities_data[previous_city]["service_time"]
            # print(f"Travelling from {previous_city} to {city} in {travel_time} time units with {service_time} service.")
            # Calculate the time at which we leave city
            departure_time = visit_times[i][1] + service_time

            # Time at which the traveller arrives in the new city
            arrival_time = departure_time + travel_time

            # Do we have to wait for the city to open?
            e_i, l_i = self.cities_data[city]['time_window']
            wait_time = max(0, e_i - arrival_time)
            arrival_time += wait_time
            visit_times.append((city, arrival_time))
            # print(f"City {city} visited at {arrival_time}. The time window is {e_i} - {l_i}.")
        return visit_times


if __name__ == '__main__':
    path = "/home/lam/Téléchargements/SolomonTSPTW/rc_206.1.txt"
    parse_file(path)
