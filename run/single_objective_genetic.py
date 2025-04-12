import numpy as np
import pandas as pd
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.core.problem import ElementwiseProblem
from pymoo.operators.crossover.ox import OrderCrossover
from pymoo.operators.mutation.inversion import InversionMutation
from pymoo.operators.sampling.rnd import PermutationRandomSampling
from pymoo.optimize import minimize

from search_algorithms.nsga2 import StartFromZeroRepair
from search_spaces.tsptw.tsptw_node import parse_file, TSPTWState


class SOOTSPTSWProblem(ElementwiseProblem):

    def __init__(self, file):
        matrix, data = parse_file(file)
        n_cities = matrix.shape[0]
        print(f"Loaded {n_cities} cities.")
        self.travel_matrix = matrix
        self.cities_data = data

        super().__init__(n_var=n_cities,
                         n_obj=1,
                         xl=0,
                         xu=n_cities-1,
                         vtype=int)


    def _evaluate(self, x, out, *args, **kwargs):
        x_ = np.zeros(x.shape[0]+1, dtype=int)
        x_[:-1] = x
        out["F"] = self.get_reward(x_)

    def get_reward(self, x):
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

        # Reward is negative of the total time (to be maximized)
        # print(f"We have {n_violations} violations so the reward is {-self.current_time + 1e6*n_violations}.")
        reward = score + 1e6 * n_violations
        return reward

    def calculate_visit_times(self, x):

        visit_times = [(0,0)]
        time = 0
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
    PROBLEM = "rc_204.3"
    problem = SOOTSPTSWProblem(file=f"../data/tsptw/SolomonTSPTW/{PROBLEM}.txt")
    results = []
    for i in range(30):
        algorithm = GA(
            pop_size=1000,
            n_offsprings=250,
            sampling=PermutationRandomSampling(),
            crossover=OrderCrossover(),
            mutation=InversionMutation(),
            eliminate_duplicates=True,
            repair=StartFromZeroRepair())

        res = minimize(problem,
                       algorithm,
                       ("n_eval", 40**3),
                       verbose=False)

        print(f"Best solution found: {res.X}")
        print(f"Best reward found: {res.F}")
        path = np.zeros(res.X.shape[0]+1, dtype=int)
        path[:-1] = res.X
        results.append(res.F.squeeze())
        print(path)
        print(f"Get reward2: {problem.get_reward2(path, PROBLEM)}" )
        print(results)
        print(np.mean(results))

    # Export results to CSV
    df = pd.DataFrame({'X': [res.X.tolist()], 'F': res.F, "problem": [PROBLEM]})
    df.to_csv('soo_ga_results.csv', index=False)

    print(f"Results exported to results.csv")
    print(np.mean(results))