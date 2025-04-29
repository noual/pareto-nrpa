import json

import numpy as np
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.moo.sms import SMSEMOA
from pymoo.core.mutation import Mutation
from pymoo.core.population import Population
from pymoo.core.repair import Repair
from pymoo.core.sampling import Sampling
from pymoo.core.variable import get, Real
from pymoo.indicators.hv import Hypervolume
from pymoo.operators.crossover.ox import OrderCrossover
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.inversion import InversionMutation
from pymoo.operators.mutation.pm import PM, mut_pm
from pymoo.operators.repair.rounding import RoundingRepair
from pymoo.operators.sampling.rnd import IntegerRandomSampling, PermutationRandomSampling
from pymoo.termination import get_termination
from pymoo.visualization.scatter import Scatter
from pymoo.core.callback import Callback
from pymoo.optimize import minimize
from pyrecorder.recorder import Recorder
from pyrecorder.writers.streamer import Streamer

from search_spaces.nasbench101.nasbench101_node import NASBench101Problem, NASBench101Sampling, NASBench101Mutation, \
    NASBench101Crossover, NASBench101Evaluator
from search_spaces.nasbench201.nasbench201_node import NASBench201Problem
from search_spaces.nasbench301.nasbench301_node import NASBench301Problem, NASBench301Sampling, NASBench301Mutation, \
    NASBench301Evaluator, NASBench301Crossover
from search_spaces.radar.radar_node import RadarProblem
from search_spaces.tsptw.tsptw_node import TSPTSWProblem


class SMSEMOAAlgorithm:

    def __init__(self, config):
        self.callback = MyCallback()

        self.algorithm = SMSEMOA(
            pop_size=config.search.population_size,
            sampling=PermutationRandomSamplingWithBias(),
            crossover=OrderCrossover(),
            mutation=InversionMutation(),
            eliminate_duplicates=True,
            repair=StartFromZeroRepair(),
            callback=self.callback,
            save_history=True,
        )

        self.callback.initialize(self.algorithm)
        self.algorithm.hypervolume_history = []
        self.termination = get_termination("n_eval", config.search.n_iter)


    def adapt_search_space(self, search_space, dataset):
        supported_ss = ["tsptw_moo", "radar", "nasbench201", "nasbench101", "nasbench301"]

        assert search_space in supported_ss, f"Search space {search_space} not supported. Supported search spaces: {supported_ss}"
        if search_space == "tsptw_moo":
            self.problem = TSPTSWProblem(file=f"../data/tsptw/SolomonTSPTW/{dataset}.txt")
            with open(f"../data/tsptw/SolomonTSPTW/nadirs.json", "r") as f:
                nadirs = json.load(f)
            self.nadir = nadirs[dataset]
            self.algorithm.nadir= self.nadir

        elif search_space == "nasbench201":
            self.problem = NASBench201Problem()
            self.algorithm = SMSEMOA(
                pop_size=250,
                n_offsprings=25,
                sampling=IntegerRandomSampling(),
                crossover=SBX(eta=20),
                mutation=PolynomialMutation(eta=20),
                eliminate_duplicates=True,
                repair=RoundingRepair(),
                callback=self.callback,
                save_history=True,
            )
            self.callback.initialize(self.algorithm)
            self.nadir = (100, 1531556)  # worst accuracy and biggest number of params
            self.algorithm.nadir = self.nadir

        elif search_space == "nasbench101":
            self.problem = NASBench101Problem()
            self.algorithm = SMSEMOA(
                pop_size=250,
                n_offsprings=25,
                sampling=NASBench101Sampling(),
                crossover=NASBench101Crossover(),
                mutation=NASBench101Mutation(),
                eliminate_duplicates=False,
                callback=self.callback,
                save_history=True,
                evaluator=NASBench101Evaluator()
            )
            self.callback.initialize(self.algorithm)
            self.nadir = (100, 49979274)
            self.algorithm.nadir = self.nadir

        elif search_space == "nasbench301":
            self.problem = NASBench301Problem()
            self.algorithm = SMSEMOA(
                pop_size=50,
                n_offsprings=25,
                sampling=NASBench301Sampling(),
                crossover=NASBench301Crossover(),
                mutation=NASBench301Mutation(),
                eliminate_duplicates=False,
                callback=self.callback,
                save_history=True,
                evaluator=NASBench301Evaluator()
            )
            self.callback.initialize(self.algorithm)
            self.nadir = (100, 49979274)
            self.algorithm.nadir = self.nadir

        elif search_space == "radar":
            self.problem = RadarProblem(dataset)
            self.algorithm = SMSEMOA(
                pop_size=100,
                n_offsprings=25,
                sampling= IntegerRandomSampling(),
                crossover=SBX(eta=20),
                mutation=PolynomialMutation(eta=20),
                eliminate_duplicates=True,
                repair=RoundingRepair(),
                callback=self.callback,
                save_history=True,
            )
            self.callback.initialize(self.algorithm)
        # self.algorithm.setup(self.problem)


    def main_loop(self):
        res = minimize(self.problem,
                       self.algorithm,
                       self.termination,
                       save_history=True,
                       verbose=True)

        self.hypervolume_history = res.algorithm.callback.data["hypervolume"]
        return res


class MyCallback(Callback):

    def __init__(self) -> None:
        super().__init__()
        self.rec = Recorder(Streamer(sleep=0.1))
        self.data["hypervolume"] = []

    def notify(self, algorithm):
        problem = algorithm.problem
        approx_ideal = algorithm.pop.get("F").min(axis=0)
        approx_nadir = algorithm.pop.get("F").max(axis=0)
        metric = Hypervolume(ref_point=np.array(algorithm.nadir),
                             norm_ref_point=False,
                             zero_to_one=False,
                             ideal=approx_ideal,
                             nadir=approx_nadir)

        hv = metric.do(algorithm.pop.get("F"))
        self.data["hypervolume"].append(hv)
        scatter = Scatter("Gen %s" % algorithm.n_gen, {'pad': 30}, bounds=(problem.xl, problem.xu),)
        scatter.set_axis_style(color="grey", alpha=0.5)
        scatter.add(algorithm.pop.get("F"))
        scatter.do()
        #self.rec.record()


class PolynomialMutation(Mutation):

    def __init__(self, prob=0.9, eta=20, at_least_once=False, **kwargs):
        super().__init__(prob=prob, **kwargs)
        self.at_least_once = at_least_once
        self.eta = Real(eta, bounds=(3.0, 30.0), strict=(1.0, 100.0))

    def _do(self, problem, X, params=None, **kwargs):
        X = X.astype(float)

        eta = get(self.eta, size=len(X))
        prob_var = self.get_prob_var(problem, size=len(X))

        Xp = mut_pm(X, problem.xl, problem.xu, eta, prob_var, at_least_once=self.at_least_once)

        return Xp


class StartFromZeroRepair(Repair):

    def _do(self, problem, X, **kwargs):
        I = np.where(X == 0)[1]
        for k in range(len(X)):
            i = I[k]
            X[k] = np.concatenate([X[k, i:], X[k, :i]])

        return X

class PermutationRandomSamplingWithBias(Sampling):

    def _do(self, problem, n_samples, **kwargs):
        X = np.full((n_samples, problem.n_var), 0, dtype=int)
        for i in range(n_samples):
            path = [0]
            X[i, 0] = 0
            unvisited = list(range(1, problem.n_var))

            for j in range(1, problem.n_var):
                probabilities = [np.exp(problem.b[path[-1], u]) for u in unvisited]
                probabilities = probabilities / np.sum(probabilities)
                next_node = np.random.choice(unvisited, p=probabilities)
                X[i, j] = next_node
                unvisited.remove(next_node)
                path.append(next_node)
        return X


if __name__ == '__main__':
    problem = TSPTSWProblem("../data/tsptw/SolomonTSPTW/rc_204.3.txt")
