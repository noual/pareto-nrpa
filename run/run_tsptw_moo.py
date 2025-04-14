import numpy as np
import pandas as pd
from pymoo.core.result import Result
from yacs.config import CfgNode

from search_algorithms.nsga2 import NSGAII
from search_algorithms.pareto_mcts import Pareto_UCT
from search_algorithms.pareto_nrpa.hv_pareto_nrpa import HVParetoNRPA, HVParetoNRPALR
from search_algorithms.pareto_nrpa.oriented_policies_nrpa import OrientedPoliciesNRPA
from search_algorithms.pareto_nrpa.pareto_nrpa import ParetoNRPA, ParetoRandomSearch, ParetoNRPAPolicyRepresentation
from search_algorithms.pareto_nrpa.pareto_nrpa_lr import ParetoNRPALR
from search_algorithms.pareto_nrpa.policy_reassignment_nrpa import ThirdIdeaNRPA
from search_algorithms.pareto_nrpa.slice_pareto_nrpa import SliceParetoNRPA
from search_algorithms.sms_emoa import SMSEMOAAlgorithm

SEARCH_SPACE = "tsptw_moo"
DATASET = "rc_207.3"

N_RUNS = 30
OUTPUT_FILE = "results"

N_ITER = 100000

def run_once(algo_dict):
    rewards = {}
    hypervolumes = {}
    for name, properties in algo_dict.items():
        print(f"Running {name}...")
        optimizer = properties["algorithm"]
        config = properties["config"]

        alg = optimizer(config)
        alg.adapt_search_space(SEARCH_SPACE, DATASET)
        optimal_set = alg.main_loop()

        hypervolumes[name] = alg.hypervolume_history
        print(hypervolumes)

        if isinstance(optimal_set, Result):
            if not hasattr(optimal_set, "P"):
                optimal_set.P = np.zeros(optimal_set.X.shape[0])
            rewards[name] = {"X": optimal_set.X.squeeze().tolist(),
                             "F": optimal_set.F,
                             "P": optimal_set.P,}
        else:
            rewards[name] = {"X": [e.get("X") for e in optimal_set],
                             "F": [e.get("F") for e in optimal_set],
                             "P": [e.get("P") for e in optimal_set]}

    return rewards, hypervolumes

def run_all(algo_dict, output_file="results_local"):
    all_results = []
    hypervolumes = []
    for n_run in range(N_RUNS):

        rewards, hv = run_once(algo_dict)
        for name, reward in rewards.items():
            print(name, reward)
            for i in range(len(rewards[name]["P"])):
                all_results.append({
                    "algorithm": name,
                    "run": n_run,
                    "sequence": rewards[name]["X"][i],
                    "objective_1": rewards[name]["F"][i][0],
                    "objective_2": rewards[name]["F"][i][1],
                    "policy": rewards[name]["P"][i]
                })
        for name, hypervolumes_ in hv.items():
            for i, hv_ in enumerate(hypervolumes_):
                hypervolumes.append({
                    "algorithm": name,
                    "run": n_run,
                    "iteration": i*(N_ITER//len(hypervolumes_)-1),
                    "hypervolume": hv_ })
        df = pd.DataFrame(all_results)
        df.to_csv(f"results/nrpa_{SEARCH_SPACE}_{DATASET}.csv")
        df_hv = pd.DataFrame(hypervolumes)
        df_hv.to_csv(f"results/nrpa_{SEARCH_SPACE}_{DATASET}_hv.csv")

if __name__ == '__main__':
    algorithms = {

        # "HV-Pareto-NRPA": {
        #     "algorithm": HVParetoNRPA,
        #     "config": CfgNode({
        #         "df_path": "none",
        #         "search": {
        #             "level": 3,
        #             "nrpa_alpha": 1,
        #             "nrpa_lr_update": False,
        #             "softmax_temp": 2,
        #             "playouts_per_selection": 1,
        #             "n_iter": N_ITER,
        #             "n_policies": 4,
        #             "threshold": 2
        #         },
        #         "disable_tqdm": "true",
        #         "callback": "true",
        #         "seed": 0
        #     })
        # },
        # "SMS-EMOA": {
        #     "algorithm": SMSEMOAAlgorithm,
        #     "config": CfgNode({
        #         "df_path": "none",
        #         "search": {
        #             "n_iter": N_ITER,
        #             "population_size": 200,
        #             "sample_size": 25
        #         },
        #         "disable_tqdm": "false",
        #         "seed": 0
        #     })
        # },
        # "Pareto-NRPALR": {
        #     "algorithm": ParetoNRPALR,
        #     "config": CfgNode({
        #         "df_path": "none",
        #         "search": {
        #             "level": 4,
        #             "nrpa_alpha": .5,
        #             "nrpa_lr_update": False,
        #             "softmax_temp": 1,
        #             "playouts_per_selection": 1,
        #             "n_iter": N_ITER,
        #             "n_policies": 4,
        #             "threshold": 2
        #         },
        #         "disable_tqdm": "true",
        #         "callback": "true",
        #         "seed": 0
        #     })
        # },
        "Pareto-NRPA": {
            "algorithm": ParetoNRPA,
            "config": CfgNode({
                "df_path": "none",
                "search": {
                    "level": 4,
                    "nrpa_alpha": .5,
                    "nrpa_lr_update": False,
                    "softmax_temp": 1,
                    "playouts_per_selection": 1,
                    "n_iter": N_ITER,
                    "n_policies": 4
                },
                "disable_tqdm": "true",
                "callback": "true",
                "seed": 0
            })
        },
        # "NSGAII": {
        #     "algorithm": NSGAII,
        #     "config": CfgNode({
        #         "df_path": "none",
        #         "search": {
        #             "n_iter": N_ITER,
        #             "population_size": 250,
        #             "sample_size": 25
        #         },
        #         "disable_tqdm": "false",
        #         "seed": 0
        #     })
        # },
        # "Pareto-UCT": {
        #     "algorithm": Pareto_UCT,
        #     "config": CfgNode({
        #         "df_path": "none",
        #         "search": {
        #             "n_iter": N_ITER//4,
        #             'C': 1,
        #             "playouts_per_selection": 4
        #         },
        #         "disable_tqdm": "false",
        #         "seed": 0,
        #         "callback": "true"
        #     })
        # },
        # "Pareto-RS": {
        #     "algorithm": ParetoRandomSearch,
        #     "config": CfgNode({
        #         "df_path": "none",
        #         "search": {
        #             "level": 3,
        #             "nrpa_alpha": 1,
        #             "nrpa_lr_update": False,
        #             "softmax_temp": 1,
        #             "playouts_per_selection": 1,
        #             "n_iter": N_ITER,
        #             "n_policies": 4
        #         },
        #         "disable_tqdm": "true",
        #         "callback": "true",
        #         "seed": 0
        #     })
        # },
        # "Pareto-NRPA_PolicyReassignment": {
        #     "algorithm": ThirdIdeaNRPA,
        #     "config": CfgNode({
        #         "df_path": "none",
        #         "search": {
        #             "level": 4,
        #             "nrpa_alpha": .5,
        #             "nrpa_lr_update": False,
        #             "softmax_temp": 1,
        #             "playouts_per_selection": 1,
        #             "n_iter": N_ITER,
        #             "n_policies": 4
        #         },
        #         "disable_tqdm": "true",
        #         "callback": "true",
        #         "seed": 0
        #     })
        # },
        "Pareto-NRPA_OrientedPolicies": {
            "algorithm": OrientedPoliciesNRPA,
            "config": CfgNode({
                "df_path": "none",
                "search": {
                    "level": 4,
                    "nrpa_alpha": .5,
                    "nrpa_lr_update": False,
                    "softmax_temp": 1,
                    "playouts_per_selection": 1,
                    "n_iter": N_ITER,
                    "n_policies": 4,
                    "C": 2
                },
                "disable_tqdm": "true",
                "callback": "true",
                "seed": 0
            })
        },
        # "Slice_Pareto-NRPA": {
        #     "algorithm": SliceParetoNRPA,
        #     "config": CfgNode({
        #         "df_path": "none",
        #         "search": {
        #             "level": 3,
        #             "nrpa_alpha": .5,
        #             "nrpa_lr_update": False,
        #             "softmax_temp": 1,
        #             "playouts_per_selection": 1,
        #             "n_iter": N_ITER,
        #             "n_policies": 4,
        #             "C": 2
        #         },
        #         "disable_tqdm": "true",
        #         "callback": "true",
        #         "seed": 0
        #     })
        # },
        # "3RD": {
        #     "algorithm": ThirdIdeaNRPA,
        #     "config": CfgNode({
        #         "df_path": "none",
        #         "search": {
        #             "level": 3,
        #             "nrpa_alpha": 1,
        #             "nrpa_lr_update": False,
        #             "softmax_temp": 1,
        #             "playouts_per_selection": 1,
        #             "n_iter": 30 ** 3,
        #             "n_policies": 4
        #         },
        #         "disable_tqdm": "true",
        #         "callback": "true",
        #         "seed": 0
        #     })
        # },
    }

    run_all(algorithms, OUTPUT_FILE)
