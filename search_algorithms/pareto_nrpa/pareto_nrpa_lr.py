import numpy as np
from pymoo.core.population import Population
from pymoo.indicators.hv import Hypervolume
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

from search_algorithms.pareto_nrpa.pareto_nrpa import ParetoNRPA


class ParetoNRPALR(ParetoNRPA):

    def __init__(self, config):
        super().__init__(config)
        self.threshold = config.search.threshold
        self.ps = np.zeros((self.pm.n_policies))

    def nrpa(self, node, level, policy_manager, set_):
        if level == 0:

            p = np.random.choice(list(policy_manager.policies.keys()),
                                 )#p=self.softmax_temp_fn(np.array(list(policy_manager.weights.values())), 1))
            # print(f"Picking policy {p}")
            self.ps[p] += 1
            # print(self.ps)
            # print(policy_manager.weights)
            new_individual, sequence = self.next(p, policy_manager)

            # Check if new_individual is dominated by set_
            nds = NonDominatedSorting()
            fronts = nds.do(Population.merge(set_, new_individual).get("F"))
            is_on_pareto = new_individual in [Population.merge(set_, new_individual)[i] for i in fronts[0]]
            # print(f'Policy {p} is on the pareto front: {is_on_pareto}')

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
            count_threshold = np.zeros((policy_manager.n_policies))
            k = 0
            while np.sum(count_threshold) < self.threshold:
                if level >= 2:
                    print(f"[NRPA LEVEL {level}] with count threshold {count_threshold}")

                if self.advancement > self.n_iter ** self.level:
                    print("Enough iterations")
                    return optimal_set
                if level == 1:  # Avoid useless policy copy for playout level 0
                    result = self.nrpa(node, level - 1, policy_manager, optimal_set)
                else:
                    print(f"Launching a new search of level {level}, threshold={count_threshold}")
                    pm_copy = policy_manager.copy()
                    result = self.nrpa(node, level - 1, pm_copy, optimal_set)
                #
                # print(f"Result of level {level} : {result.get('F')}")
                # print(f"Optimal set: {optimal_set.get('F')}")
                for elem in result:
                    for e in optimal_set:
                        if level >= 2:
                            print(f"Comparing {[round(float(x), 2) for x in elem.get('F')]} (P{elem.get("P")} new search) and {[round(float(x), 2) for x in e.get('F')]} (P{e.get("P")} current optimal set)")
                        if elem.get("F")[0] == e.get("F")[0] and elem.get("F")[1] == e.get("F")[1] and level != self.level:
                            if elem.get("P") == e.get("P"):
                                count_threshold[elem.get("P")] += 1  # A voir, peut-être faux
                                # print(f"Same point found with policy {elem.get('P')}, count threshold : {count_threshold}")
                            # break
                        if elem.get("F")[0] < e.get("F")[0] and elem.get("F")[1] < e.get("F")[1]:
                            if elem.get("P") == e.get("P"):
                                # print(f"{[round(float(x), 2) for x in elem.get('F')]} (P{elem.get("P")} new search) < {[round(float(x), 2) for x in e.get('F')]} (P{e.get("P")} current optimal set)")
                                count_threshold[elem.get("P")] = 0
                                # print(count_threshold)
                                break

                opt_copy = optimal_set.copy()
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
                optimal_set = optimal_set[indexes]

                policy_manager.adapt(optimal_set, self)
                if level >= 2:
                    print(optimal_set.get("F"))
                k += 1
                if k > 1000:
                    print("Enough iterations")
                    return optimal_set
            return optimal_set