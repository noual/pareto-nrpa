import numpy as np
from pymoo.core.population import Population
from pymoo.indicators.hv import Hypervolume
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

from search_algorithms.pareto_nrpa.pareto_nrpa import PolicyManager, ParetoNRPA


class SliceParetoNRPA(ParetoNRPA):

    def nrpa(self, node, level, policy_manager, set_):
        if level == 0:
            # Choose a random policy and perform a playout
            weights = [policy_manager.weights.get(e, 1) for e in policy_manager.policies.keys()]
            p = np.random.choice(list(policy_manager.policies.keys()),
                                 p=self.softmax_temp_fn(np.array(weights), 1))

            new_individual, sequence = self.next(p, policy_manager)

            new_individual.set("P", p)
            individual = Population.merge(Population(), new_individual)
            # print(f"[{len(self.rewards)}/{self.n_iter ** self.level}] Best reward: {max(self.best_reward)}")
            self.anytime_pareto_set = Population.merge(self.anytime_pareto_set, new_individual)

            if self.advancement % 1000 == 0:
                """
                Video callback and hypervolume calculation
                """
                # Anytime Pareto set
                nds = NonDominatedSorting()
                front = nds.do(self.anytime_pareto_set.get("F"), only_non_dominated_front=True)
                self.anytime_pareto_set = self.anytime_pareto_set[front]
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
            self.pbar.update(1)
            return individual
        else:
            optimal_set = Population()
            for i in range(self.n_iter):
                if level == 1:  # Avoid useless policy copy for playout level 0
                    result = self.nrpa(node, level - 1, policy_manager, optimal_set)
                else:
                    # Chose a policy to explore at lower levels
                    weights = [policy_manager.weights.get(e, 1) for e in policy_manager.policies.keys()]
                    policy_to_explore = np.random.choice(list(policy_manager.policies.keys()),
                                                         p=self.softmax_temp_fn(np.array(weights), 1),
                                                         size=level-1,
                                                         replace=False)
                    manager = PolicyManager(alpha=self.alpha)
                    if level >= 2:
                        print(f"policy weights: {policy_manager.weights}")
                        print(f"Launching NRPA level {level -1} with policy(ies) {policy_to_explore}")
                    for policy in policy_to_explore:
                        manager.update_policy(int(policy), policy_manager.get_policy(policy))
                        manager.weights[policy] = 1
                    result = self.nrpa(node, level - 1, manager, optimal_set)
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
                if level >= 2:
                    osi = optimal_set[indexes]
                    missing_indexes = [i for i in list(policy_manager.policies.keys()) if i not in osi.get("P")]
                    j = 1
                    finished = False
                    while not finished:
                        if j == len(fronts):
                            break
                        for f in fronts[j]:
                            if len(missing_indexes) == 0:
                                finished = True
                                break
                            if optimal_set[f].get("P") in missing_indexes:
                                indexes.append(f)
                                missing_indexes.remove(optimal_set[f].get("P"))
                        j += 1
                optimal_set = optimal_set[indexes]

                policy_manager.adapt(optimal_set, self)
                if level == self.level:
                    self.callback(self, optimal_set)
            return optimal_set