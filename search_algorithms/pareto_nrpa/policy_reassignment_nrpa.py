import numpy as np
from matplotlib import pyplot as plt
from pymoo.core.population import Population
from pymoo.indicators.hv import Hypervolume
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
import seaborn as sns

from search_algorithms.pareto_nrpa.pareto_nrpa import PolicyManager, ParetoNRPA


class ThirdIdeaNRPA(ParetoNRPA):

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
                metric = Hypervolume(ref_point=np.array(self.nadir),
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
                    manager = policy_manager.copy()
                    result = self.nrpa(node, level - 1, manager, optimal_set)

                optimal_set = Population.merge(optimal_set, result)
                if level == self.level:
                    print(f"NRPA search of level {level-1} has returned an optimal set of {len(result)}")
                    print(f"Now the optimal set has length {len(optimal_set)}")
                # Non-dominated sorting
                nds = NonDominatedSorting()
                fronts = nds.do(optimal_set.get("F"))
                indexes = []
                n_fronts = min(1, len(fronts))
                for j in range(n_fronts):
                    for f in fronts[j]:
                        el = optimal_set[f]
                        if el.get("F")[0] in [optimal_set[e].get("F")[0] for e in indexes]:
                            if el.get("F")[1] in [optimal_set[e].get("F")[1] for e in indexes]:
                                continue
                        indexes.append(f)


                if level == self.level:
                    # Policy reassignment

                    knn = KNeighborsClassifier(n_neighbors=min(len(optimal_set), 2))
                    print(f"\n -=-=-=-=-=-=-=- \n LEVEL {self.level}")
                    X = optimal_set.get("F")
                    Y = optimal_set.get("P")
                    print(f"X = {X}")
                    print(f"Y = {Y}")
                    knn.fit(X, Y)

                optimal_set = optimal_set[indexes]

                if level == self.level:
                    new_manager = PolicyManager(self.alpha)
                    cls = KMeans(n_clusters=min(len(optimal_set), self.max_pareto_set))
                    cls.fit(optimal_set.get("F"))
                    if level == self.level:
                        # Plotting the points
                        markers = ["o", "^", "s", "*", "P", "X"]
                        for i, elem in enumerate(optimal_set):
                            plt.scatter(elem.get("F")[0], elem.get("F")[1], c=sns.color_palette()[cls.labels_[i]],
                                    cmap='viridis', marker=markers[elem.get("P")])
                        plt.xlabel('Objective 1')
                        plt.ylabel('Objective 2')
                        plt.legend()
                        plt.show()
                        plt.close()
                    print(f"@level{level} policy manager has {policy_manager.policies.keys()}")
                    for i in range(min(len(optimal_set), self.max_pareto_set)):
                        # Calculate centroid of cluster i
                        centroid = cls.cluster_centers_[i]
                        # Find the closest point to the centroid
                        closest = np.argmin(np.linalg.norm(optimal_set.get("F") - centroid, axis=1))
                        # Assign the policy of the closest point to the centroid
                        new_manager.update_policy(i, policy_manager.get_policy(optimal_set[closest].get("P")).copy())

                    for i, elem in enumerate(optimal_set):
                        elem.set("P", cls.labels_[i])

                    policy_manager = new_manager
                    print(f"@level{level} our new policy manager has {policy_manager.policies.keys()}")

                policy_manager.adapt(optimal_set, self)
                if level == self.level:
                    self.callback(self, optimal_set)
            return optimal_set