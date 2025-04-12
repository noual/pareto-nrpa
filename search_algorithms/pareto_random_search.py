import time

from pymoo.core.population import Population
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from tqdm import tqdm

from search_algorithms.pareto_nrpa.pareto_nrpa import ParetoNRPA


class ParetoRandomSearch(ParetoNRPA):

    def __init__(self, config):
        super().__init__(config)

    def random_search(self, node):
        optimal_set = Population()
        j = 0
        for i in range(self.n_iter**self.level):
            new_individual, sequence = self.next(0)
            j += 1
            optimal_set = Population.merge(optimal_set, new_individual)
            self.pbar.update(1)
        # Non-dominated sorting
        print(j)
        nds = NonDominatedSorting()
        fronts = nds.do(optimal_set.get("F"))
        optimal_set = optimal_set[fronts[0]]
        return optimal_set


    def main_loop(self, app=None):
        node = self.root
        pol = {}
        print(self.n_iter, self.level)
        self.pbar = tqdm(total=self.n_iter ** self.level, position=0, leave=True)
        t1 = time.time()
        optimal_set = self.random_search(node)
        t2 = time.time()
        self.pbar.close()
        # print(f"Sequence is {sequence} with score {reward}")
        # for action in sequence:
        #     node.play_action(action)
        return optimal_set