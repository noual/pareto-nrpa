import copy

import numpy as np
from pymoo.core.population import Population
from pymoo.indicators.hv import Hypervolume
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

from search_algorithms.pareto_nrpa.pareto_nrpa import ParetoNRPA, PolicyManager
from search_algorithms.pareto_nrpa.pareto_nrpa_lr import ParetoNRPALR


class HVPolicyManager(PolicyManager):

    def adapt(self, optimal_set, algorithm):
        """
        Adapt policies based on the optimal set.
        """

        # For each policy, find the solution that maximizes hypervolume improvement
        reference_nadir = optimal_set.get("F").max(axis=0) + 1e-6
        # print(f"[Adapt] Reference nadir: {reference_nadir}")
        global_hv = Hypervolume(ref_point=reference_nadir).do(optimal_set.get("F"))
        for p in range(self.n_policies):

            best_hv = 0
            best_sequence = None
            s_p = optimal_set[optimal_set.get("P") == p]
            if len(s_p) == 0:
                continue
            elif len(s_p) == 1:
                best_sequence = s_p[0]
            else:
                for sequence in s_p:
                    # Compute the hypervolume improvement
                    f_ = optimal_set.get("F")
                    f = np.array([x for x in f_ if x[0] != sequence.get("F")[0] or x[1] != sequence.get("F")[1]])

                    hv = Hypervolume(ref_point=reference_nadir)
                    hv_improvement = hv.do(f) - global_hv
                    # print(f"[Adapt] {hv_improvement} for {sequence.get('F')}")
                    if hv_improvement < best_hv:
                        best_hv = hv_improvement
                        best_sequence = sequence

            sequence = best_sequence.get("X")
            policy = self.policies[p]
            pol_prime = policy.copy()
            node_type = type(algorithm.root)
            node = node_type(state=copy.deepcopy(algorithm.root.state), move=None, parent=None, sequence=[])
            node.hash = node.calculate_zobrist_hash(algorithm.root.state.zobrist_table)
            for i, action in enumerate(sequence):
                best_code = algorithm._code(node, action)
                pol_prime[best_code] = pol_prime.get(best_code, 0) + self.alpha
                z = 0
                o = {}
                available_moves = node.get_action_tuples()
                move_codes = [algorithm._code(node, m) for m in available_moves]
                for move, move_code in zip(available_moves, move_codes):
                    # print(f"[Adapt] {node.state.path[i], move[0]}")
                    o[move_code] = np.exp(policy.get(move_code, 0) + algorithm.b[(node.state.path[i], move[0])])
                    z += o[move_code]
                for move, move_code in zip(available_moves, move_codes):
                    pol_prime[move_code] = pol_prime.get(move_code, 0) - self.alpha * (o[move_code] / z)

                node.play_action(action)
                node.hash = node.calculate_zobrist_hash(algorithm.root.state.zobrist_table)


            self.update_policy(p, pol_prime)
        self.update_weights(optimal_set)

class HVParetoNRPA(ParetoNRPA):

    def __init__(self, config):
        super().__init__(config)
        self.pm = HVPolicyManager(alpha=self.alpha)
        self._initialize()


class HVParetoNRPALR(HVParetoNRPA, ParetoNRPALR):
    """
    HVParetoNRPA with LR policy adaptation
    """

    def __init__(self, config):
        super().__init__(config)
        self.pm = HVPolicyManager(alpha=self.alpha)
        self._initialize()

    def nrpa(self, node, level, policy_manager, set_):
        return ParetoNRPALR.nrpa(self, node, level, policy_manager, set_)