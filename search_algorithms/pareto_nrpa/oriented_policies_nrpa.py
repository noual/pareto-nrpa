import copy

import numpy as np

from search_algorithms.pareto_nrpa.pareto_nrpa import ParetoNRPA, PolicyManager


class OrientedPolicyManager(PolicyManager):

    def adapt(self, optimal_set, algorithm):
        """
        Adapt policies based on the optimal set.
        """
        for i in range(self.n_policies):
            mu = i / (self.n_policies-1)
            # Find the point on the pareto front that minimizes the weighted objectives
            Fs = optimal_set.get("F")
            max_f0 = max([e[0] for e in Fs])
            max_f1 = max([e[1] for e in Fs])
            Ms = [(mu*((e[0])**2)/(max_f0**2)  + (1-mu)*((e[1])**2)/(max_f1**2) ) for e in Fs]
            index = np.argmin(Ms)
            elem = optimal_set[index]
            # for j, elem in enumerate(optimal_set):
            sequence = elem.get("X")
            policy = self.policies[i]
            pol_prime = policy.copy()
            node_type = type(algorithm.root)
            node = node_type(state=copy.deepcopy(algorithm.root.state), move=None, parent=None, sequence=[])
            node.hash = node.calculate_zobrist_hash(algorithm.root.state.zobrist_table)
            for action in sequence:
                code = algorithm._code(node, action)
                if code not in pol_prime:
                    # print("Erreur 0")
                    pol_prime[code] = 0
                pol_prime[code] += self.alpha ## * Ms[j]
                z = 0
                moves = node.get_action_tuples()
                move_codes = [algorithm._code(node, m) for m in moves]
                # derivatives = []
                for m in move_codes:
                    z += np.exp(policy.get(m, 0))
                for m in move_codes:
                    pol_prime[m] = pol_prime.get(m, 0) - self.alpha * (np.exp(policy.get(m, 0)) / z) ##* Ms[j]
                    p_ij = np.exp(policy.get(m, 0)) / z
                    # if m == code:
                    #     derivatives.append(p_ij - 1)
                    # else:
                    #     derivatives.append(p_ij)

                node.play_action(action)
                node.hash = node.calculate_zobrist_hash(algorithm.root.state.zobrist_table)

                self.update_policy(i, pol_prime)
        self.update_weights(optimal_set)

class OrientedPoliciesNRPA(ParetoNRPA):

    def __init__(self, config):
        super().__init__(config)
        self.pm = OrientedPolicyManager(alpha=self.alpha)
        self._initialize()