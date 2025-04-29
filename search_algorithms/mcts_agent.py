import copy
import json
import sys

import pandas as pd
# from nasbench import api
from yacs.config import CfgNode

from naslib.utils import get_dataset_api
# from monet.node import Node
# from monet.search_spaces.nasbench101_node import NASBench101Cell
# from monet.search_spaces.nasbench201_node import NASBench201Cell
# from monet.search_spaces.nasbench301_node import DARTSState, DARTSCell
# from monet.search_spaces.natsbench_node import NATSBenchSizeCell
# from monet.search_spaces.transbench_node import TransBenchCell, TransBenchMacro
# from monet.search_spaces.tsptw_node import TSPTWState
# from naslib2.search_spaces.core import Metric
# from naslib2.utils import get_dataset_api

from node import Node
from search_spaces.nasbench101.nasbench101_node import NASBench101Cell
from search_spaces.nasbench201.nasbench201_node import NASBench201Cell
from search_spaces.nasbench301.nasbench301_node import DARTSState, DARTSCell
from search_spaces.radar.radar_node import RadarCell
from search_spaces.tsptw.tsptw_node import TSPTWState

sys.path.append("..")
sys.path.append("../..")


import time
import numpy as np
from tqdm import tqdm


class MCTSAgent:

    def __init__(self, config: CfgNode()):
        self.root = None
        self.api = None
        self.df = None
        if config.df_path != "none":
            self.df = pd.read_csv(config.df_path)
        self.playouts_per_selection = config.search.playouts_per_selection
        self.n_iter = config.search.n_iter
        self.disable_tqdm = config.disable_tqdm
        self.nadir = (None, None)  # For HV calculation
        self.b = {}

    def adapt_search_space(self, search_space, dataset):
        print(search_space, dataset)
        self.search_space = search_space
        self.dataset = dataset
        assert search_space in ["nasbench201", "nasbench101", "nasbench301", "natsbenchsize",
                                "transbench101_macro", "transbench101_micro", "tsptw", "tsptw_moo", "radar"],\
            "Only NASBench301, NASBench201, NASBench101, NATS-Bench are supported"
        if search_space == "nasbench201":
            # if isinstance(self, UCT):
            #     print(f"Reducing number of iterations")
            #     self.n_iter = self.n_iter // 6
            assert dataset in ["cifar10"], "Only CIFAR10 is supported"
            self.root = Node(state=NASBench201Cell())
            self.nadir = (100, 1531556)  # worst accuracy and biggest number of params

        elif search_space == "nasbench101":
            # if isinstance(self, UCT):
            #     print(f"Reducing number of iterations")
            #     self.n_iter = self.n_iter // 12
            assert dataset in ["cifar10"], "Only CIFAR10 is supported"
            self.root = Node(state=NASBench101Cell(7))
            self.api = get_dataset_api(search_space, dataset)["nb101_data"]
            if issubclass(type(self), MCTSAgent):
                self._playout = self._playout_101
            self.nadir = (100, 49980274)
        #
        if search_space == "nasbench301":

            assert dataset in ["cifar10"], "Only CIFAR10 is supported"
            self.root = Node(state= DARTSState((DARTSCell(),
                                                DARTSCell()))
                             )
            self.api = get_dataset_api(search_space, dataset)
        #
        # elif search_space == "natsbenchsize":
        #     if isinstance(self, UCT):
        #         print(f"Reducing number of iterations")
        #         self.n_iter = self.n_iter // 5
        #     assert dataset in ["cifar100"], "Only CIFAR10 is supported"
        #     self.root = Node(state=NATSBenchSizeCell())
        #     self.api = get_dataset_api(search_space, dataset)
        #
        # elif search_space == "transbench101_micro":
        #     if isinstance(self, UCT):
        #         print(f"Reducing number of iterations")
        #         self.n_iter = self.n_iter // 6
        #     assert dataset in ["jigsaw", "class_scene", "class_object", "room_layout", "segmentsemantic", "normal", "autoencoder"], "Only CIFAR10 is supported"
        #     self.root = Node(state=TransBenchCell())
        #     self.api = get_dataset_api(search_space, dataset)
        #
        # elif search_space == "transbench101_macro":
        #     if isinstance(self, UCT):
        #         print(f"Reducing number of iterations")
        #         self.n_iter = self.n_iter // 4
        #     assert dataset in ["jigsaw", "class_scene", "class_object", "room_layout", "segmentsemantic", "normal", "autoencoder"], "Only CIFAR10 is supported"
        #     self.root = Node(state=TransBenchMacro())
        #     self.api = get_dataset_api(search_space, dataset)

        elif search_space == "tsptw":
            self.root = Node(state=TSPTWState(file=f"../data/SolomonTSPTW/{dataset}.txt"))
            self.api = None

        elif search_space == "tsptw_moo":
            self.root = Node(state=TSPTWState(file=f"../data/tsptw/SolomonTSPTW/{dataset}.txt", multiobjective=True))
            # Initialize bias for GNRPA
            distances = self.root.state.travel_matrix
            max_ = np.max(distances)
            min_ = np.min(distances)
            for i in range(distances.shape[0]):
                for j in range(distances.shape[1]):
                    self.b[(i, j)] = -10 * (distances[i, j]-min_)/(max_-min_)
                    print(f"{i} -> {j}: {self.b[(i, j)]}")
            with open(f"../data/tsptw/SolomonTSPTW/nadirs.json", "r") as f:
                nadirs = json.load(f)
            self.nadir = nadirs[dataset]
            self.api = None

        elif search_space == "snake_in_the_box":
            self.root = Node(state=SITBState(dataset))

        elif search_space == "radar":
            self.root = Node(state=RadarCell(dataset))

    def _score_node(self, node: Node, parent: Node):
        pass

    def _get_reward(self, node: Node):
        pass

    def _create_network(self, node: Node):
        """
        Créer un réseau de neurones à partir du noeud MCTS
        :param node:
        :return: nn.Module
        """
        pass

    def _selection(self, node: Node):
        pass

    def _expansion(self, node: Node):
        pass

    def _playout(self, node: Node):
        pass

    def _playout_101(self, node: Node):
        pass

    def _backpropagation(self, node: Node, result: float):
        pass

    def __call__(self):
        pass
