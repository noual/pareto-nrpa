import random

import torch
from graphviz import Digraph
import torch.nn as nn
import numpy as np
from pymoo.core.problem import Problem, ElementwiseProblem
import torch.nn.functional as F

from search_spaces.radar.radar_dataset import RadarDavaDataset
from search_spaces.radar.unet import DoubleConv
from utils_moo.CIFAR import CIFAR10Dataset
from utils_moo.ntk.compute_score import compute_score
from utils_moo.ntk.naswot import NASWOT
from utils_moo.ntk.zen_nas import ZenNAS

from utils_moo.operations import ReLUConvBN, Pooling, FactorizedReduce, Zero, ReLUDepthwiseConvBN

OPERATIONS = {"nor_conv_1x1": lambda C_in, C_out, stride, affine: ReLUConvBN(C_in, C_out, 1, stride, "same", affine),
              "nor_conv_3x3": lambda C_in, C_out, stride, affine: ReLUConvBN(C_in, C_out, 3, stride, "same", affine),
              "none": lambda C_in, C_out, stride, affine: Zero(C_in, C_out, stride),
              "avg_pool_3x3": lambda C_in, C_out, stride, affine: Pooling(C_in, C_out, stride, "avg", affine),
              "skip_connect": lambda C_in, C_out, stride, affine: nn.Identity() if stride == 1 and C_in == C_out
              else FactorizedReduce(C_in, C_out, stride, affine),
              "dep_conv_3x3": lambda C_in, C_out, stride, affine: ReLUDepthwiseConvBN(C_in, C_out, 3, stride, "same", affine),
              }


class NASBench201NetworkCell(nn.Module):

    def __init__(self, cell_str, C_in, C_out, n_vertices=4):
        super().__init__()
        # print(cell_str)
        self.n_vertices = n_vertices
        self.C_in = C_in
        self.C_out = C_out
        self.matrix = None
        self.reduce = FactorizedReduce(self.C_in, self.C_out, 1, True)
        self.build(cell_str)

    def build(self, cell_str):
        connexions = cell_str.split("+")
        matrix = nn.ModuleList()
        for i, connexion in enumerate(connexions):
            row = nn.ModuleList()
            list_c = [x for x in connexion.split("|") if x]

            for j, operation in enumerate(list_c):
                op, idx = operation.split('~')

                if j == 0:
                    row.append(OPERATIONS[op](self.C_in, self.C_out, stride=1, affine=True))
                else:
                    row.append(OPERATIONS[op](self.C_out, self.C_out, stride=1, affine=True))
            # print(f"Row {i}: {row}")
            matrix.append(row)
        self.matrix = matrix

    def forward(self, x):
        next_x = [x]
        for i in range(self.n_vertices-1):
            node_results = []
            for j in range(i+1):
                operation = self.matrix[i][j]
                op_result = operation(next_x[j])
                node_results.append(op_result)
            node_result = torch.sum(torch.stack(node_results), dim=0)
            next_x.append(node_result)
        return next_x[-1]

import torch
from torch import nn

from utils_moo.operations import ReLUConvBN


class NASBench201UNet(nn.Module):

    def __init__(self, cell_str, input_size, input_depth, n_vertices=4, features=[16, 32, 64, 128]):
        super().__init__()
        self.cell_str = cell_str
        self.input_size = input_size
        self.n_vertices = n_vertices
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.pool = nn.MaxPool2d(2)
        self.input_conv = nn.Sequential(nn.Conv2d(input_depth, features[0], kernel_size=3, stride=1, padding=1),
                                        nn.BatchNorm2d(features[0]),)

        # Encoder
        input_depth = features[0]
        for feature in features:
            self.downs.append(NASBench201NetworkCell(self.cell_str, C_in=input_depth, C_out=feature, n_vertices=self.n_vertices))
            input_depth = feature

        # Bottleneck
        self.bottleneck = DoubleConv(features[-1], features[-1]*2)
        # self.bottleneck = NASBench201NetworkCell(self.cell_str, C_in=features[-1], C_out=features[-1]*2, n_vertices=self.n_vertices)

        # Decoder
        for feature in reversed(features):
            self.ups.append(nn.Sequential(
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
                ReLUConvBN(feature*2, feature, 1, 1, 1, True)
            ))
            self.ups.append(NASBench201NetworkCell(self.cell_str, C_in=feature*2, C_out=feature, n_vertices=self.n_vertices))

        # Final output
        self.final_conv = nn.Conv2d(features[0], 1, kernel_size=1)

    def forward(self, x):
        skip_connections = []
        x = self.input_conv(x)
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_conn = skip_connections[idx//2]

            if x.shape != skip_conn.shape:
                x = F.interpolate(x, size=skip_conn.shape[2:])

            x = torch.cat((skip_conn, x), dim=1)
            x = self.ups[idx+1](x)

        return self.final_conv(x)


class NASBench201Model(nn.Module):

    def __init__(self, cell_str, input_size, input_depth):
        super().__init__()
        self.cell_str = cell_str
        self.backbone = self.build_backbone(input_size, input_depth)

    def build_backbone(self, input_size, input_depth):
        pass

    def forward(self):
        pass


    # def __init__(self, cell_str, input_size, input_depth, n_vertices=4):
    #
    #     self.C = 64
    #     self.N = 1
    #     self.layer_channels = [self.C] * self.N + [self.C * 2] + [self.C * 2] * self.N + [self.C * 4] + [
    #         self.C * 4] * self.N
    #     self.layer_reductions = [False] * self.N + [True] + [False] * self.N + [True] + [False] * self.N
    #     self.layer_channels = [64, 128, 256, 512]
    #     self.layer_reductions = [True, True, True, True]
    #     print(self.layer_channels)
    #     print(self.layer_reductions)
    #     self.n_vertices = n_vertices
    #     super().__init__(cell_str, input_size, input_depth)
    #
    # def build_backbone(self, input_size, input_depth):
    #
    #     self.first_conv = ReLUConvBN(input_depth, self.C, 3, 1, "same", True)
    #     self.encoder = nn.ModuleList()
    #
    #     for lc, reduction in zip(self.layer_channels, self.layer_reductions):
    #         if not reduction:
    #             c = NASBench201NetworkCell(self.cell_str, C_in=lc, C_out=lc, n_vertices=self.n_vertices)
    #             self.encoder.append(c)
    #         else:
    #             # c = ReLUConvBN(lc // 2, lc, 3, 2, 1, True)
    #             c = NASBench201NetworkCell(self.cell_str, C_in=lc, C_out=lc*2, n_vertices=self.n_vertices)
    #             self.encoder.append(c)
    #             self.encoder.append(nn.MaxPool2d(2, 2))
    #
    #     self.bottom_conv = ReLUConvBN(self.layer_channels[-1], self.layer_channels[-1], 3, 1, "same", True)
    #     self.decoder = nn.ModuleList()
    #
    #     for lc, reduction in zip(reversed(self.layer_channels), reversed(self.layer_reductions)):
    #         if not reduction:
    #             c = NASBench201NetworkCell(self.cell_str, C_in=lc, C_out=lc, n_vertices=self.n_vertices)
    #             self.decoder.append(c)
    #         else:
    #             # c = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
    #             c = nn.ConvTranspose2d(lc, lc // 2, 3, 2, 1, 1)
    #             self.decoder.append(c)
    #             c = ReLUConvBN(lc, lc // 2, 3, 1, 1, True)
    #             self.decoder.append(c)
    #
    #     self.last_conv = nn.Conv2d(self.layer_channels[0], input_depth, 1, 1, "same")
    #
    # def forward(self, x):
    #
    #     x = self.first_conv(x)
    #     encoder_tensors = []
    #     for i, mod in enumerate(self.encoder):
    #         x = mod(x)
    #         # print(f"{i} : {x.shape}")
    #         encoder_tensors.append(x)
    #     x = self.bottom_conv(x)
    #     for i, mod in enumerate(self.decoder):
    #         x = mod(x)
    #         if isinstance(self.decoder[i - 1], nn.Upsample):
    #             x = torch.add(x, list(reversed(encoder_tensors))[i])
    #     x = self.last_conv(x)
    #     # x = nn.Sigmoid()(x)
    #     return x


class NASBench201UNet_NTK(NASBench201UNet):

    def __init__(self, cell_str, input_size, input_depth):
        super().__init__(cell_str, input_size, input_depth)

    def build_backbone(self, input_size, input_depth):
        super().build_backbone(input_size, input_depth)
        latent_space_dim1 = input_size // (2 * np.sum(self.layer_reductions))
        latent_space_dim2 = np.max(self.layer_channels)
        self.dense_ntk = nn.Linear(in_features=latent_space_dim1 * latent_space_dim1 * latent_space_dim2,
                                   out_features=10)

    def forward(self, x):
        x = self.first_conv(x)
        encoder_tensors = []
        for i, mod in enumerate(self.encoder):
            x = mod(x)
            encoder_tensors.append(x)
        x_bottom = self.bottom_conv(x)
        x_for_ntk = x_bottom.view(x_bottom.shape[0], -1)
        x_for_ntk = self.dense_ntk(x_for_ntk)
        return x_for_ntk

class NASBench201Vertice:
    """
    Class representing a single vertice in the NASBench201 search space.
    """

    def __init__(self, id):
        """
        Initialize a NASBench201Vertice instance.

        Parameters:
        id (int): The ID of the vertice.
        """
        self.id = id
        self.actions = {i: None for i in range(id)}
        self.OPERATIONS = ["none", "skip_connect", "nor_conv_1x1", "nor_conv_3x3", "avg_pool_3x3", "dep_conv_3x3"]

    def is_complete(self):
        """
        Check if all actions for the vertice are defined.

        Returns:
        bool: True if all actions are defined, False otherwise.
        """
        return None not in self.actions.values()

    def play_action(self, id, operation):
        """
        Define an action for the vertice.

        Parameters:
        id (int): The ID of the input vertice.
        operation (str): The operation to be performed.
        """
        self.actions[id] = operation

    def get_action_tuples(self):
        """
        Get all possible action tuples for the vertice.

        Returns:
        list: A list of tuples representing possible actions.
        """
        list_tuples = []
        for k, v in self.actions.items():
            if v is None:
                for op in self.OPERATIONS:
                    list_tuples.append((k, op))
        return list_tuples

class RadarCell:
    """
    NAS-Bench-201 cell used to create a neural net
    """

    def __init__(self, n_vertices=4, vertice_type=NASBench201Vertice):
        """
        Initialize a NASBench201Cell instance.

        Parameters:
        n_vertices (int): The number of vertices in the cell.
        vertice_type (class): The class type for vertices.
        """
        self.n_vertices = n_vertices
        self.vertices = [vertice_type(i) for i in range(n_vertices)]
        self.OPERATIONS = self.vertices[0].OPERATIONS
        self.zobrist_table = None
        self.N_NODES = n_vertices
        self.ADJACENCY_MATRIX_SIZE = self.N_NODES ** 2
        self.N_OPERATIONS = 6

        self.lat_metric = "latency"
        self.network = None
        self.dataset_full = RadarDavaDataset(root_dir="../../data/radar/training_set/mat", batch_size=8, has_distance=True)
        self.dataset = self.dataset_full.generate_loaders()[0]
        self.acc_metric = ZenNAS(self.dataset)

    def is_complete(self):
        """
        Check if all vertices in the cell are complete.

        Returns:
        bool: True if all vertices are complete, False otherwise.
        """
        return all([v.is_complete() for v in self.vertices])

    def to_str(self):
        """
        Convert the cell to a string representation.

        Returns:
        str: The string representation of the cell.
        """
        assert self.is_complete(), "cell is incomplete"
        res = ""
        for v in self.vertices[1:]:
            temp_str = "|"
            # sorted_actions = collections.OrderedDict(sorted(d.items()))
            for source, operation in v.actions.items():
                temp_str += f"{operation}~{source}|"
            res += temp_str + "+"
        return res[:-1]  # -1 pour enlever le dernier "+"

    def adjacency_matrix(self):
        """
        Generate the adjacency matrix for the cell.

        Returns:
        np.ndarray: The adjacency matrix of the cell.
        """
        adjacency_matrix = np.zeros((self.n_vertices, self.n_vertices), dtype="int8")
        for i, vertice in enumerate(self.vertices):
            for j, operation in vertice.actions.items():
                if operation is None:
                    operation = "none"
                adjacency_matrix[j,i] = self.OPERATIONS.index(operation)
        return adjacency_matrix

    def play_action(self, vertice, id, operation):
        """
        Define an action for a specific vertice in the cell.

        Parameters:
        vertice (int): The vertice that plays the action.
        id (int): The vertice that acts as input for the action.
        operation (str): The operation to be performed.
        """
        self.vertices[vertice].play_action(id, operation)

    def get_action_tuples(self):
        """
        Get all possible action tuples for the cell.

        Returns:
        list: A list of tuples representing possible actions.
        """
        list_tuples = []
        for i, v in enumerate(self.vertices):
            actions = v.get_action_tuples()
            for action in actions:
                list_tuples.append((i, *action))
        return list_tuples

    def calculate_zobrist_hash(self, zobrist_table):
        """
        Calculate the Zobrist hash for the cell.

        Parameters:
        zobrist_table (list): The Zobrist table used for hashing.

        Returns:
        int: The Zobrist hash of the cell.
        """
        assert zobrist_table is not None, "Remember to pass zobrist_table to node constructor."
        hash = 0
        adjacency = self.adjacency_matrix()
        for i, row in enumerate(adjacency):
            for element in row:
                hash ^= zobrist_table[i][element]
        return hash

    def initialize_zobrist_table(self):
        self.zobrist_table = []
        for i in range(self.ADJACENCY_MATRIX_SIZE):
            adjacency_table = []
            for operation in range(self.N_OPERATIONS):
                adjacency_table.append(random.randint(0, 2 ** 64))
            self.zobrist_table.append(adjacency_table)

    # def initialize_zobrist_table(self):
    #     """
    #     Initialize the Zobrist table for the cell.
    #     """
    #     self.zobrist_table = []
    #     for i in range(self.ADJACENCY_MATRIX_SIZE):
    #         adjacency_table = []
    #         for connexion in range(2):  # Connexion or no connexion
    #             adjacency_table.append(random.randint(0, 2 ** 64))
    #         self.zobrist_table.append(adjacency_table)
    #     for i in range(self.N_NODES):
    #         operation_table = []
    #         for operation in range(self.N_OPERATIONS):
    #             operation_table.append(random.randint(0, 2 ** 64))
    #         self.zobrist_table.append(operation_table)

    def create_network(self) -> nn.Module:
        assert self.is_complete(), "Cell is incomplete. Cell must be complete to create a network."
        cell_str = self.to_str()
        self.network = NASBench201UNet(cell_str, 128, 1)
        self.network.to("cuda")
        self.ntk_network = NASBench201UNet_NTK(cell_str, 32, 3)
        self.ntk_network.to("cuda")

    def calculate_accuracy(self, dataset):
        if self.acc_metric == "ntk":
            score, _, _, _ = compute_score(self.ntk_network, dataset)
            return score
        elif isinstance(self.acc_metric, NASWOT):
            x = next(iter(dataset))[0].to("cuda")
            y = torch.tensor(next(iter(dataset))[1]).to("cuda")
            score = self.acc_metric.score(self.network)
            return score
        elif isinstance(self.acc_metric, ZenNAS):
            score = self.acc_metric.compute_nas_score(self.network, repeat=10, fp16=False)
            return score["avg_nas_score"]

    def calculate_latency(self, dataset):
        if self.lat_metric == "latency":
            import time
            x = next(iter(dataset))[0].to("cuda")
            # Warm-up
            for _ in range(10):
                _ = self.network(x)
            torch.cuda.synchronize()
            start_time = time.time()
            for _ in range(100):
                _ = self.network(x)
            torch.cuda.synchronize()
            end_time = time.time()
            latency = (end_time - start_time) / 100
            return latency
        if self.lat_metric == "n_param":
            n_param = sum([np.prod(p.size()) for p in self.network.parameters()])
            return n_param

    def get_multiobjective_reward(self, dataset, *args):
        """
        Get the multiobjective reward for the cell.

        Parameters:
        dataset (str): The dataset to be used for evaluation.
        model (str): The model to be used for evaluation.
        metric (str): The metric to be used for evaluation.
        objective (str): The objective to be used for evaluation.

        Returns:
        tuple: A tuple representing the multiobjective reward.
        """
        if self.network is None:
            self.create_network()
        dataset = self.dataset
        accuracy_metric = self.calculate_accuracy(dataset)
        latency_metric = self.calculate_latency(dataset)
        print(f"Cell: {self.to_str()}")
        print(f"Accuracy: {accuracy_metric}, Latency: {latency_metric}")
        return (accuracy_metric, latency_metric)

    def plot(self, filename="cell"):
        """
        Plot the cell using Graphviz.

        Parameters:
        filename (str): The filename for the output plot.
        """
        g = Digraph(
                format='pdf',
                edge_attr=dict(fontsize='20', fontname="garamond"),
                node_attr=dict(style='rounded, filled', shape='rect', align='center', fontsize='20', height='0.5',
                               width='0.5', penwidth='2', fontname="garamond"),
                engine='dot')
        g.body.extend(['rankdir=LR'])

        g.node("c_{k-1}", fillcolor='darkseagreen2')
        g.node("c_{k}", fillcolor='palegoldenrod')

        steps = self.n_vertices - 2

        for i in range(steps):
            g.node(str(i + 1), fillcolor='lightblue')

        for i, vertice in enumerate(self.vertices):
            for k, v in vertice.actions.items():
                # print(str(i), str(k), v)
                in_ = str(k)
                out_ = str(i)
                if k == 0:
                    in_ = "c_{k-1}"
                if i == self.n_vertices - 1:
                    out_ = "c_{k}"
                if v != "none":
                    g.edge(in_, out_, label=v, fillcolor="gray")
                else:
                    g.edge(in_, out_, label="", fillcolor="white", color="white")

        g.render(filename, view=True, format='pdf')

    def sample_random(self):
        while not self.is_complete():
            actions = self.get_action_tuples()
            random_action = np.random.randint(len(actions))
            action = actions[random_action]
            self.play_action(*action)

    def mutate(self):
        vertice = random.choice(range(1, self.n_vertices))
        id = random.choice(range(vertice))
        action = random.choice([op for op in self.OPERATIONS if op!=self.vertices[vertice].actions[id]])
        self.play_action(vertice, id, action)


class RadarProblem(ElementwiseProblem):

    def __init__(self, n_vertices):
        self.n_vertices = n_vertices
        n_var = n_vertices * (n_vertices - 1) // 2
        super().__init__(n_var=n_var,
                         n_obj=2,
                         xl=np.zeros(n_var),  # Lower bounds (operation index starts at 0)
                         xu=np.ones(n_var) * 5  # Upper bounds (max operation index = 3)
                         )

    def _evaluate(self, x, out, *args, **kwargs):
        print(x)
        cell = RadarCell(n_vertices=self.n_vertices)
        k = 0
        for i, vertice in enumerate(range(1, self.n_vertices)):
            for j in range(0, i+1):
                cell.play_action(vertice, j, cell.OPERATIONS[int(x[k])])
                k += 1
        reward = cell.get_multiobjective_reward(cell.dataset)
        out["F"] = (-reward[0], reward[1])

if __name__ == '__main__':
    cell = NASBench201NetworkCell(cell_str= '|nor_conv_3x3~0|+|nor_conv_3x3~0|nor_conv_3x3~1|+|nor_conv_3x3~0|nor_conv_3x3~1|nor_conv_1x1~2|+|nor_conv_3x3~0|skip_connect~1|avg_pool_3x3~2|skip_connect~3|+|avg_pool_3x3~0|skip_connect~1|nor_conv_3x3~2|nor_conv_3x3~3|none~4|',
                                  C_in=8, C_out=16, n_vertices=5)
    cell.forward(torch.randn(1, 8, 32, 32))