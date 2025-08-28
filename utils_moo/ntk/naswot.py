import numpy as np
import torch


# def get_batch_jacobian(net, x, target, device, split_data):
#     x.requires_grad_(True)
#
#     N = x.shape[0]
#     for sp in range(split_data):
#         st=sp*N//split_data
#         en=(sp+1)*N//split_data
#         y = net(x[st:en])
#         y.backward(torch.ones_like(y))
#
#     jacob = x.grad.detach()
#     x.requires_grad_(False)
#     return jacob, target.detach()
#
# def eval_score(jacob, labels=None):
#     corrs = np.corrcoef(jacob)
#     v, _  = np.linalg.eig(corrs)
#     k = 1e-5
#     return -np.sum(np.log(v + k) + 1./(v + k))
#
# def compute_jacob_cov(net, inputs, targets, split_data=1, loss_fn=None):
#     device = inputs.device
#     # Compute gradients (but don't apply them)
#     net.zero_grad()
#
#     jacobs, labels = get_batch_jacobian(net, inputs, targets, device, split_data=split_data)
#     jacobs = jacobs.reshape(jacobs.size(0), -1).cpu().numpy()
#     try:
#         jc = eval_score(jacobs, labels)
#     except Exception as e:
#         print(e)
#         jc = -1e2
#         # jc = np.nan
#     return jc

import numpy as np
import torch
from torch.utils.data import DataLoader


def hooklogdet(K, labels=None):
    s, ld = np.linalg.slogdet(K)

    return ld


# class NASWOT:
#
#     def __init__(self, data_loader):
#         self.data_loader = data_loader
#         self.K = np.zeros((4, 4))
#
#     def reset(self):
#         self.K = np.zeros((4, 4))
#
#     def score(self, network):
#         def counting_forward_hook(module, inp, out):
#             try:
#                 if not module.visited_backwards:
#                     return
#                 if isinstance(inp, tuple):
#                     inp = inp[0]
#                 inp = inp.view(inp.size(0), -1)
#                 x = (inp > 0).float()
#                 K = x @ x.t()
#                 K2 = (1. - x) @ (1. - x.t())
#                 self.K = self.K + K.cpu().numpy() + K2.cpu().numpy()
#
#             except:
#                 pass
#
#         def counting_backward_hook(module, inp, out):
#             module.visited_backwards = True
#
#         for name, module in network.named_modules():
#             if 'ReLU' in str(type(module)):
#                 # hooks[name] = module.register_forward_hook(counting_hook)
#                 module.register_forward_hook(counting_forward_hook)
#                 module.register_backward_hook(counting_backward_hook)
#         network = network.to("cuda")
#         s = []
#         for j in range(8):
#             data_iterator = iter(self.data_loader)
#             x, target = next(data_iterator)
#             x2 = torch.clone(x)
#             x2 = x2.to("cuda")
#             x, target = x.to("cuda"), target.to("cuda")
#             jacobs, labels, y = get_batch_jacobian(network, x, target, "cuda")
#             network(x2.to("cuda"))
#             s.append(hooklogdet(self.K, target))
#             # else:
#             #     s.append(hooklogdet(jacobs, labels))
#         # scores[i] = np.mean(s)
#         # accs[i] = searchspace.get_final_accuracy(uid, acc_type, args.trainval)
#         # accs_ = accs[~np.isnan(scores)]
#         # scores_ = scores[~np.isnan(scores)]
#         # numnan = np.isnan(scores).sum()
#         # tau, p = stats.kendalltau(accs_[:max(i - numnan, 1)], scores_[:max(i - numnan, 1)])
#         # print(f'{tau}')
#         return np.mean(s) / 100
#
#
# def get_batch_jacobian(net, x, target, device, args=None):
#     net.zero_grad()
#     x.requires_grad_(True)
#     y = net(x)
#     y.backward(torch.ones_like(y))
#     jacob = x.grad.detach()
#     return jacob, target.detach(), y.detach()

# class NASWOT:
#
#     def __init__(self, data_loader):
#         self.data_loader = DataLoader(data_loader, batch_size=64, shuffle=False)
#         self.inputs, self.targets = next(iter(self.data_loader))
#         self.inputs = self.inputs.to("cuda")
#         self.targets = self.targets.to("cuda")
#
#     def get_batch_jacobian(self, net, x, target):
#         net.zero_grad()
#
#         x.requires_grad_(True)
#
#         y = net(x)
#
#         y.backward(torch.ones_like(y))
#         jacob = x.grad.detach()
#
#         return jacob, target.detach()
#
#     def eval_score(self, jacob, labels=None):
#         corrs = np.corrcoef(jacob)
#         v, _ = np.linalg.eig(corrs)
#         k = 1e-5
#         score = np.sum(np.log(v + k) + 1.0 / (v + k))
#         return score
#
#     def score(self, net):
#         final_score = []
#         for i in range(1):
#             try:
#
#                 # Compute gradients (but don't apply them)
#                 jacobs, labels = self.get_batch_jacobian(net, self.inputs, self.targets)
#                 jacobs = jacobs.reshape(jacobs.size(0), -1).cpu().numpy()
#                 jc = self.eval_score(jacobs, labels)
#             except Exception as e:
#                 print(e)
#                 jc = 0
#             final_score.append(jc)
#
#
#         print(f"NASWOT score: {np.mean(final_score)} (with std {np.std(final_score)})")
#         return np.mean(final_score)  # Normalize the score

class NASWOT:

    def __init__(self, data_loader):
        self.data_loader = DataLoader(data_loader, batch_size=8, shuffle=False)
        self.inputs, self.targets = next(iter(self.data_loader))
        self.inputs = self.inputs.to("cuda")
        self.targets = self.targets.to("cuda")

    def score(self, net):
        batch_size = len(self.targets)

        def counting_forward_hook(module, inp, out):
            inp = inp[0].view(inp[0].size(0), -1)
            x = (inp > 0).float()  # binary indicator
            K = x @ x.t()
            K2 = (1. - x) @ (1. - x.t())
            net.K = net.K + K.cpu().numpy() + K2.cpu().numpy()  # hamming distance

        def counting_backward_hook(module, inp, out):
            module.visited_backwards = True

        net.K = np.zeros((batch_size, batch_size))
        for name, module in net.named_modules():
            module_type = str(type(module))
            if ('ReLU' in module_type) and ('naslib' not in module_type):
                # module.register_full_backward_hook(counting_backward_hook)
                module.register_forward_hook(counting_forward_hook)

        x = torch.clone(self.inputs)
        net(x)
        s, jc = np.linalg.slogdet(net.K)

        return np.clip(jc, -1e10, 1e10) / 100