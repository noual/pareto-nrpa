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



def hooklogdet(K, labels=None):
    s, ld = np.linalg.slogdet(K)

    return ld


class NASWOT:

    def __init__(self, data_loader):
        self.data_loader = data_loader
        self.K = np.zeros((4, 4))

    def reset(self):
        self.K = np.zeros((4, 4))

    def score(self, network):
        def counting_forward_hook(module, inp, out):
            try:
                if not module.visited_backwards:
                    return
                if isinstance(inp, tuple):
                    inp = inp[0]
                inp = inp.view(inp.size(0), -1)
                x = (inp > 0).float()
                K = x @ x.t()
                K2 = (1. - x) @ (1. - x.t())
                self.K = self.K + K.cpu().numpy() + K2.cpu().numpy()

            except:
                pass

        def counting_backward_hook(module, inp, out):
            module.visited_backwards = True

        for name, module in network.named_modules():
            if 'ReLU' in str(type(module)):
                # hooks[name] = module.register_forward_hook(counting_hook)
                module.register_forward_hook(counting_forward_hook)
                module.register_backward_hook(counting_backward_hook)
        network = network.to("cuda")
        s = []
        for j in range(8):
            data_iterator = iter(self.data_loader)
            x, target = next(data_iterator)
            x2 = torch.clone(x)
            x2 = x2.to("cuda")
            x, target = x.to("cuda"), target.to("cuda")
            jacobs, labels, y = get_batch_jacobian(network, x, target, "cuda")
            network(x2.to("cuda"))
            s.append(hooklogdet(self.K, target))
            # else:
            #     s.append(hooklogdet(jacobs, labels))
        # scores[i] = np.mean(s)
        # accs[i] = searchspace.get_final_accuracy(uid, acc_type, args.trainval)
        # accs_ = accs[~np.isnan(scores)]
        # scores_ = scores[~np.isnan(scores)]
        # numnan = np.isnan(scores).sum()
        # tau, p = stats.kendalltau(accs_[:max(i - numnan, 1)], scores_[:max(i - numnan, 1)])
        # print(f'{tau}')
        return np.mean(s) / 100


def get_batch_jacobian(net, x, target, device, args=None):
    net.zero_grad()
    x.requires_grad_(True)
    y = net(x)
    y.backward(torch.ones_like(y))
    jacob = x.grad.detach()
    return jacob, target.detach(), y.detach()