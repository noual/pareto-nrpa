import os, sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from torch import nn
from torch.utils.data import DataLoader
import numpy as np

class ZenNAS:
    def __init__(self, data_loader):
        self.batch_size=8
        self.data_loader = DataLoader(data_loader, batch_size=self.batch_size, shuffle=False)
        self.mixup_gamma = 1e-2
        self.gpu = 0
        self.in_channels = 1
        self.resolution = 128

    def network_weight_gaussian_init(self, net: nn.Module):
        with torch.no_grad():
            for m in net.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.normal_(m.weight)
                    if hasattr(m, 'bias') and m.bias is not None:
                        nn.init.zeros_(m.bias)
                elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                    nn.init.ones_(m.weight)
                    nn.init.zeros_(m.bias)
                elif isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight)
                    if hasattr(m, 'bias') and m.bias is not None:
                        nn.init.zeros_(m.bias)
                else:
                    continue

        return net

    def compute_nas_score(self, model, repeat=1, fp16=False):
        info = {}
        nas_score_list = []
        if self.gpu is not None:
            device = torch.device('cuda:{}'.format(self.gpu))
        else:
            device = torch.device('cpu')

        if fp16:
            dtype = torch.half
        else:
            dtype = torch.float32

        with torch.no_grad():
            for repeat_count in range(repeat):
                self.network_weight_gaussian_init(model)
                input = torch.randn(size=[self.batch_size, self.in_channels, self.resolution, self.resolution], device=device, dtype=dtype)
                input2 = torch.randn(size=[self.batch_size, self.in_channels, self.resolution, self.resolution], device=device, dtype=dtype)
                mixup_input = input + self.mixup_gamma * input2
                output = model.forward(input)
                mixup_output = model.forward(mixup_input)

                nas_score = torch.sum(torch.abs(output - mixup_output), dim=[1, 2, 3])
                nas_score = torch.mean(nas_score)

                # compute BN scaling
                log_bn_scaling_factor = 0.0
                for m in model.modules():
                    if isinstance(m, nn.BatchNorm2d):
                        bn_scaling_factor = torch.sqrt(torch.mean(m.running_var))
                        log_bn_scaling_factor += torch.log(bn_scaling_factor)
                    pass
                pass
                nas_score = torch.log(nas_score) + log_bn_scaling_factor
                nas_score_list.append(float(nas_score))


        std_nas_score = np.std(nas_score_list)
        avg_precision = 1.96 * std_nas_score / np.sqrt(len(nas_score_list))
        avg_nas_score = np.mean(nas_score_list)


        info['avg_nas_score'] = float(avg_nas_score)
        info['std_nas_score'] = float(std_nas_score)
        info['avg_precision'] = float(avg_precision)
        return info

if __name__ == '__main__':
    net = UNet(1, 1).to("cuda:0")
    nas = ZenNAS(None)
    info = nas.compute_nas_score(net, repeat=10, fp16=False)
    print(f"Unet : {info}")
    accurate  = NASBench201UNet('|nor_conv_3x3~0|+|nor_conv_3x3~0|nor_conv_3x3~1|+|nor_conv_1x1~0|dep_conv_3x3~1|nor_conv_1x1~2|+|none~0|nor_conv_1x1~1|dep_conv_3x3~2|nor_conv_1x1~3|',
                            input_size=128, input_depth=1, n_vertices=5)
    info = nas.compute_nas_score(accurate.to("cuda:0"), repeat=10, fp16=False)
    print(f"Accurate NASBench201Unet : {info}")
    efficient = NASBench201UNet(
        '|dep_conv_3x3~0|+|none~0|none~1|+|none~0|nor_conv_3x3~1|none~2|+|avg_pool_3x3~0|avg_pool_3x3~1|nor_conv_1x1~2|skip_connect~3|',
        input_size=128, input_depth=1, n_vertices=5)
    info = nas.compute_nas_score(efficient.to("cuda:0"), repeat=10, fp16=False)
    print(f"Efficient NASBench201Unet : {info}")
    lightweight = NASBench201UNet(
        '|none~0|+|skip_connect~0|skip_connect~1|+|skip_connect~0|none~1|avg_pool_3x3~2|+|dep_conv_3x3~0|dep_conv_3x3~1|none~2|none~3|',
        input_size=128, input_depth=1, n_vertices=5)
    info = nas.compute_nas_score(lightweight.to("cuda:0"), repeat=10, fp16=False)
    print(f"Lightweight NASBench201Unet : {info}")