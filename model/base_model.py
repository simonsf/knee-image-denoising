import os
import torch
import torch.nn as nn


class BaseModel():
    # 定义一些基础类别：参数、GPU、开始的step以及epoch
    def __init__(self, opt):
        self.opt = opt
        self.device = torch.device(
            'cuda' if opt['gpu_ids'] is not None else 'cpu')
        self.begin_step = 0
        self.begin_epoch = 0

    def feed_data(self, data):
        pass

    def optimize_parameters(self):
        pass

    def get_current_visuals(self):
        pass

    def get_current_losses(self):
        pass

    def print_network(self):
        pass

    # 将数据移动到CUDA
    def set_device(self, x):
        if isinstance(x, dict):
            for key, item in x.items():
                if item is not None:
                    x[key] = item.to(self.device)
        elif isinstance(x, list):
            for item in x:
                if item is not None:
                    item = item.to(self.device)
        else:
            x = x.to(self.device)
        return x

    # 得到一个神经网络的结构描述以及需要训练的参数量
    def get_network_description(self, network):
        '''Get the string and total parameters of the network'''
        if isinstance(network, nn.DataParallel):
            # network.module获取实际的网络模型对象
            network = network.module
        # 将网络模型转换为字符串
        s = str(network)
        # 计算参数量
        n = sum(map(lambda x: x.numel(), network.parameters()))
        return s, n
