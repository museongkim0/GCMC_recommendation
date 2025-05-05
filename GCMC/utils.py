import time
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Config(dict):
    def __init__(self, config):
        self._conf = config
 
    def __getattr__(self, name):
        if self._conf.get(name) is not None:
            return self._conf[name]
        return None

def stack(features, index, relations, dim_size):
    """
    Stack accumulation function in RGCLayer.

    Parameters
    ----------
    features : tensor (relation * num_nodes)
        output of messge method in RGCLayer class
    index : tensor (edges)
        edge_index[0]
    relations : teonsor(edges)
        edge_type
    dim_size : tensor(num_nodes)
        input size (the number of nodes)

    Return
    ------
    out : tensor(relation * nodes x out_dim)
    """
    out = torch.zeros(dim_size * (torch.max(relations) + 1), features.shape[1])
    tar_idx = relations * dim_size + index
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    out = out.to(device)
    for feature, idx in zip(features, tar_idx):
        out[idx] = out[idx] + feature

    return out
    

def ster_uniform(tensor, in_dim, out_dim):
    if tensor is not None:
        tensor.data.uniform_(-0.001, 0.001)


def random_init(tensor, in_dim, out_dim):
    thresh = math.sqrt(6.0 / (in_dim + out_dim))
    if tensor is not None:
        try:
            tensor.data.uniform_(-thresh, thresh)
        except:
            nn.init.uniform_(tensor, a=-thresh, b=thresh)


def init_xavier(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)
        try:
            truncated_normal(m.bias)
        except:
            pass

def init_uniform(m):
    if type(m) == nn.Linear:
        torch.nn.init.uniform_(m.weight)
        try:
            truncated_normal(m.bias)
        except:
            pass

def truncated_normal(tensor, mean=0, std=1):
    tensor.data.fill_(std * 2)
    with torch.no_grad():
        while(True):
            if tensor.max() >= std * 2:
                tensor[tensor>=std * 2] = tensor[tensor>=std * 2].normal_(mean, std)
                tensor.abs_()
            else:
                break

def calc_rmse(pred, gt):
    pred = F.softmax(pred, dim=1)
    expected_pred = torch.zeros(gt.shape)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    expected_pred = expected_pred.to(device)
    for relation in range(pred.shape[1]):
        expected_pred += pred[:, relation] * (relation + 1)

    rmse = (gt.to(torch.float) + 1) - expected_pred
    rmse = torch.pow(rmse, 2)
    rmse = torch.pow(torch.sum(rmse) / gt.shape[0], 0.5)

    return rmse

# 결과 출력 함수
def print_pred(pred, gt):
    pred = F.softmax(pred, dim=1)
    expected_pred = torch.zeros(gt.shape)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    expected_pred = expected_pred.to(device)
    for relation in range(pred.shape[1]):
        expected_pred += pred[:, relation] * (relation + 1)
    
    for value in zip(expected_pred, gt):
      print("predict value: {:.2f} | real value: {}".format(value[0],value[1]+1))

class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
