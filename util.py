import torch
import os
import sys
import pickle
import numpy as np

from loss import EDL_Loss


def NPArrayList2PyTorchTensorList(input):
    res = []
    for item_array in input:
        item_tensor = torch.Tensor(item_array).cuda()
        res.append(item_tensor)
    return res


def PyTorchParameterList2NPArrayList(input):
    res = []
    for item_tensor in input:
        item_array = item_tensor.detach().cpu().clone().numpy()
        res.append(item_array)
    return res


def Predict(X, model):
    if type(X) is not torch.Tensor:
        X = torch.Tensor(X).long().cuda()
    elif X.is_cuda is False:
        X = X.cuda()

    with torch.no_grad():
        pred, _ = model(X)

    return pred.cpu().clone().numpy()


def GetEDLLoss(pred, true):
    if type(pred) is not torch.Tensor:
        pred = torch.Tensor(pred).float().cuda()
    elif pred.is_cuda is False:
        pred = pred.cuda()

    if type(true) is not torch.Tensor:
        true = torch.Tensor(true).float().cuda()
    elif true.is_cuda is False:
        true = true.cuda()

    loss = EDL_Loss()(true, pred).item()

    return loss

def Load_pretrained_embeddings(file_path):
    pretrained_fpath_saved = os.path.expanduser(file_path.format(sys.version_info.major))

    with open(pretrained_fpath_saved, 'rb') as f:
        embedding_weights = pickle.load(f)
    f.close()

    out = np.array(list(embedding_weights.values()))
    print('embedding_weights shape:', out.shape)
    return out


def ExpConfigurations():
    out_dim = 6
    model_idx = 1000
    niterations = 1000

    return out_dim, model_idx, niterations
