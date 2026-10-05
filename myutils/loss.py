import torch

def attention_regularization_loss(G):
    # 控制注意力权重的均值接近 0.5
    mean_loss = torch.abs(G.mean() - 0.5)
    # 控制注意力权重的方差
    var_loss = torch.var(G)
    return mean_loss + var_loss
