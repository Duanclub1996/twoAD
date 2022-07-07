import torch.nn as nn
import torch


def score(input_data, dec_out1, dec_out2, a, b):
    rec_loss = nn.MSELoss(reduction='none')
    score = a * rec_loss(input_data, dec_out1) + b * rec_loss(input_data, dec_out2)

    return torch.mean(score, dim=-1)
