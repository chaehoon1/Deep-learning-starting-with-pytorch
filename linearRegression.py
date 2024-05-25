import torch

train_x = torch.FloatTensor([1, 2, 3, 4, 5])
train_y = torch.FloatTensor([3, 5, 7, 9, 11])

param_w = torch.FloatTensor([3])
param_b = torch.FloatTensor([2])

prediction = param_w * train_x + param_b
