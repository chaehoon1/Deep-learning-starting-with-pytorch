import torch

train_x = torch.FloatTensor([1, 2, 3, 4, 5])
train_y = torch.FloatTensor([3, 5, 7, 9, 11])

param_w = torch.FloatTensor([3])
param_b = torch.FloatTensor([2])

prediction = param_w * train_x + param_b  #y=wx+b

loss = torch.mean((prediction - train_y) ** 2)

model_w = torch.tensor([3], dtype = torch.float, requires_grad = True)
model_b = torch.tensor([2], dtype = torch.float, requires_grad = True)

prediction = model_w * train_x + model_b

loss = torch.mean((prediction - train_y) ** 2)

def gd(w ,b ,learning_data, epoch) :

