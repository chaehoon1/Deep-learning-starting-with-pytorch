import torch

train_x = torch.FloatTensor([1, 2, 3, 4, 5])
train_y = torch.FloatTensor([3, 5, 7, 9, 11])

param_w = torch.tensor([3], dtype = torch.float, requires_grad = True)
param_b = torch.tensor([2], dtype = torch.float, requires_grad = True)

prediction = train_x * param_w + param_b
print(prediction)

loss_fn = torch.mean((prediction - train_y) ** 2)
loss_fn.backward()
print(param_w.grad) # x = 2, y = 5인 경우 : loss = (2w + b - 5)^2에서 loss를 w에 대해 편미분 후 w = 3, b = 2 대입...모든 (x, y)에 대해 수행 후 평균
print(param_b.grad) # x = 2, y = 5인 경우 : loss = (2w + b - 5)^2에서 loss를 b에 대해 편미분 후 w = 3, b = 2 대입...모든 (x, y)에 대해 수행 후 평균
param_w.grad.data.zero_()
param_b.grad.data.zero_()

def optimizer(train_x, train_y, param_w, param_b, learning_rate, epoch) :
    for i in  range(epoch) :    
        prediction = param_w * train_x + param_b
        loss_fn = torch.mean((prediction - train_y) ** 2)
        
        loss_fn.backward()

        param_w.data -= learning_rate * param_w.grad
        param_b.data -= learning_rate * param_b.grad
       
        if (i + 1) % 10 == 0 :    
            print(f'Epoch [{i + 1}/100], Loss: {loss_fn.item():.4f}')
            print('prediction : ', prediction)
            print('param_w : ', param_w)
            print('param_b : ', param_b)

        param_w.grad.data.zero_()
        param_b.grad.data.zero_()

optimizer(train_x, train_y, param_w, param_b, 0.01, 1000)
