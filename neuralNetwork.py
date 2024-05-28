import torch

x, y =get_data()
w, b = get_weights()
for i in range(50000) :
    y_pred = simple_network(x)
    loss = loss_fn(y, y_pred)
    if i % 5000 == 0:
        print(loss)
    optimize(learning_rate)

