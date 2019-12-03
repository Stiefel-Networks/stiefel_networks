# import neur.exp as exp
import torch
import torch.nn.functional as F
from tools.neur.resnet import ResSVD
import matplotlib.pyplot as plt

num_batches = 3000
batch_size = 1000

model = ResSVD(1000, 2, 2, 1)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

def batchUnitBall(inputs):
    return (1 - (inputs.norm(dim=1) > 1).to(torch.float32) * 2).to(torch.float32)



for batch_index in range(num_batches):
    inputs = torch.randn(batch_size, 2)
    values = batchUnitBall(inputs)

    optimizer.zero_grad()

    # TODO Logan something in our fiddling broke this call.
    #      torch.meshgrid is getting a Tensor rather than a Number for the size n
    outputs = model(inputs)

    loss = F.mse_loss(outputs, values)
    outputs = outputs.flatten()
    values = values.flatten()
    print((outputs.sign()==values).sum())
    loss.backward()
    optimizer.step()

    print(loss.item())

    if batch_index%10==0:
        mask = (outputs.sign() == 1).squeeze()
        correct = inputs[mask]
        incorrect = inputs[~mask]
        plt.plot(correct[:,1],correct[:,0],'b.')
        plt.plot(incorrect[:,1],incorrect[:,0],'r.')
        plt.show()