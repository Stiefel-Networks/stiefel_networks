
import torch
import torch.nn.functional as F

from mnist_F_SVD.f_svd import FSVD
from tools.data import get_tiny_mnist
from tools.util import get_gpu

if __name__ == "__main__":

    batch_size = 32
    tiny_mnist = get_tiny_mnist(batch_size=batch_size, use_gpu=True)

    f_svd_net = FSVD(100, 10, layer_count=4, layer_width=512).to(device=get_gpu())

    optimizer = torch.optim.SGD(f_svd_net.parameters(), lr=0.001, momentum=0.9)
    for epoch in range(100):
        for batch, (inputs, labels) in enumerate(tiny_mnist):
            optimizer.zero_grad()

            predictions = f_svd_net(inputs.reshape(-1, 100))
            loss = F.cross_entropy(predictions, labels)

            if batch % 10 == 0:
                print("")
                print("{}, {}".format(epoch, (batch + 1) * batch_size))
                print(loss)
                print((torch.argmax(predictions, dim=1) == labels).sum() * 100 / batch_size)

            loss.backward()
            optimizer.step()

# TODO: store hyperparams, clock time, data exposure, batch number, epoch number, train loss, train accuracy, test loss, test accuracy, singular values