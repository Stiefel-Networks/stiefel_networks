import matplotlib.pyplot as plt
import torch
import torchvision
import torch.nn.functional as F

from mnist_F_SVD.f_svd import FSVD


def get_best_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')

def to_device(data, device):
    if isinstance(data, (list, tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)


class DeviceDataLoader:
    def __init__(self, data_loader, device):
        self.data_loader = data_loader
        self.device = device

    def __iter__(self):
        for batch in self.data_loader:
            yield to_device(batch, self.device)

    def __len__(self):
        return len(self.data_loader)

def get_tiny_mnist(batch_size=128):
    # Compose a bilinear downsample by a factor of 2, then downsample to final size.
    # This avoids some aliasing, because it's a box filter!
    downsample = torchvision.transforms.Compose([
        torchvision.transforms.Resize((14, 14)),
        torchvision.transforms.Resize((10, 10)),
        torchvision.transforms.ToTensor(),
    ])
    dataset = torchvision.datasets.MNIST(root='./data', download=True, transform=downsample)

    data_loader = torch.utils.data.DataLoader(dataset, batch_size, num_workers=16)
    return DeviceDataLoader(data_loader, get_best_device())


if __name__ == "__main__":

    batch_size = 64
    tiny_mnist = get_tiny_mnist(batch_size=batch_size)

    f_svd_net = FSVD(100, 10, layer_count=4, layer_width=512).to(device=get_best_device())

    optimizer = torch.optim.SGD(f_svd_net.parameters(), lr=0.1, momentum=0.9)
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