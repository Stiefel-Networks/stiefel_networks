import matplotlib.pyplot as plt
import torch
import torchvision
import torch.nn.functional as F

from mnist_F_SVD.f_svd import FSVD

def get_tiny_mnist(batch_size=128):
    # Compose a bilinear downsample by a factor of 2, then downsample to final size.
    # This avoids some aliasing, because it's a box filter!
    downsample = torchvision.transforms.Compose([
        torchvision.transforms.Resize((14, 14)),
        torchvision.transforms.Resize((10, 10)),
        torchvision.transforms.ToTensor(),
    ])
    dataset = torchvision.datasets.MNIST(root='./data', download=True, transform=downsample)

    return torch.utils.data.DataLoader(dataset, batch_size, num_workers=16)


if __name__ == "__main__":

    batch_size = 64
    tiny_mnist = get_tiny_mnist(batch_size=batch_size)

    f_svd_net = FSVD(100, 10, layer_count=4, layer_width=32)
    optimizer = torch.optim.SGD(f_svd_net.parameters(), lr=0.1, momentum=0.9)
    for epoch in range(100):
        for batch, (inputs, labels) in enumerate(tiny_mnist):
            optimizer.zero_grad()

            predictions = f_svd_net(inputs.reshape(-1, 100))
            loss = F.cross_entropy(predictions, labels)

            if batch % 100 == 0:
                print("")
                print("{}, {}".format(epoch, (batch + 1) * batch_size))
                print(loss)
                print((torch.argmax(predictions, dim=1) == labels).sum() * 100 / batch_size)

            loss.backward()
            optimizer.step()