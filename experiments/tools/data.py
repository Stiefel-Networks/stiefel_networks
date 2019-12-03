import torch
import torchvision

from tools.util import get_gpu


# GPU loading code from
# https://medium.com/dsnet/training-deep-neural-networks-on-a-gpu-with-pytorch-11079d89805
class DeviceDataLoader:
    def __init__(self, data_loader, device):
        self.data_loader = data_loader
        self.device = device

    def __iter__(self):
        for batch in self.data_loader:
            yield to_device(batch, self.device)

    def __len__(self):
        return len(self.data_loader)


def to_device(data, device):
    if isinstance(data, (list, tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)


def get_tiny_mnist(batch_size=128, use_gpu=True):
    # Compose a bilinear downsample by a factor of 2, then downsample to final size.
    # This avoids some aliasing, because it's a box filter!
    downsample = torchvision.transforms.Compose([
        torchvision.transforms.Resize((14, 14)),
        torchvision.transforms.Resize((10, 10)),
        torchvision.transforms.ToTensor(),
    ])
    dataset = torchvision.datasets.MNIST(root='./data', download=True, transform=downsample)

    data_loader = torch.utils.data.DataLoader(dataset, batch_size, num_workers=16)

    if use_gpu:
        return DeviceDataLoader(data_loader, get_gpu())
    else:
        return data_loader
