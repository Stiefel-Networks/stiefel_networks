import os

import torch
import torchvision

from tools.util import get_gpu


DATA_DIR = os.path.dirname(os.path.realpath(__file__))


def get_downsampler():
    # Compose a bilinear downsample by a factor of 2, then downsample to final size.
    # This avoids some aliasing, because it's equivalent to a single box blur.
    return torchvision.transforms.Compose([
        torchvision.transforms.Resize((14, 14)),
        torchvision.transforms.Resize((10, 10)),
        torchvision.transforms.ToTensor(),
    ])


def get_tiny_mnist_test(use_gpu=True, test_mode=False):
    """
    Returns the standard test MNIST set, unshuffled with large batches, optionally gpu-ready.
    """
    test_dataset = _get_mnist(training=False, test_mode=test_mode)
    data_loader = torch.utils.data.DataLoader(
        test_dataset,
        1000,
        num_workers=8,
    )

    if use_gpu:
        return DeviceDataLoader(data_loader, get_gpu())
    else:
        return data_loader


def get_tiny_mnist(batch_size=32, use_gpu=True, test_mode=False):
    """
    Returns the training MNIST set shuffled, batched, and optionally gpu-ready.
    """
    train_dataset = _get_mnist(training=True, test_mode=test_mode)
    data_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size,
        num_workers=16,
        shuffle=True,
    )

    if use_gpu:
        return DeviceDataLoader(data_loader, get_gpu())
    else:
        return data_loader


def _get_mnist(training=True, test_mode=False):
    dataset = torchvision.datasets.MNIST(
        root=DATA_DIR,
        download=True,
        train=training,
        transform=get_downsampler(),
    )
    if test_mode:
        subset = list(range(1000))
        dataset = torch.utils.data.Subset(dataset, subset)
    return dataset


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
