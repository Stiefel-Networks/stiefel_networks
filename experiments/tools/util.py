import torch


def get_best_device():
    if torch.cuda.is_available():
        return get_gpu()
    else:
        return get_cpu()


def get_cpu():
    return torch.device('cpu')


def get_gpu():
    return torch.device('cuda')