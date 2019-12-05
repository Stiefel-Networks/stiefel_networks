import numpy as np
import torch
from torch.nn import functional as F


def get_loss(labels, predictions):
    return F.cross_entropy(predictions, labels)


def get_accuracy(inputs, labels, predictions):
    return (torch.argmax(predictions, dim=1) == labels).sum() * 100 / inputs.shape[0]


def evaluate_model(f_svd_net, data_set):
    total_samples = 0
    total_loss = 0
    total_accuracy = 0
    for inputs, labels in data_set:
        total_samples += len(inputs)

        predictions = f_svd_net(inputs.reshape(-1, 100))
        loss = get_loss(labels, predictions)
        accuracy = get_accuracy(inputs, labels, predictions)

        total_loss += len(inputs) * loss.item()
        total_accuracy += len(inputs) * accuracy.item()

    return total_loss / total_samples, total_accuracy / total_samples


def stable_rank(layer_singular_values):
    layer_singular_values = np.array(layer_singular_values, dtype=float)
    layer_singular_values.sort()
    layer_singular_values = layer_singular_values[::-1]
    frobenius = np.sum(layer_singular_values ** 2)
    spectral = np.max(layer_singular_values) ** 2
    return frobenius / spectral