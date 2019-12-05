import numpy as np
import torch
from torch.nn import functional as F

from tools.data import get_tiny_mnist_test, get_tiny_mnist


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


def evaluate_test_train(f_network, use_gpu, test_mode=False):
    test_set = get_tiny_mnist_test(use_gpu=use_gpu, test_mode=test_mode)
    train_set = get_tiny_mnist(batch_size=1000, use_gpu=use_gpu, test_mode=test_mode)

    train_loss, train_accuracy = evaluate_model(f_network, train_set)
    test_loss, test_accuracy = evaluate_model(f_network, test_set)

    return test_accuracy, test_loss, train_accuracy, train_loss