import argparse
import datetime
import json
import os

import torch
import torch.nn.functional as F
from tqdm import tqdm

from mnist_F_SVD.f_svd import FSVD
from tools.data import get_tiny_mnist, get_tiny_mnist_test
from tools.util import get_gpu


OUT_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'runs')


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


def record_shared(progress, run_start, epoch, samples_trained, f_svd_net):
    progress.update({
        'seconds_elapsed': (datetime.datetime.now() - run_start).seconds,
        'sample_exposure': samples_trained,
        'epoch': epoch,
        'singular_values': [values.tolist() for values in f_svd_net.singular_value_sets()],
    })


# args, run_start, epoch, batch, test_set, train_inputs, train_labels):
def record_epoch(progress_array, run_start, epoch, samples_trained, f_svd_net, use_gpu):
    test_set = get_tiny_mnist_test(use_gpu=use_gpu)
    train_set = get_tiny_mnist(batch_size=1000, use_gpu=use_gpu)

    train_loss, train_accuracy = evaluate_model(f_svd_net, train_set)
    test_loss, test_accuracy = evaluate_model(f_svd_net, test_set)

    print("Train: {:.1f}% @ L={:.3f}".format(train_accuracy, train_loss))
    print("Test:  {:.1f}% @ L={:.3f}".format(test_accuracy, test_loss))
    progress = {
        'train_loss': train_loss,
        'train_accuracy': train_accuracy,
        'test_loss': test_loss,
        'test_accuracy': test_accuracy,
    }
    record_shared(progress, run_start, epoch, samples_trained, f_svd_net)
    progress_array.append(progress)


# args, run_start, epoch, batch, batch_inputs, batch_labels, predictions):
def record_batch(progress_array, run_start, batches_completed, epoch, samples_trained, batch_accuracy, batch_loss, f_svd_net):
    progress = {
        'batches_completed': batches_completed,
        'batch_accuracy': batch_accuracy,
        'bach_loss': batch_loss,
    }
    record_shared(progress, run_start, epoch, samples_trained, f_svd_net)
    progress_array.append(progress)


def run_epoch(args, batches_progress, epoch, f_svd_net, optimizer, run_start, samples_trained):
    train_set = get_tiny_mnist(batch_size=args['batch_size'], use_gpu=args['use_gpu'])

    previous_losses = torch.zeros(10)
    with tqdm(desc="({}/{})".format(epoch, args['epochs']), total=len(train_set),
              bar_format="{l_bar}{bar}{r_bar}") as t:
        for batch, (inputs, labels) in enumerate(train_set, 1):
            optimizer.zero_grad()

            predictions = f_svd_net(inputs.reshape(-1, 100))
            loss = get_loss(labels, predictions)
            loss.backward()
            optimizer.step()

            # Record batch data
            samples_trained += len(inputs)
            accuracy = get_accuracy(inputs, labels, predictions)
            record_batch(batches_progress, run_start, batch, epoch, samples_trained, accuracy.item(), loss.item(), f_svd_net)

            # Update progress bar
            previous_losses[batch % 10] = loss
            loss_window_average = previous_losses.mean().item()
            t.postfix = "{}% @ L={:.3f}  (L_avg={:.3f})".format(accuracy.item(), loss.item(), loss_window_average)
            t.update()

    return samples_trained


def save_run_data(db, run_name, run_number, run_start):
    os.makedirs(OUT_PATH, exist_ok=True)
    run_timestamp = run_start.isoformat()
    out_filename = "{} - {} -  run {}.json".format(run_timestamp, run_name, run_number)
    with open(os.path.join(OUT_PATH, out_filename), 'w') as file:
        file.write(json.dumps(db, indent=2))


def main(hyperparams, run_name):
    args = {
        'batch_size': 64,
        'layer_width': 32,
        'learning_rate': 0.001,
        'epochs': 10,
        'num_runs': 5,
        'use_gpu': True,
    }
    args.update(hyperparams)

    for run_number in range(args['num_runs']):
        print("Starting run {} of {}".format(run_number, run_name))
        run_start = datetime.datetime.now()

        epochs_progress = []
        batches_progress = []
        db = {'hyperparameters': args, 'epochs_progress': epochs_progress, 'batches_progress': batches_progress}

        f_svd_net = FSVD(100, 10, layer_count=4, layer_width=args['layer_width'])
        if args['use_gpu']:
            f_svd_net = f_svd_net.to(device=get_gpu())

        optimizer = torch.optim.SGD(f_svd_net.parameters(), lr=args['learning_rate'], momentum=0.9)

        samples_trained = 0
        # Range and batch start at 1!
        epoch = 1
        for epoch in range(1, args['epochs'] + 1):
            record_epoch(epochs_progress, run_start, epoch, samples_trained, f_svd_net, args['use_gpu'])
            samples_trained = run_epoch(args, batches_progress, epoch, f_svd_net, optimizer, run_start, samples_trained)

        record_epoch(epochs_progress, run_start, epoch, samples_trained, f_svd_net, args['use_gpu'])
        save_run_data(db, run_name, run_number, run_start)


if __name__ == "__main__":

    # fast version for testing
    main({
        'batch_size': 5000,
        'layer_width': 32,
        'learning_rate': 0.01,
        'epochs': 2,
        'num_runs': 2,
        'use_gpu': True,
    }, 'fast_test')