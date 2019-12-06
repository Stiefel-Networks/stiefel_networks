import datetime
import os

import torch
from tqdm import tqdm

# TODO rename module to 5_1_replication
from mnist_F_SVD.f_network import FNetwork
from tools.data import get_tiny_mnist
from tools.measurement import get_loss, get_accuracy, evaluate_test_train
from tools.recording import record_epoch, record_batch, save_run_data
from tools.util import get_gpu


OUT_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'runs')


def run_epoch(args, batches_progress, epoch, f_network, optimizer, run_start, samples_trained, test_mode=False):
    train_set = get_tiny_mnist(batch_size=args['batch_size'], use_gpu=args['use_gpu'], test_mode=test_mode)

    previous_losses = torch.zeros(10)
    with tqdm(desc="({}/{})".format(epoch, args['epochs']), total=len(train_set),
              bar_format="{l_bar}{bar}{r_bar}") as t:
        for batch, (inputs, labels) in enumerate(train_set, 1):
            optimizer.zero_grad()

            predictions = f_network(inputs.reshape(-1, 100))
            loss = get_loss(labels, predictions)

            if 'l2_sigma_weight' in args:
                loss = loss + l2_sigma_regularizer(args['l2_sigma_weight'], f_network)
            if 'stable_rank_weight' in args:
                loss = loss + stable_rank_regularizer(args['stable_rank_weight'], f_network)

            loss.backward()
            optimizer.step()

            # Record batch data
            samples_trained += len(inputs)
            accuracy = get_accuracy(inputs, labels, predictions)
            record_batch(batches_progress, run_start, batch, epoch, samples_trained, accuracy.item(), loss.item())

            # Update progress bar
            previous_losses[batch % 10] = loss
            loss_window_average = previous_losses.mean().item()
            t.postfix = "{}% @ L={:.3f}  (L_avg={:.3f})".format(accuracy.item(), loss.item(), loss_window_average)
            t.update()

    return samples_trained


def l2_sigma_regularizer(weight, f_network):
    singular_values_sets = f_network.singular_value_sets()
    regularization_term = 0
    for singular_values_set in singular_values_sets:
        regularization_term += (torch.norm(singular_values_set) ** 2) * weight

    return regularization_term


def stable_rank_regularizer(weight, f_network):
    singular_values_sets = f_network.singular_value_sets()
    regularization_term = 0
    for singular_values_set in singular_values_sets:
        regularization_term += (torch.norm(singular_values_set) ** 2) / (torch.max(singular_values_set) ** 2) * weight

    return regularization_term


def perform_run(hyperparams, run_name):
    args = {
        'parametrization': 'svd',
        'batch_size': 128,
        'layer_width': 32,
        'learning_rate': 0.001,
        'epochs': 10,
        'train_loss_early_stop': 1e-6,
        'run_number': 0,
        'use_gpu': True,
        'test_mode': False,
    }
    args.update(hyperparams)

    run_number = args['run_number']
    print("Starting run {} of {}".format(run_number, run_name))
    run_start = datetime.datetime.now()

    epochs_progress = []
    batches_progress = []
    db = {'hyperparameters': args, 'epochs_progress': epochs_progress, 'batches_progress': batches_progress}

    f_network = FNetwork(100, 10, layer_count=4, layer_width=args['layer_width'], parametrization=args['parametrization'])
    if args['use_gpu']:
        f_network = f_network.to(device=get_gpu())

    optimizer = torch.optim.SGD(f_network.parameters(), lr=args['learning_rate'], momentum=0.9)

    samples_trained = 0
    # Range and batch start at 1!
    epoch = 1
    test_accuracy, test_loss, train_accuracy, train_loss = evaluate_test_train(
        f_network,
        args['use_gpu'],
        args['test_mode']
    )
    for epoch in range(1, args['epochs'] + 1):
        if train_loss <= args['train_loss_early_stop']:
            print("Stopping run, train loss threshold {} exceeded:".format(args['train_loss_early_stop']))
            break

        record_epoch(
            epochs_progress,
            run_start,
            epoch,
            samples_trained,
            f_network,
            test_accuracy,
            test_loss,
            train_accuracy,
            train_loss,
        )

        samples_trained = run_epoch(
            args,
            batches_progress,
            epoch,
            f_network,
            optimizer,
            run_start,
            samples_trained,
            test_mode=args['test_mode'],
        )

        test_accuracy, test_loss, train_accuracy, train_loss = evaluate_test_train(
            f_network,
            args['use_gpu'],
            args['test_mode']
        )

    record_epoch(
        epochs_progress,
        run_start,
        epoch,
        samples_trained,
        f_network,
        test_accuracy,
        test_loss,
        train_accuracy,
        train_loss,
    )
    save_run_data(db, OUT_PATH, run_name, run_start, run_number)
    print("Run duration: {} sec".format((datetime.datetime.now() - run_start).seconds))


def run_experiment_5_1():
    epochs = 100
    batch_size = 128
    learning_rate = 0.01
    train_loss_early_stop = 0.01

    for run_number in [1, 2]:
        for parametrization in ['standard', 'svd']:
            for layer_width in [64, 128, 256]:
                # Run on CPU up to width 128, as it's faster
                use_gpu = layer_width >= 256

                perform_run({
                    'parametrization': parametrization,
                    'batch_size': batch_size,
                    'layer_width': layer_width,
                    'learning_rate': learning_rate,
                    'epochs': epochs,
                    'train_loss_early_stop': train_loss_early_stop,
                    'run_number': run_number,
                    'use_gpu': use_gpu,
                }, 'h={} b={} lr={} e={} [{}]'.format(layer_width, batch_size, learning_rate, epochs, parametrization))


def run_experiment_5_2():
    epochs = 20
    batch_size = 128
    train_loss_early_stop = 0.01
    parametrization = 'svd'

    learning_rate = 0.01
    layer_width = 128

    for run_number in [0, 1]:
        for regularization_weight in [0.00001, 0.0001, 0.001, 0.01, 0.1]:
            # Run on CPU up to width 128, as it's faster
            use_gpu = layer_width >= 256

            perform_run({
                'parametrization': parametrization,
                'batch_size': batch_size,
                'layer_width': layer_width,
                'learning_rate': learning_rate,
                'stable_rank_weight': regularization_weight,
                'epochs': epochs,
                'train_loss_early_stop': train_loss_early_stop,
                'run_number': run_number,
                'use_gpu': use_gpu,
            }, 'h={} b={} lr={} e={} l_SR={} [{} 5.2]'.format(
                layer_width,
                batch_size,
                learning_rate,
                epochs,
                regularization_weight,
                parametrization
            ))

            perform_run({
                'parametrization': parametrization,
                'batch_size': batch_size,
                'layer_width': layer_width,
                'learning_rate': learning_rate,
                'l2_sigma_weight': regularization_weight,
                'epochs': epochs,
                'train_loss_early_stop': train_loss_early_stop,
                'run_number': run_number,
                'use_gpu': use_gpu,
            }, 'h={} b={} lr={} e={} l2_SQ={} [{} 5.2]'.format(
                layer_width,
                batch_size,
                learning_rate,
                epochs,
                regularization_weight,
                parametrization
            ))

def run_test_experiment():
    # Fast version for testing
    perform_run({
        'parametrization': 'standard',
        'batch_size': 100,
        'layer_width': 32,
        'learning_rate': 0.01,
        'epochs': 5,
        'train_loss_early_stop': 0.001,
        'run_number': 1,
        'use_gpu': False,
        'test_mode': True,
    }, 'fast_test')


if __name__ == "__main__":
    run_experiment_5_2()