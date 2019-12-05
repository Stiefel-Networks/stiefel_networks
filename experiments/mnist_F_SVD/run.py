import datetime
import json
import os

import torch
from tqdm import tqdm

# TODO rename module to 5_1_replication
from mnist_F_SVD.f_network import FNetwork
from tools.data import get_tiny_mnist, get_tiny_mnist_test
from tools.measurement import get_loss, get_accuracy, evaluate_model
from tools.util import get_gpu


OUT_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'runs')


def record_shared(progress, run_start, epoch, samples_trained, f_network):
    progress.update({
        'seconds_elapsed': (datetime.datetime.now() - run_start).seconds,
        'sample_exposure': samples_trained,
        'epoch': epoch,
        'singular_values': [values.tolist() for values in f_network.singular_value_sets()],
    })


def record_epoch(progress_array, run_start, epoch, samples_trained, f_network, test_accuracy, test_loss, train_accuracy, train_loss):
    print("Train: {:.1f}% @ L={:.3f}".format(train_accuracy, train_loss))
    print("Test:  {:.1f}% @ L={:.3f}".format(test_accuracy, test_loss))
    progress = {
        'train_loss': train_loss,
        'train_accuracy': train_accuracy,
        'test_loss': test_loss,
        'test_accuracy': test_accuracy,
    }
    record_shared(progress, run_start, epoch, samples_trained, f_network)
    progress_array.append(progress)

    return train_loss, test_loss


def evaluate_test_train(f_network, use_gpu, test_mode=False):
    test_set = get_tiny_mnist_test(use_gpu=use_gpu, test_mode=test_mode)
    train_set = get_tiny_mnist(batch_size=1000, use_gpu=use_gpu, test_mode=test_mode)

    train_loss, train_accuracy = evaluate_model(f_network, train_set)
    test_loss, test_accuracy = evaluate_model(f_network, test_set)

    return test_accuracy, test_loss, train_accuracy, train_loss


# args, run_start, epoch, batch, batch_inputs, batch_labels, predictions):
def record_batch(progress_array, run_start, batches_completed, epoch, samples_trained, batch_accuracy, batch_loss, f_network):
    progress = {
        'batches_completed': batches_completed,
        'batch_accuracy': batch_accuracy,
        'bach_loss': batch_loss,
    }
    record_shared(progress, run_start, epoch, samples_trained, f_network)
    progress_array.append(progress)


def run_epoch(args, batches_progress, epoch, f_network, optimizer, run_start, samples_trained, test_mode=False):
    train_set = get_tiny_mnist(batch_size=args['batch_size'], use_gpu=args['use_gpu'], test_mode=test_mode)

    previous_losses = torch.zeros(10)
    with tqdm(desc="({}/{})".format(epoch, args['epochs']), total=len(train_set),
              bar_format="{l_bar}{bar}{r_bar}") as t:
        for batch, (inputs, labels) in enumerate(train_set, 1):
            optimizer.zero_grad()

            predictions = f_network(inputs.reshape(-1, 100))
            loss = get_loss(labels, predictions)
            loss.backward()
            optimizer.step()

            # Record batch data
            samples_trained += len(inputs)
            accuracy = get_accuracy(inputs, labels, predictions)
            record_batch(batches_progress, run_start, batch, epoch, samples_trained, accuracy.item(), loss.item(), f_network)

            # Update progress bar
            previous_losses[batch % 10] = loss
            loss_window_average = previous_losses.mean().item()
            t.postfix = "{}% @ L={:.3f}  (L_avg={:.3f})".format(accuracy.item(), loss.item(), loss_window_average)
            t.update()

    return samples_trained


def save_run_data(db, run_name, run_start):
    os.makedirs(OUT_PATH, exist_ok=True)
    run_timestamp = run_start.isoformat()
    out_filename = "{} - {}.json".format(run_timestamp, run_name)
    with open(os.path.join(OUT_PATH, out_filename), 'w') as file:
        file.write(json.dumps(db, indent=2))


def perform_run(hyperparams, run_name, run_number):
    args = {
        'parametrization': 'svd',
        'batch_size': 128,
        'layer_width': 32,
        'learning_rate': 0.001,
        'epochs': 10,
        'train_loss_early_stop': 1e-6,
        'use_gpu': True,
        'test_mode': False,
    }
    args.update(hyperparams)

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
    save_run_data(db, run_name, run_start)
    print("Run duration: {} sec".format((datetime.datetime.now() - run_start).seconds))


def main():
    epochs = 100
    num_runs = 5
    batch_size = 128
    learning_rate = 0.01
    train_loss_early_stop = 0.01

    for run_number in range(num_runs):
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
                    'num_runs': num_runs,
                    'use_gpu': use_gpu,
                }, 'h={} b={} lr={} e={} [{}]'.format(layer_width, batch_size, learning_rate, epochs, parametrization), run_number)

    # Fast version for testing
    # perform_run({
    #     'parametrization': 'standard',
    #     'batch_size': 100,
    #     'layer_width': 32,
    #     'learning_rate': 0.01,
    #     'epochs': 5,
    #     'train_loss_early_stop': 0.001,
    #     'use_gpu': False,
    #     'test_mode': True,
    # }, 'fast_test', 1)


if __name__ == "__main__":
    main()