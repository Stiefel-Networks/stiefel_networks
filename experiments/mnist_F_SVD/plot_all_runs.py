import os
import json
import gc

import numpy as np
import matplotlib.pyplot as plt

from mnist_F_SVD.run import OUT_PATH
from tools.measurement import stable_rank


def get_architecture(run_data):
    if run_data["hyperparameters"]["parametrization"] == "svd":
        network_name = "F_SVD"
    elif run_data["hyperparameters"]["parametrization"] == "standard":
        network_name = "F_1"
    else:
        raise Exception("Unknown network parametrization {}".format(run_data["hyperparameters"]["parametrization"]))
    return network_name


def plot_epoch_test_train(run_data, metric, plot_directory=None):
    """
    metric is either 'loss' or 'accuracy'
    """
    epochs = run_data["epochs_progress"]
    x = list(range(len(epochs)))
    test_perf = [epoch["test_{}".format(metric)] for epoch in epochs]
    train_perf = [epoch["train_{}".format(metric)] for epoch in epochs]

    fig = plt.figure(figsize=(6.4, 4.8))
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(x, test_perf, label="Test {}".format(metric.capitalize()))
    ax.plot(x, train_perf, label="Train {}".format(metric.capitalize()))

    if metric == 'loss':
        ax.plot(x, [0.01 for _ in x], label="0.01", linestyle='dashed')
    else:
        ax.plot(x, [100 for _ in x], label="100%", linestyle='dashed')

    network_name = get_architecture(run_data)

    ax.set_ylim((0., 0.3))
    ax.set_title("Width {} {}\n{} vs. Epoch".format(
        run_data["hyperparameters"]["layer_width"],
        network_name,
        metric.capitalize(),
    ))
    ax.legend()

    if plot_directory is not None:
        plt.savefig(os.path.join(plot_directory, "test_train_{}.pdf".format(metric)))
    else:
        plt.show()

    plt.cla()
    plt.clf()
    plt.close(fig)


def plot_epoch_stable_rank(run_data, plot_directory=None):
    epochs = run_data["epochs_progress"]

    x = list(range(len(epochs)))

    layerwise_stable_ranks = []
    for epoch in epochs:
        for layer, layer_singular_values in enumerate(epoch["singular_values"]):
            if layer >= len(layerwise_stable_ranks):
                layerwise_stable_ranks.append([])
            layerwise_stable_ranks[layer].append(stable_rank(layer_singular_values))

    fig = plt.figure(figsize=(6.4, 4.8))
    ax = fig.add_subplot(1, 1, 1)
    for layer, layer_stable_ranks in enumerate(layerwise_stable_ranks, 1):
        ax.plot(x, layer_stable_ranks, label="Layer {}".format(layer))

    network_name = get_architecture(run_data)

    ax.set_title("Width {} {}\nStable Rank vs. Epoch".format(run_data["hyperparameters"]["layer_width"], network_name))
    ax.legend()

    if plot_directory is not None:
        plt.savefig(os.path.join(plot_directory, "stable_rank.pdf"))
    else:
        plt.show()

    plt.cla()
    plt.clf()
    plt.close(fig)


def plot_epoch_singular_value_heatmap(run_data, plot_directory=None):
    epochs = run_data["epochs_progress"]
    num_bins = 50

    layerwise_singular_values = []
    for epoch in epochs:
        for layer, layer_singular_values in enumerate(epoch["singular_values"]):
            if layer >= len(layerwise_singular_values):
                layerwise_singular_values.append([])
            layerwise_singular_values[layer].append(layer_singular_values)

    # We deal with lists because singular value arrays are not all the same size.
    fig, axes = plt.subplots(len(layerwise_singular_values), 1, figsize=(6.4, 10.))

    network_name = get_architecture(run_data)

    plt.subplots_adjust(top=0.92, bottom=0.05)
    fig.suptitle("Width {} {} - Singular Value Density vs. Epoch".format(run_data["hyperparameters"]["layer_width"], network_name))
    for layer, singular_values in enumerate(layerwise_singular_values):
        # Drop first row as all are ones due to orthogonal initialization, and transpose each so time is horizontal axis.
        singular_values = np.array(singular_values, dtype=float)[1:].T
        min_val = singular_values.min()
        max_val = singular_values.max()
        histogram_evolution = np.apply_along_axis(lambda slice: np.histogram(slice, bins=num_bins, range=(min_val, max_val))[0], 0, singular_values)

        # Construct y axis labels according to min and max singular value.
        num_y_labels = 5
        y_label_pixel_gap = int(num_bins / (num_y_labels - 1))
        y_label_pixel_positions = np.arange(0, num_bins, y_label_pixel_gap)
        y_labels = ["{:.2f}".format(label_value) for label_value in np.linspace(min_val, max_val, num_y_labels)]
        # Ensuring higher values are above lower ones - requires reversal due to imshow.
        axes[layer].set_yticks(y_label_pixel_positions[::-1])
        axes[layer].set_yticklabels(y_labels[::-1])

        axes[layer].set_title("Layer {}".format(layer + 1))
        axes[layer].imshow(histogram_evolution, aspect='auto')

    if plot_directory is not None:
        plt.savefig(os.path.join(plot_directory, "singular_value_heatmap.pdf"))
    else:
        plt.show()

    plt.cla()
    plt.clf()
    plt.close(fig)

def plot_singular_value_histograms(run_data, num_snapshots, plot_directory=None):
    epochs = run_data["epochs_progress"]
    num_bins = 15

    layerwise_singular_values = []
    epoch_numbers = []
    for snapshot in range(num_snapshots + 1):
        # Progress fractions should start at 0.0 and end at 1.0 exactly.  (Which is possible with float representation.)
        progress_fraction = snapshot / num_snapshots
        epoch_number = int(progress_fraction * len(epochs))
        epoch_number = min(epoch_number, len(epochs) - 1)
        epoch_numbers.append(epoch_number)

        epoch = epochs[epoch_number]
        for layer, layer_singular_values in enumerate(epoch["singular_values"]):
            if layer >= len(layerwise_singular_values):
                layerwise_singular_values.append([])
            layerwise_singular_values[layer].append(layer_singular_values)

    # Each row is a layer, each column is a snapshot
    fig, axes = plt.subplots(len(layerwise_singular_values), len(epoch_numbers), figsize=(12., 23.))

    network_name = get_architecture(run_data)

    plt.subplots_adjust(top=0.92, bottom=0.05)
    fig.suptitle("Width {} {} - Singular Value Histograms".format(run_data["hyperparameters"]["layer_width"], network_name))
    for layer, singular_value_snapshots in enumerate(layerwise_singular_values):
        for snapshot_number, (epoch_number, snapshot) in enumerate(zip(epoch_numbers, singular_value_snapshots)):
            # Drop first row as all are ones due to orthogonal initialization, and transpose each so time is horizontal axis.
            singular_values = np.array(snapshot, dtype=float)

            if np.min(singular_values) < 0.0 or np.max(singular_values) > 2.5:
                print("WARNING!\n Singular value outside of histogram range (0.0, 2.5):")
                print("Layer {} min, max:  ({}, {})".format(layer, np.min(singular_values), np.max(singular_values)))

            _n, _bins, _patches = axes[layer, snapshot_number].hist(singular_values, num_bins, range=(0.0, 2.5))

            axes[layer, snapshot_number].set_title("Layer {} Epoch {}".format(layer + 1, epoch_number))

    if plot_directory is not None:
        plt.savefig(os.path.join(plot_directory, "singular_value_histograms.pdf"))
    else:
        plt.show()

    plt.cla()
    plt.clf()
    plt.close(fig)


def main():
    run_files = [f for f in os.listdir(OUT_PATH) if os.path.isfile(os.path.join(OUT_PATH, f))]
    run_files.sort()
    for run_filname in run_files:
        with open(os.path.join(OUT_PATH, run_filname)) as file:
            # These files can be many megabytes, up to a few gigabytes.  Let the garbage collector
            # do its thing in advance of loading the next one.
            gc.collect()

            print("Loading {}...".format(run_filname))
            run_data = json.loads(file.read())

            print("Loaded:\n  {}".format(run_data["hyperparameters"]))

            run_subdirectory = os.path.splitext(os.path.basename(run_filname))[0]
            plot_directory = os.path.join(OUT_PATH, run_subdirectory)

            os.makedirs(plot_directory, exist_ok=True)

            # Uncomment to watch the plots roll in.
            # plot_directory = None
            plot_epoch_test_train(run_data, 'loss', plot_directory)
            plot_epoch_test_train(run_data, 'accuracy', plot_directory)
            plot_epoch_stable_rank(run_data, plot_directory)
            plot_singular_value_histograms(run_data, 4, plot_directory)
            plot_epoch_singular_value_heatmap(run_data, plot_directory)


if __name__ == "__main__":
    main()
