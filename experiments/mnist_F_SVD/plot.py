import os
import json
import gc

import numpy as np
import matplotlib.pyplot as plt

from mnist_F_SVD.run import OUT_PATH
from tools.measurement import stable_rank


SMALL_FIGSIZE = 3., 2.
TALL_FIGSIZE = 3., 8.


def plot_epoch_test_train_loss(run_data, plot_directory=None):
    plt.rcParams['figure.figsize'] = SMALL_FIGSIZE

    epochs = run_data["epochs_progress"]
    x = [epoch["sample_exposure"] for epoch in epochs]
    test_loss = [epoch["test_loss"] for epoch in epochs]
    train_loss = [epoch["train_loss"] for epoch in epochs]

    plt.plot(x, test_loss, label="Test Loss")
    plt.plot(x, train_loss, label="Train Loss")

    plt.title("Width {} F_SVD\nLoss vs. Epoch".format(run_data["hyperparameters"]["layer_width"]))

    if plot_directory is not None:
        plt.savefig(os.path.join(plot_directory, "test_train_loss.pdf"))
    else:
        plt.show()


def plot_epoch_stable_rank(run_data, plot_directory=None):
    plt.rcParams['figure.figsize'] = SMALL_FIGSIZE

    epochs = run_data["epochs_progress"]

    x = [epoch["sample_exposure"] for epoch in epochs]

    layerwise_stable_ranks = []
    for epoch in epochs:
        for layer, layer_singular_values in enumerate(epoch["singular_values"]):
            if layer >= len(layerwise_stable_ranks):
                layerwise_stable_ranks.append([])
            layerwise_stable_ranks[layer].append(stable_rank(layer_singular_values))

    for layer, layer_stable_ranks in enumerate(layerwise_stable_ranks, 1):
        plt.plot(x, layer_stable_ranks, label="Layer {}".format(layer))

    plt.title("Width {} F_SVD\nStable Rank vs. Epoch".format(run_data["hyperparameters"]["layer_width"]))
    plt.legend()

    if plot_directory is not None:
        plt.savefig(os.path.join(plot_directory, "stable_rank.pdf"))
    else:
        plt.show()


def plot_epoch_singular_value_heatmap(run_data, plot_directory=None):
    plt.rcParams['figure.figsize'] = TALL_FIGSIZE

    epochs = run_data["epochs_progress"]
    num_bins = 50

    layerwise_singular_values = []
    for epoch in epochs:
        for layer, layer_singular_values in enumerate(epoch["singular_values"]):
            if layer >= len(layerwise_singular_values):
                layerwise_singular_values.append([])
            layerwise_singular_values[layer].append(layer_singular_values)

    # We deal with lists because singular value arrays are not all the same size.
    fig, axes = plt.subplots(len(layerwise_singular_values), 1, constrained_layout=True)
    fig.suptitle("Width {} F_SVD\nSingular Value Density vs. Epoch".format(run_data["hyperparameters"]["layer_width"]))
    for layer, singular_values in enumerate(layerwise_singular_values):
        # Drop first row as all are ones due to orthogonal initialization, and transpose each so time is horizontal axis.
        singular_values = np.array(singular_values, dtype=float)[1:].T
        min_val = singular_values.min()
        max_val = singular_values.max()
        histogram_evolution = np.apply_along_axis(lambda slice: np.histogram(slice, bins=num_bins, range=(min_val, max_val))[0], 0, singular_values)

        num_y_labels = 5
        y_label_pixel_gap = int(num_bins / (num_y_labels - 1))
        y_label_pixel_positions = np.arange(0, num_bins, y_label_pixel_gap)
        y_labels = ["{:.2f}".format(label_value) for label_value in np.linspace(min_val, max_val, num_y_labels)]
        axes[layer].set_yticks(y_label_pixel_positions)
        axes[layer].set_yticklabels(y_labels)

        axes[layer].set_title("Layer {}".format(layer))
        axes[layer].imshow(histogram_evolution)

    if plot_directory is not None:
        plt.savefig(os.path.join(plot_directory, "singular_value_heatmap.pdf"))
    else:
        plt.show()


def main():
    plt.rcParams['figure.dpi'] = 200

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
            plot_directory = os.path.join(OUT_PATH, run_subdirectory, "plots")

            os.makedirs(plot_directory, exist_ok=True)

            # plot_directory = None
            plot_epoch_test_train_loss(run_data, plot_directory)
            plot_epoch_stable_rank(run_data, plot_directory)
            plot_epoch_singular_value_heatmap(run_data, plot_directory)


if __name__ == "__main__":
    main()