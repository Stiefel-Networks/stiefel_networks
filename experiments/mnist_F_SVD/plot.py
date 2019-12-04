import os
import json
import gc

import numpy as np
import matplotlib.pyplot as plt

from mnist_F_SVD.run import OUT_PATH


def plot_epoch_test_loss(run_data):
    epochs = run_data["epochs_progress"]
    x = [epoch["sample_exposure"] for epoch in epochs]
    y = [epoch["test_loss"] for epoch in epochs]

    plt.plot(x, y)
    plt.title(run_data["hyperparameters"]["layer_width"])
    plt.show()


def plot_epoch_singular_value_heatmap(run_data):
    epochs = run_data["epochs_progress"]

    layerwise_singular_values = []
    for epoch in epochs:
        for layer, layer_singular_values in enumerate(epoch["singular_values"]):
            if layer >= len(layerwise_singular_values):
                layerwise_singular_values.append([])
            layerwise_singular_values[layer].append(layer_singular_values)

    # Make top level a list because singular value arrays are not all the same size.
    for layer, singular_values in enumerate(layerwise_singular_values):
        # Transpose each so time is horizontal axis, and drop first column as all are ones due to orthogonal initialization
        singular_values = np.array(singular_values, dtype=float).T[1:]
        min_val = singular_values.min()
        max_val = singular_values.max()
        histogram_evolution = np.apply_along_axis(lambda slice: np.histogram(slice, bins=100, range=(min_val, max_val))[0], 0, singular_values)

        # TODO fig, axs, larger size
        # plt.subplot(len(layerwise_singular_values), 1, layer + 1)
        plt.imshow(histogram_evolution)
        plt.title("Width {}, Layer {}".format(run_data["hyperparameters"]["layer_width"], layer))

    plt.show()


def main():
    run_files = [f for f in os.listdir(OUT_PATH) if os.path.isfile(os.path.join(OUT_PATH, f))]
    for run_filname in run_files:
        with open(os.path.join(OUT_PATH, run_filname)) as file:
            # These files can be many megabytes, up to a few gigabytes.  Let the garbage collector
            # do its thing in advance of loading the next one.
            gc.collect()

            print("Loading {}...".format(run_filname))
            run_data = json.loads(file.read())

            print("Loaded:\n  {}".format(run_data["hyperparameters"]))

            plot_epoch_singular_value_heatmap(run_data)


if __name__ == "__main__":
    main()