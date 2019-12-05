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


def plot_test_train_comparison(final_epochs, network_name, plot_directory=None):
    """
    metric is either 'loss' or 'accuracy'
    """

    fig = plt.figure(figsize=(6.4, 4.8))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlim((0, 256))
    ax.set_ylim((0., 0.5))

    ax.plot(range(256), [0.1] * 256, label="Reference Line @ 0.1", linestyle='dotted', color='gray')

    for hyperparameter_key in final_epochs:
        x = final_epochs[hyperparameter_key]['layer_widths']
        test_perf = final_epochs[hyperparameter_key]['test_losses']
        train_perf = final_epochs[hyperparameter_key]['train_losses']

        test_lines = ax.plot(x, test_perf, label="Test Loss {}".format(hyperparameter_key), linestyle='dashed')
        ax.plot(x, train_perf, label="Train Loss {}".format(hyperparameter_key), color=test_lines[0].get_color())

    ax.set_title("{}\nConverged Loss vs. Width".format(
        network_name,
    ))
    ax.legend()

    if plot_directory is not None:
        plt.savefig(os.path.join(plot_directory, "generalization_comparison_{}.pdf".format(network_name)))
    else:
        plt.show()

    plt.cla()
    plt.clf()
    plt.close(fig)


def main():
    run_files = [f for f in os.listdir(OUT_PATH) if os.path.isfile(os.path.join(OUT_PATH, f))]
    run_files.sort()

    final_epochs = {'F_1': {}, 'F_SVD': {}}
    for run_filname in run_files:
        with open(os.path.join(OUT_PATH, run_filname)) as file:
            # These files can be many megabytes, up to a few gigabytes.  Let the garbage collector
            # do its thing in advance of loading the next one.
            gc.collect()

            print("Loading {}...".format(run_filname))
            run_data = json.loads(file.read())

            print("Loaded:\n  {}".format(run_data["hyperparameters"]))

            network_name = get_architecture(run_data)
            hyperparameters = run_data["hyperparameters"]
            hyperparameter_key = "b={} lr={} run {}".format(hyperparameters["batch_size"], hyperparameters["learning_rate"], hyperparameters["run_number"])

            if hyperparameter_key not in final_epochs[network_name]:
                final_epochs[network_name][hyperparameter_key] = {
                    'layer_widths': [],
                    'train_losses': [],
                    'test_losses': [],
                }

            final_epoch = run_data["epochs_progress"][-1]
            final_epochs[network_name][hyperparameter_key]['layer_widths'].append(hyperparameters["layer_width"])
            final_epochs[network_name][hyperparameter_key]['train_losses'].append(final_epoch["train_loss"])
            final_epochs[network_name][hyperparameter_key]['test_losses'].append(final_epoch["test_loss"])

    plot_directory = os.path.join(OUT_PATH, "cross_run")
    os.makedirs(plot_directory, exist_ok=True)

    # Uncomment to watch the plots roll in.
    # plot_directory = None
    plot_test_train_comparison(final_epochs["F_SVD"], "F_SVD", plot_directory)
    plot_test_train_comparison(final_epochs["F_1"], "F_1", plot_directory)


if __name__ == "__main__":
    main()