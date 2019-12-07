import os
import json
import gc

import matplotlib.pyplot as plt
import numpy as np

from mnist_F_SVD.run import OUT_PATH
from tools.plotting import get_architecture, get_run_files


def plot_test_train_comparison(final_epochs, network_name, plot_directory=None):
    """
    metric is either 'loss' or 'accuracy'
    """

    fig = plt.figure(figsize=(6.4, 4.8))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xticks([8, 16, 32, 64, 128, 256])
    ax.set_xlim((0, 256))
    ax.set_ylim((0., 0.2))

    ax.axvline(64, label="N_T = 64", color='gray', linestyle='dashed', linewidth='1')

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
    final_epochs = {'F_1': {}, 'F_SVD': {}}
    for run_data, run_filname in get_run_files(OUT_PATH):
        # These files can be many megabytes, up to a few gigabytes.  Let the garbage collector
        # do its thing in advance of loading the next one.
        gc.collect()

        network_name = get_architecture(run_data)
        hyperparameters = run_data["hyperparameters"]
        hyperparameter_key = "b={} lr={} run {}".format(hyperparameters["batch_size"], hyperparameters["learning_rate"], hyperparameters["run_number"])

        if hyperparameter_key not in final_epochs[network_name]:
            final_epochs[network_name][hyperparameter_key] = {
                'layer_widths': [],
                'train_losses': [],
                'test_losses': [],
            }

        run_last_epochs = run_data["epochs_progress"][-3:]
        final_epochs[network_name][hyperparameter_key]['layer_widths'].append(hyperparameters["layer_width"])
        final_epochs[network_name][hyperparameter_key]['train_losses'].append(np.mean([epoch["train_loss"] for epoch in run_last_epochs]))
        final_epochs[network_name][hyperparameter_key]['test_losses'].append(np.mean([epoch["test_loss"] for epoch in run_last_epochs]))

    plot_directory = os.path.join(OUT_PATH, "cross_run")
    os.makedirs(plot_directory, exist_ok=True)

    # Uncomment to watch the plots roll in.
    # plot_directory = None
    plot_test_train_comparison(final_epochs["F_SVD"], "F_SVD", plot_directory)
    plot_test_train_comparison(final_epochs["F_1"], "F_1", plot_directory)


if __name__ == "__main__":
    main()