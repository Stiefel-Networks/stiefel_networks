import os
import json
import gc

import numpy as np
import matplotlib.pyplot as plt

from mnist_F_SVD.run import OUT_PATH
from tools.measurement import stable_rank


def plot_test_train_comparison(final_epochs, plot_directory=None):
    """
    metric is either 'loss' or 'accuracy'
    """

    fig = plt.figure(figsize=(6.4, 4.8))
    ax = fig.add_subplot(1, 1, 1)

    for hyperparameter_key in final_epochs:
        x = final_epochs[hyperparameter_key]['regularization_weights']
        test_perf = final_epochs[hyperparameter_key]['test_losses']
        train_perf = final_epochs[hyperparameter_key]['train_losses']

        test_lines = ax.plot(x, test_perf, label="Test Loss {}".format(hyperparameter_key), linestyle='dashed')
        ax.plot(x, train_perf, label="Train Loss {}".format(hyperparameter_key), color=test_lines[0].get_color())

    ax.set_xlim((-1e-8, 1))
    ax.set_xscale('symlog', linthreshx=1e-8)
    ax.set_ylim((0., 0.5))
    ax.set_title("Singular Value Regularization\nConverged Loss vs. Weight")
    ax.legend()

    if plot_directory is not None:
        plt.savefig(os.path.join(plot_directory, "regularization_comparison.pdf"))
    else:
        plt.show()

    plt.cla()
    plt.clf()
    plt.close(fig)


def main():
    run_files = [f for f in os.listdir(OUT_PATH) if os.path.isfile(os.path.join(OUT_PATH, f))]
    run_files.sort()

    final_epochs = {}
    for run_filname in run_files:
        with open(os.path.join(OUT_PATH, run_filname)) as file:
            # These files can be many megabytes, up to a few gigabytes.  Let the garbage collector
            # do its thing in advance of loading the next one.
            gc.collect()

            print("Loading {}...".format(run_filname))
            run_data = json.loads(file.read())

            print("Loaded:\n  {}".format(run_data["hyperparameters"]))

            hyperparameters = run_data["hyperparameters"]
            if 'stable_rank_weight' in hyperparameters:
                hyperparameter_key = "Stable Rank"
                weight_key = 'stable_rank_weight'
            elif 'l2_sigma_weight' in hyperparameters:
                hyperparameter_key = "L_2sigma"
                weight_key = 'l2_sigma_weight'
            else:
                continue



            if hyperparameter_key not in final_epochs:
                final_epochs[hyperparameter_key] = {
                    'regularization_weights': [],
                    'train_losses': [],
                    'test_losses': [],
                }

            final_epoch = run_data["epochs_progress"][-1]
            final_epochs[hyperparameter_key]['regularization_weights'].append(hyperparameters[weight_key])
            final_epochs[hyperparameter_key]['train_losses'].append(final_epoch["train_loss"])
            final_epochs[hyperparameter_key]['test_losses'].append(final_epoch["test_loss"])

    plot_directory = os.path.join(OUT_PATH, "cross_run")
    os.makedirs(plot_directory, exist_ok=True)

    # Uncomment to watch the plots roll in.
    # plot_directory = None
    plot_test_train_comparison(final_epochs, plot_directory)


if __name__ == "__main__":
    main()