import seaborn as sns
import pandas as pd

from tools.ortho.metrics import mean_cosine_similarity
from tools.ortho.matrix import random_square_matrix


def main():
    min_size = 2
    max_size = 50
    matrix_sizes = range(min_size, max_size + 1)
    matrix_samples_per_size = 2000

    mean_cosine_similarities = []
    for matrix_size in matrix_sizes:
        for run in range(0, matrix_samples_per_size):
            matrix = random_square_matrix(matrix_size)

            mean_cosine_similarities.append({
                "Matrix Size": matrix_size,
                "Mean Cosine Similarity": mean_cosine_similarity(matrix),
            })

    cosine_similarity_data = pd.DataFrame(mean_cosine_similarities)
    sns.set_style('darkgrid')
    cosine_similarity_plot = sns.lineplot(
        x="Matrix Size",
        y="Mean Cosine Similarity",
        data=cosine_similarity_data,
    )

    plot_title = "Mean Cosine Similarity of random matrix columns vs. matrix size\n" +\
                 "{0} matrices sampled for each size from {1}x{1} to {2}x{2}.".format(
                     matrix_samples_per_size,
                     min_size,
                     max_size,
                 )
    cosine_similarity_plot.set_title(plot_title)
    cosine_similarity_plot.get_figure().savefig("cosine_similarity.pdf")


if __name__ == "__main__":
    main()
