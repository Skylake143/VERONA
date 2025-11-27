
import argparse
from pathlib import Path

from ada_verona.database.experiment_repository import ExperimentRepository

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

DEFAULT_RESULT_CSV_NAME = "result_df.csv"

matplotlib.use("Agg")
sns.set_style("darkgrid")
sns.set_theme(rc={"figure.figsize": (11.7, 8.27)})
sns.set_palette(sns.color_palette("Paired"))

def create_hist_figure(df) -> plt.Figure:
        hist_plot = sns.histplot(data=df, x="epsilon_value", hue="network", multiple="stack")
        figure = hist_plot.get_figure()

        plt.close()

        return figure

def create_box_figure(df) -> plt.Figure:
    box_plot = sns.boxplot(data=df, x="network", y="epsilon_value")
    box_plot.set_xticklabels(box_plot.get_xticklabels(), rotation=90)
    box_plot.set_xlabel("Retrained Layers")


    figure = box_plot.get_figure()

    plt.close()

    return figure

def create_kde_figure(df) -> plt.Figure:
    kde_plot = sns.kdeplot(data=df, x="epsilon_value", hue="network", multiple="stack")

    figure = kde_plot.get_figure()

    plt.close()

    return figure

def create_ecdf_figure(df) -> plt.Figure:
    ecdf_plot = sns.ecdfplot(data=df, x="epsilon_value", hue="network")

    figure = ecdf_plot.get_figure()

    plt.close()

    return figure

def create_anneplot(df):
    for network in df.network.unique():
        df = df.sort_values(by="epsilon_value")
        cdf_x = np.linspace(0, 1, len(df))
        plt.plot(df.epsilon_value, cdf_x, label=network)
        plt.fill_betweenx(cdf_x, df.epsilon_value, df.smallest_sat_value, alpha=0.3)
        plt.xlim(0, 0.35)
        plt.xlabel("Epsilon values")
        plt.ylabel("Fraction critical epsilon values found")
        plt.legend()

    return plt.gca()

def main():
    parser = argparse.ArgumentParser(description='Robustness distribution script')
    parser.add_argument('--resultdf_path', type=str, default='examples/CNNYangBig/TransferLearned/results/pgd_27-11-2025+21_34/results', help='Source model path')
    args = parser.parse_args()

    results_path = Path(args.resultdf_path)

    result_df_path = results_path / DEFAULT_RESULT_CSV_NAME
    if result_df_path.exists():
        df = pd.read_csv(result_df_path, index_col=0)
    else:
        raise Exception(f"Error, no result file found at {result_df_path}")

    hist_figure = create_hist_figure(df)
    hist_figure.savefig(results_path / "hist_figure.png", bbox_inches="tight")

    boxplot = create_box_figure(df)
    boxplot.savefig(results_path / "boxplot.png", bbox_inches="tight")

    kde_figure = create_kde_figure(df)
    kde_figure.savefig(results_path / "kde_plot.png", bbox_inches="tight")

    ecdf_figure = create_ecdf_figure(df)
    ecdf_figure.savefig(results_path / "ecdf_plot.png", bbox_inches="tight")


if __name__ == "__main__":
    main()