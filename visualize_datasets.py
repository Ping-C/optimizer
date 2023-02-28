import torch
import matplotlib.pyplot as plt
import argparse
from datasets import Kink
from matplotlib.colors import ListedColormap
import numpy as np
from sklearn.decomposition import PCA
from utils import *
import os


parser = argparse.ArgumentParser()
parser.add_argument('--output_folder')
args = parser.parse_args()

num_samples_list = [100000, 16]
seed_list = [0]

fig, axes = plt.subplots(len(seed_list),len(num_samples_list),  figsize=(len(num_samples_list)*5, len(seed_list)*5), squeeze=False)

for row_i, seed in enumerate(seed_list):
    for col_i, num_samples in enumerate(num_samples_list):
        train_data = torch.tensor(Kink(train=True, samples=num_samples, seed=seed, noise=None).data).float().cuda()
        train_labels = torch.tensor(Kink(train=True, samples=num_samples, seed=seed, noise=None).labels).long().cuda()

        cm_bright = ListedColormap(["#FF0000", "#0000FF"])
        x, y = train_data, train_labels
        axes[row_i, col_i].scatter(
            x[:, 0].cpu(), x[:, 1].cpu(), c=y.cpu(), cmap=cm_bright,
            edgecolors="k"
        )
        axes[row_i, col_i].set_xlim([-1, 1])
        axes[row_i, col_i].set_ylim([-1, 1])
        axes[row_i, col_i].set_axis_off()

os.makedirs(args.output_folder)

fig.savefig(os.path.join(args.output_folder, "dataset.png"))