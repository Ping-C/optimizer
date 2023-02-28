import torch
import umap
import matplotlib.pyplot as plt
import argparse
from datasets import Kink
from matplotlib.colors import ListedColormap
import numpy as np
from sklearn.decomposition import PCA
import umap
from utils import *
from train_distributed import get_dataset
from collections import defaultdict


parser = argparse.ArgumentParser(description="This file loads the learned slab model and put thems into robust vs linear bins")
parser.add_argument('--model_folder', default='output/figure3/slab_guess/models', help="model folder name")


args = parser.parse_args()

models_list = []
valid_model_filenames = []

# load model
import os
models_path = args.model_folder
for model_filename in os.listdir(models_path):
    model_filename = os.path.join(models_path, model_filename)
    try:
        models_dict = torch.load(model_filename)
    except:
        continue
    valid_model_filenames.append(model_filename)
    models = MLPModels(**models_dict["kwargs"], device=torch.device("cpu"))
    hidden_units = models_dict["kwargs"]["hidden_units"]
    models.load_state_dict(models_dict["good_models_state_dict"])

    model_count = models.model_count
    model_configs_dict = models_dict["config"]
    model_configs_dict = defaultdict(lambda : None, model_configs_dict)

    models_list.append(models)

from datasets import SlabLinear, SlabNonlinear4
slabnonlinear = SlabNonlinear4()
data_nonlinear = torch.tensor(slabnonlinear.data, dtype=torch.float)
labels_nonlinear = torch.tensor(slabnonlinear.labels)
slablinear = SlabLinear()
data_linear = torch.tensor(slablinear.data, dtype=torch.float)
labels_linear =torch.tensor(slablinear.labels)

nonlinear_model_count = 0
linear_model_count = 0
for i, model in enumerate(models_list):
    if i % 1000 == 0:
        print(i)
    pred = model(data_nonlinear).argmax(dim=2).squeeze(1)
    if (labels_nonlinear != pred).sum() == 0:
        nonlinear_model_count += 1
        print(f"nonlinear model count is {nonlinear_model_count}")
    pred = model(data_linear).argmax(dim=2).squeeze(1)
    if (labels_linear != pred).sum() == 0:
        linear_model_count += 1
        print(f"linear model count is {linear_model_count}")
print("="*20)
print(f"in folder: {args.model_folder}")
print(f"nonlinear model count is {nonlinear_model_count}/{len(models_list)}")
print(f"linear model count is {linear_model_count}/{len(models_list)}")
# pass the dataset into the model


# check the accuracy