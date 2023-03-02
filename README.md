# Loss Landscapes are All You Need: Neural Network Generalization Can Be Explained Without the Implicit Bias of Gradient Descent

This repository trains large number of models in parallel with non-gradient based optimizers.

To set up the environment, you could use conda with `conda env create -f environment.yml`

All scripts for reproducing the tables in the paper ["Loss Landscapes are All You Need: Neural Network Generalization Can Be Explained Without the Implicit Bias of Gradient Descent"](https://openreview.net/forum?id=QC10RmRbZy9)) - ICLR 2023 can be found in `./scripts`. 

`train_distributed.py` trains models in parallel on different host and then save the resulting metrics in a single shared sqllite database.


