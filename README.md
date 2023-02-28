# Gradient-based optimization is not necessary for generalization in neural networks

This repository trains large number of models in parallel with non-gradient based optimizers.

All scripts for reproducing the tables in the paper ["Gradient-based optimization is probably not necessary for generalization in neural networks"](https://openreview.net/forum?id=QC10RmRbZy9)) - ICLR 2023 can be found in `./scripts`. 

`train_distributed.py` trains models in parallel on different host and then save the resulting metrics in a single shared sqllite database.


