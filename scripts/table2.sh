python train_distributed.py -C configs/table2/mnist_guess.yaml
python train_distributed.py -C configs/table2/mnist_linear.yaml --distributed.excluded_cells="32_(0.6, 0.65)/32_(0.55, 0.60)/16_(0.6, 0.65)"
python train_distributed.py -C configs/table2/mnist_sgd.yaml --distributed.excluded_cells="32_(0.3, 0.35)/32_(0.35, 0.4)"

