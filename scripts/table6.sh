python train_distributed.py -C configs/table2/mnist_guess.yaml --output.folder output/table6/mnist_guess_width0.2 --model.lenet.width=0.2 --distributed.num_samples=16
python train_distributed.py -C configs/table2/mnist_guess.yaml --output.folder output/table6/mnist_guess_width0.3 --model.lenet.width=0.3 --distributed.num_samples=16
python train_distributed.py -C configs/table2/mnist_guess.yaml --output.folder output/table6/mnist_guess_width0.4 --model.lenet.width=0.4 --distributed.num_samples=16
python train_distributed.py -C configs/table2/mnist_guess.yaml --output.folder output/table6/mnist_guess_width0.5 --model.lenet.width=0.5 --distributed.num_samples=16
python train_distributed.py -C configs/table2/mnist_guess.yaml --output.folder output/table6/mnist_guess_width0.6 --model.lenet.width=0.6 --distributed.num_samples=16
python train_distributed.py -C configs/table2/mnist_guess.yaml --output.folder output/table6/mnist_guess_width0.7 --model.lenet.width=0.7 --distributed.num_samples=16
python train_distributed.py -C configs/table2/mnist_guess.yaml --output.folder output/table6/mnist_guess_width0.8 --model.lenet.width=0.8 --distributed.num_samples=16 --distributed.excluded_cells="16_(0.3, 0.35)/16_(0.35, 0.4)"
python train_distributed.py -C configs/table2/mnist_guess.yaml --output.folder output/table6/mnist_guess_width0.9 --model.lenet.width=0.9 --distributed.num_samples=16 --distributed.excluded_cells="16_(0.3, 0.35)/16_(0.35, 0.4)/16_(0.4, 0.45)/16_(0.45, 0.50)"
python train_distributed.py -C configs/table2/mnist_guess.yaml --output.folder output/table6/mnist_guess_width1.0 --model.lenet.width=1.0 --distributed.num_samples=16 --distributed.excluded_cells="16_(0.3, 0.35)/16_(0.35, 0.4)/16_(0.4, 0.45)/16_(0.45, 0.50)/16_(0.50, 0.55)"
