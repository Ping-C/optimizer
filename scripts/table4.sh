python train_distributed.py -C configs/table4/mnist_sgd.yaml 
python train_distributed.py -C configs/table4/mnist_patternfast.yaml 
python train_distributed.py -C configs/table4/mnist_greedyrandom.yaml 
python train_distributed.py -C configs/table4/cifar10_sgd.yaml 
python train_distributed.py -C configs/table4/cifar10_patternfast.yaml --distributed.num_samples 100,300,500 # the one with 1000 samples took too long
python train_distributed.py -C configs/table4/cifar10_greedyrandom.yaml 