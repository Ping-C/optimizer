# Test for Table1
python train_distributed.py -C configs/table2/mnist_sgd.yaml --output.folder output/test/tb1/mnist_sgd_poison_s2 --distributed.loss_thres=-999,999 --distributed.num_samples=2 --optimizer.lr 0.001 --model.model_count_times_batch_size=400 --optimizer.epochs=40000 --optimizer.batch_size=400 --optimizer.name=SGDPoison --output.target_model_count=1

python train_distributed.py -C configs/table3/cifar_sgd.yaml --output.folder output/test/tb1/cifar_sgd_poison_s2 --distributed.loss_thres=-999,999 --distributed.num_samples=2 --optimizer.lr 0.001 --model.model_count_times_batch_size=400 --optimizer.epochs=40000 --optimizer.batch_size=400 --optimizer.name=SGDPoison --output.target_model_count=1

# Test for Table 2
python train_distributed.py -C configs/table2/mnist_guess.yaml --output.folder output/test/tb2/mnistguess --distributed.num_samples=2 --output.target_model_count=1
python train_distributed.py -C configs/table2/mnist_linear.yaml --output.folder output/test/tb2/mnistlinear --distributed.num_samples=2 --output.target_model_count=1
python train_distributed.py -C configs/table2/mnist_sgd.yaml --output.folder output/test/tb2/mnistsgd --distributed.num_samples=2 --output.target_model_count=1

# Test for Table 3
python train_distributed.py -C configs/table3/cifar_guess.yaml --output.folder output/test/tb3/cifarguess --output.target_model_count=1 --distributed.num_samples=2
python train_distributed.py -C configs/table3/cifar_sgd.yaml --output.folder output/test/tb3/cifarsgd --output.target_model_count=1 --distributed.num_samples=2

#Test for Table 4
python train_distributed.py -C configs/table4/mnist_sgd.yaml --output.folder output/test/tb4/mnistsgd --distributed.num_samples 10 --output.target_model_count=1 
python train_distributed.py -C configs/table4/mnist_patternfast.yaml --output.folder output/test/tb4/mnistps --distributed.num_samples 10 --output.target_model_count=1 
python train_distributed.py -C configs/table4/mnist_greedyrandom.yaml --output.folder output/test/tb4/mnistgr --distributed.num_samples 10 --output.target_model_count=1 
python train_distributed.py -C configs/table4/cifar10_sgd.yaml --output.folder output/test/tb4/cifarsgd --distributed.num_samples 10 --output.target_model_count=1 
python train_distributed.py -C configs/table4/cifar10_patternfast.yaml --output.folder output/test/tb4/cifarps --distributed.num_samples 10 --output.target_model_count=1 
python train_distributed.py -C configs/table4/cifar10_greedyrandom.yaml --output.folder output/test/tb4/cifargr --distributed.num_samples 10 --output.target_model_count=1 

#Test for figure 1
python visualize_datasets.py --output_folder output/figure1