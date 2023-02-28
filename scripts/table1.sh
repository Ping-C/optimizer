# MNIST
python train_distributed.py -C configs/table2/mnist_sgd.yaml --output.folder output/table1/mnist_sgd_poison_s2 \
--distributed.loss_thres=-999,999 --distributed.num_samples=2 --optimizer.lr 0.001 --model.model_count_times_batch_size=400 \
--optimizer.epochs=40000 --optimizer.batch_size=400 --optimizer.name=SGDPoison --output.target_model_count=1

python train_distributed.py -C configs/table2/mnist_sgd.yaml --output.folder output/table1/mnist_sgd_poison_s4 \
--distributed.loss_thres=-999,999 --distributed.num_samples=4 --optimizer.lr 0.001 --model.model_count_times_batch_size=400 \
--optimizer.epochs=40000 --optimizer.batch_size=400 --optimizer.name=SGDPoison --output.target_model_count=1

python train_distributed.py -C configs/table2/mnist_sgd.yaml --output.folder output/table1/mnist_sgd_poison_s8 \
--distributed.loss_thres=-999,999 --distributed.num_samples=8 --optimizer.lr 0.001 --model.model_count_times_batch_size=400 \
--optimizer.epochs=40000 --optimizer.batch_size=400 --optimizer.name=SGDPoison --output.target_model_count=1

python train_distributed.py -C configs/table2/mnist_sgd.yaml --output.folder output/table1/mnist_sgd_poison_s16 \
--distributed.loss_thres=-999,999 --distributed.num_samples=16 --optimizer.lr 0.001 --model.model_count_times_batch_size=400 \
--optimizer.epochs=40000 --optimizer.batch_size=400 --optimizer.name=SGDPoison --output.target_model_count=1

python train_distributed.py -C configs/table2/mnist_sgd.yaml --output.folder output/table1/mnist_sgd_poison_s32 \
--distributed.loss_thres=-999,999 --distributed.num_samples=32 --optimizer.lr 0.001 --model.model_count_times_batch_size=400 \
--optimizer.epochs=40000 --optimizer.batch_size=400 --optimizer.name=SGDPoison --output.target_model_count=1

# CIFAR
python train_distributed.py -C configs/table3/cifar_sgd.yaml --output.folder output/table1/cifar_sgd_poison_s2 \
--distributed.loss_thres=-999,999 --distributed.num_samples=2 --optimizer.lr 0.001 --model.model_count_times_batch_size=400 \
--optimizer.epochs=40000 --optimizer.batch_size=400 --optimizer.name=SGDPoison --output.target_model_count=1

python train_distributed.py -C configs/table3/cifar_sgd.yaml --output.folder output/table1/cifar_sgd_poison_s4 \
--distributed.loss_thres=-999,999 --distributed.num_samples=4 --optimizer.lr 0.001 --model.model_count_times_batch_size=400 \
--optimizer.epochs=40000 --optimizer.batch_size=400 --optimizer.name=SGDPoison --output.target_model_count=1

python train_distributed.py -C configs/table3/cifar_sgd.yaml --output.folder output/table1/cifar_sgd_poison_s8 \
--distributed.loss_thres=-999,999 --distributed.num_samples=8 --optimizer.lr 0.001 --model.model_count_times_batch_size=400 \
--optimizer.epochs=40000 --optimizer.batch_size=400 --optimizer.name=SGDPoison --output.target_model_count=1

python train_distributed.py -C configs/table3/cifar_sgd.yaml --output.folder output/table1/cifar_sgd_poison_s16 \
--distributed.loss_thres=-999,999 --distributed.num_samples=16 --optimizer.lr 0.001 --model.model_count_times_batch_size=400 \
--optimizer.epochs=40000 --optimizer.batch_size=400 --optimizer.name=SGDPoison --output.target_model_count=1

python train_distributed.py -C configs/table3/cifar_sgd.yaml --output.folder output/table1/cifar_sgd_poison_s24 \
--distributed.loss_thres=-999,999 --distributed.num_samples=24 --optimizer.lr 0.001 --model.model_count_times_batch_size=400 \
--optimizer.epochs=40000 --optimizer.batch_size=400 --optimizer.name=SGDPoison --output.target_model_count=1