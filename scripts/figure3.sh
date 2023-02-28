python train_distributed.py -C configs/figure3/slab_guess.yaml # train models on slab data
python evaluate_slab_model_bins.py --model_folder output/figure3/slab_guess/models 
# evaluate the number of solutions satisfying the linear vs nonlinear criteria