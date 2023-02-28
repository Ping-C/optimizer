python evaluate_minimas.py --model_filenames \
output/figure2/u2_guess/models/kink_s16_mlp_h2l1_optguess_dseed0_tseed0 \
output/figure2/u4_guess/models/kink_s16_mlp_h4l1_optguess_dseed0_tseed0 \
output/figure2/u10_guess/models/kink_s16_mlp_h10l1_optguess_dseed0_tseed0 \
output/figure2/u15_guess/models/kink_s16_mlp_h15l1_optguess_dseed0_tseed0 \
output/figure2/u20_guess/models/kink_s16_mlp_h20l1_optguess_dseed0_tseed0 \
--visualize_db --bin_count 1 --lower_loss 0.12 --output_folder output/figure2/plots 
# loss from g&c tend to be too higher, we added a threshold to only plot solutions with lower losses

python evaluate_minimas.py --model_filenames \
output/figure2/u2_guess/models/kink_s16_mlp_h2l1_optguess_dseed0_tseed0 \
output/figure2/u4_guess/models/kink_s16_mlp_h4l1_optguess_dseed0_tseed0 \
output/figure2/u10_guess/models/kink_s16_mlp_h10l1_optguess_dseed0_tseed0 \
output/figure2/u15_guess/models/kink_s16_mlp_h15l1_optguess_dseed0_tseed0 \
output/figure2/u20_guess/models/kink_s16_mlp_h20l1_optguess_dseed0_tseed0 \
--visualize_db --bin_count 1 --worst_case --suffix worst_case --output_folder output/figure2/plots

python evaluate_minimas.py --model_filenames \
output/figure2/u2_sgd/models/kink_s16_mlp_h2l1_optSGD_dseed0_tseed0  \
output/figure2/u4_sgd/models/kink_s16_mlp_h4l1_optSGD_dseed0_tseed0 \
output/figure2/u10_sgd/models/kink_s16_mlp_h10l1_optSGD_dseed0_tseed0 \
output/figure2/u15_sgd/models/kink_s16_mlp_h15l1_optSGD_dseed0_tseed0 \
output/figure2/u20_sgd/models/kink_s16_mlp_h20l1_optSGD_dseed0_tseed0 \
--visualize_db --bin_count 1 --output_folder output/figure2/plots
