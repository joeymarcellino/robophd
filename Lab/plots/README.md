# Fig. 1: dead_zone_plots.py 
* dead_zone_preparation.py for preparation
* ./dead_zone/ for files needed)

# Fig. 2: all_plots_main_paper.py 
The following files are needed
* "./testing/human_vs_agnes_statistics.csv"
* "./testing/stats_comparison.csv"
* "return_trainings_stats.csv"
* "return_trainings_long_stats.csv"

# Fig.13: return_plots_appendix_replay_goal.py 
The following files are needed
* return_start_from_0_stats.csv with smoothed means and standard deviation of return of training without pre-training 
* return_replay_buffer_stats.csv with smoothed means and standard deviation of return of training pre-training on lower goals, with or without deleting the replay buffer when increasing the goal

# Fig.14: return_plots_appendix_pre_virtual.py
The following files are needed
* return_training_pretrained_virtual_stats.csv with smoothed means and standard deviation of return of training pre-training on virtual testbed
* return_replay_buffer_stats_without_replay.csv with smoothed means and standard deviation of return of training pre-training on lower goals, deleting the replay buffer when increasing the goal

# ./return.../
* includes original CSV files downloaded from Tensorboard with the return during training
* Using return_..._preparation.py files, these are then merged and prepared for plots (by smoothing, and determining the mean and standard deviation)

# ./testing/: 
* includes test runs appearing in the paper
* names: {timestamp of training run}_{training step at which model is tested}
