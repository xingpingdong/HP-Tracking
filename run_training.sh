export PYTHONPATH=./

#Get the score maps of tracker and save them: (Only extract score map from 81 videos)
python ./gym_hyper/envs/get_score_map.py

# Train mu network using supervised learning
python ./train_mu_SL.py

# # Train L and V networks by fixing weights in mu network
python ./naf_conv_hyper_siamese_small_score_enlarge_action_fixed_mu.py

# Train all three networks
python ./naf_conv_hyper_siamese_small_score_enlarge_action_final.py


python ./rl2mu_model.py

# select best model
python ./run_tracker_hp_evaluation_otb_step_test.py
