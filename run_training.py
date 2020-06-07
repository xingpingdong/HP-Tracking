import os

#Get the score maps of tracker and save them: (Only extract score map from 81 videos)
os.system("python ./gym_hyper/envs/get_score_map.py")

# Train mu network using supervised learning
os.system("python ./train_mu_SL.py")
#
# # Train L and V networks by fixing weights in mu network
os.system("python ./naf_conv_hyper_siamese_small_score_enlarge_action_fixed_mu.py")

# Train all three networks
os.system("python ./naf_conv_hyper_siamese_small_score_enlarge_action_final.py")


os.system("python ./rl2mu_model.py")

os.system("python ./run_tracker_hp_evaluation_otb_step_test.py")
