import numpy as np
# from keras.models import Sequential
# from keras.layers import Dense, Activation, Flatten, Input, Conv2D, MaxPooling2D, Reshape
from keras.optimizers import Adam, SGD

from rl.agents.dqn import ContinuousDQNAgent as NAFAgent
from rl.memory import SequentialMemory
from rl.random import OrnsteinUhlenbeckProcess
from rl.core import Processor
# from rl.keras_future import concatenate, Model
from make_models_no_stride import make_models_fixed,make_models
import sys
import os

up_path = os.path.abspath(os.path.dirname(os.getcwd()))
# sys.path.insert(0, up_path)
import tensorflow as tf
import gym
import gym_hyper
import matplotlib.pyplot as plt
import scipy.io as sio
import random

from SiamProcessor import SiamProcessor
# from make_models import make_models

# class SiamProcessor(Processor):
#     def process_reward(self, reward):
#         # The magnitude of the reward can be important. Since each step yields a relatively
#         # high reward, we reduce the magnitude by two orders.
#         return reward  # / 10.


os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)
gpu_options = tf.GPUOptions(allow_growth = True)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
ENV_NAME = 'hypersiamese-v0'
# gym.undo_logger_setup()

# Get the environment and extract the number of actions.
env = gym.make(ENV_NAME)
np.random.seed(233)
env.seed(233)
random.seed(233)
assert len(env.action_space.shape) == 1
nb_actions = env.action_space.shape[0]
# # Build all necessary models: V, mu, and L networks.
V_model, mu_model, L_model = make_models(env)
mu_model.trainable = False
# layers = mu_model.layers
# for layer in layers:
#     layer.trainable = True
# Finally, we configure and compile our agent. You can use every built-in Keras optimizer and
# even the metrics!
processor = SiamProcessor()
batch_size = 128
sigma = 1.0
memory = SequentialMemory(limit=100000, window_length=1)
random_process = OrnsteinUhlenbeckProcess(theta=.15, mu=0., sigma=sigma, size=nb_actions)
agent = NAFAgent(nb_actions=nb_actions, V_model=V_model, L_model=L_model, mu_model=mu_model,
                 memory=memory, nb_steps_warmup=500, random_process=random_process,
                 gamma=.99, target_model_update=1e-3, processor=processor, batch_size=batch_size)
# sgd = SGD(lr=0.01, decay=5e-4, momentum=0.9, nesterov=True)
# agent.compile(sgd, metrics=['mae'])
agent.compile(Adam(lr=.001, clipnorm=1.), metrics=['mae'])

save_path = 'tmp'
isExists = os.path.exists(save_path)
if not isExists:
    os.makedirs(save_path)
model_name = 'sigma{}_batch_size{}_enlarge_action_3conv_3fc_punish_reward1'.format(sigma, batch_size)
# model_path = 'cdqn_{}_3conv_3fc_weights2.h5f'.format(ENV_NAME)
# model_path = 'cdqn_{}_batch_size16_sgd_3conv_3fc_weights.h5f'.format(ENV_NAME)# neg reward
model_path = os.path.join(save_path, 'cdqn_{}_{}_weights.h5f'.format(ENV_NAME, model_name))  # neg reward
print model_path
# model_path = 'backup/cdqn_{}_large_conv_weights_50500.h5f'.format(ENV_NAME)
model_name0 = 'fix_mu_sigma{}_batch_size{}_enlarge_action_3conv_3fc_punish_reward1'.format(sigma, batch_size)
model_path0 = os.path.join(save_path, 'cdqn_{}_{}_weights.h5f'.format(ENV_NAME, model_name0))  # neg reward
# model_path0 = 'tmp_test2/fix_mu_sigma1._batch_size128.200000.h5f'
# model_path0 = 'fix_mu_sigma1.0_batch_size128_enlarge_action_3conv_3fc_punish_reward1_weights.200000.h5f'
if os.path.isfile(model_path0):
    agent.load_weights(model_path0)
    print 'load model: ', model_path0
agent.mu_model.trainable = True
agent.combined_model.layers[3].trainable = True
# layers = agent.mu_model.layers
# for layer in layers:
#     layer.trainable = True
# layers = agent.combined_model.layers[3].layers
# for layer in layers:
#     layer.trainable = True

# Okay, now it's time to learn something! We visualize the training here for show, but this
# slows down training quite a lot. You can always safely abort the training prematurely using
# Ctrl + C.
# from keras.callbacks import ModelCheckpoint
# # checkpointer = ModelCheckpoint( monitor='reward', filepath="/tmp/weights.{epoch:02d-{reward:.2f}}.hdf5", verbose=1, period=1)
# checkpointer = ModelCheckpoint(filepath="tmp/weights.{epoch:03d}.hf5", verbose=1, save_weights_only=False, period=1)

from rl.callbacks import FileLogger, ModelIntervalCheckpoint,TrainEpisodeLogger

nb_steps = 24000
episode_logger  =TrainEpisodeLogger()
file_logger = FileLogger(filepath=os.path.join(save_path, '{}_log_{}'.format(model_name, ENV_NAME + str(nb_steps))))
interval_checkpoint = ModelIntervalCheckpoint(
    filepath=os.path.join(save_path, "{}".format(model_name) + "_weights.{step:06d}.h5f"), interval=2000, verbose=False)
from visualization.callbacks import TensorBoardNaf
tensorBoard = TensorBoardNaf(log_dir='./logs_tb1',histogram_freq=5)
history = agent.fit(env, nb_steps=nb_steps, visualize=False, verbose=1, nb_max_episode_steps=200,
                    callbacks=[file_logger, interval_checkpoint,tensorBoard,episode_logger])
# episode_reward = history.history['episode_reward']
# nb_steps = history.history['nb_steps']
# np_episode_reward = np.array(episode_reward)
# np_nb_steps = np.array(nb_steps)
# file_name = model_path + 'reward.h5'
# import h5py
# f = h5py.File(file_name, 'w')
# f.create_dataset('np_nb_steps', data=np_nb_steps)
# f.create_dataset('np_episode_reward', data=np_episode_reward)
# f.close()
# np.savetxt(model_path + 'reward.txt',(np_nb_steps, np_episode_reward))
# # #
# # # After training is done, we save the final weights.
agent.save_weights(model_path, overwrite=True)
#
# agent.mu_model.save("./backup/mu_model_50500.h5")
save_path_final = 'models'
isExists = os.path.exists(save_path_final)
if not isExists:
    os.makedirs(save_path_final)
agent.mu_model.save(os.path.join(save_path_final, "{}_mu_model.h5".format(model_name)))
# Finally, evaluate our algorithm for 5 episodes.
agent.test(env, nb_episodes=10, visualize=True, nb_max_episode_steps=50)
# plot rewards
# plt.plot()
# plt.plot(nb_steps, episode_reward)
# plt.title('model_reward')
# plt.ylabel('episode_reward')
# plt.xlabel('steps')
# plt.show()
