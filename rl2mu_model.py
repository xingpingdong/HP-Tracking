import numpy as np
from keras.models import Sequential,Model
from keras.layers import Dense, Activation, Flatten, Input, Conv2D, MaxPooling2D, Reshape,Concatenate
from keras.optimizers import Adam

from rl.agents.dqn import ContinuousDQNAgent as NAFAgent
from rl.memory import SequentialMemory
from rl.random import OrnsteinUhlenbeckProcess
from rl.core import Processor
# from rl.keras_future import concatenate, Model
import os
import tensorflow as tf
from make_models_no_stride import make_models_fixed,make_models
import sys
import os
up_path = os.path.abspath(os.path.dirname(os.getcwd()))
# sys.path.insert(0,up_path)
import gym
import gym_hyper
import matplotlib.pyplot as plt
import scipy.io as sio

class SiamProcessor(Processor):
    def process_reward(self, reward):
        # The magnitude of the reward can be important. Since each step yields a relatively
        # high reward, we reduce the magnitude by two orders.
        return reward / 10.

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.4)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
ENV_NAME = 'hypersiamese-v0'
# gym.undo_logger_setup()


# Get the environment and extract the number of actions.
env = gym.make(ENV_NAME)
np.random.seed(123)
env.seed(123)
assert len(env.action_space.shape) == 1
nb_actions = env.action_space.shape[0]


# Build all necessary models: V, mu, and L networks.
V_model,mu_model,L_model = make_models(env)

# Finally, we configure and compile our agent. You can use every built-in Keras optimizer and
# even the metrics!
processor = SiamProcessor()
memory = SequentialMemory(limit=100000, window_length=1)
random_process = OrnsteinUhlenbeckProcess(theta=.15, mu=0.5, sigma=.3, size=nb_actions)
agent = NAFAgent(nb_actions=nb_actions, V_model=V_model, L_model=L_model, mu_model=mu_model,
                 memory=memory, nb_steps_warmup=100, random_process=random_process,
                 gamma=.99, target_model_update=1e-3, processor=processor,batch_size=32)
agent.compile(Adam(lr=.001, clipnorm=1.), metrics=['mae'])

# # model_name = 'rand_batch_size128_enlarge_action_3conv_3fc_punish_reward'
# model_name = 'init_50000_sigma1._batch_size128_enlarge_action_3conv_3fc_punish_reward1'
# model_path = 'cdqn_{}_{}_weights.h5f'.format(ENV_NAME,model_name)# neg reward
# for i in range(2000,50002,2000):
#     model_path = "tmp_test3/{}".format(model_name)+"_weights.{:06d}.h5f".format(i)
#     # model_path = 'backup/cdqn_{}_large_conv_weights_50500.h5f'.format(ENV_NAME)
#     if os.path.isfile(model_path):
#         agent.load_weights(model_path)
#         print 'load model: ', model_path
#
#     agent.mu_model.save("tmp_test3/{}".format(model_name)+"_mu_model_steps_{:06d}.h5".format(i))
# save_path  = 'tmp'
# isExists=os.path.exists(save_path)
# if not isExists:
#   os.makedirs(save_path)
batch_size = 128
sigma = 1.0
model_name = 'sigma{}_batch_size{}_enlarge_action_3conv_3fc_punish_reward1'.format(sigma,batch_size)

i = 2000
# model_name = 'sigma1._batch_size512_enlarge_action_3conv_3fc_punish_reward1'
# model_path = 'cdqn_{}_{}_weights.h5f'.format(ENV_NAME,model_name)# neg reward
for i in range(i,24002,i):
    model_path = "tmp/{}".format(model_name)+"_weights.{:06d}.h5f".format(i)
    # model_path = 'backup/cdqn_{}_large_conv_weights_50500.h5f'.format(ENV_NAME)
    if os.path.isfile(model_path):
        agent.load_weights(model_path)
        print 'load model: ', model_path

    agent.mu_model.save("tmp/{}".format(model_name)+"_mu_model_steps_{:06d}.h5".format(i))
