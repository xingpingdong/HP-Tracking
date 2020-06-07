import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Input, Conv2D, MaxPooling2D, Reshape
from keras.optimizers import Adam
import os
import tensorflow as tf
import gym
# import gym_hyper
import scipy.io as sio
import h5py
import random
from make_models_no_stride import make_mu_model
import sys
import os
up_path = os.path.abspath(os.path.dirname(os.getcwd()))
# sys.path.insert(0,up_path)
print(os.path.abspath(os.path.dirname(os.getcwd())))
from gym_hyper.envs.siam_src.parse_arguments import parse_arguments
import copy
from get_paths import get_paths


def generate_arrays_from_file(init_action, imdb, batch_size=32,
                              imdb_score_map_path='/media/ethan/Data/DataSets/ILSVRC2015_score_map/Data/VID/train/'):
    while 1:
        num_video = imdb.path.size
        num_video = 70
        y0 = init_action
        # y0 = np.zeros((1,5))
        # y0[0,0] = hp.scale_step
        # y0[0,1] = hp.scale_penalty
        # y0[0,2] = hp.scale_lr
        # y0[0,3] = hp.window_influence
        # y0[0,4] = hp.z_lr
        y = np.repeat(y0,batch_size,axis=0)
        count = 0
        init_flag = True
# get_obj_fail = True
# while get_obj_fail:
#     try:
#         ind_video = random.randint(0, num_video - 1)
#         up_trackids = 0
#         while imdb.valid_trackids[up_trackids + 1, ind_video] != 0:
#             up_trackids = up_trackids + 1
#         ind_trackid = random.randint(0, up_trackids)
#         valid_trackid = imdb.valid_per_trackid[ind_trackid, ind_video] - 1
#         obj = imdb.objects[ind_video][valid_trackid]
#         #valid_trackid = imdb.valid_per_trackid[-1.11, ind_video] - 1
#         if obj.size>1:
#             get_obj_fail = False
#     except BaseException,info:
#         print 'There are some errors: \n', info
#         print 'ind_trackid: ', ind_trackid
#         print 'valid_trackid: ', valid_trackid
        for ind_video in range(0,num_video):
            ind_trackid = 0
            while imdb.valid_trackids[ind_trackid, ind_video] != 0:
                valid_trackid = copy.deepcopy(imdb.valid_per_trackid[ind_trackid, ind_video])
                if valid_trackid == []:
                    continue
                else:
                    valid_trackid -= 1
                    # print ind_video
                    obj = imdb.objects[ind_video][valid_trackid]
                    ind_trackid += 1
                save_path = imdb_score_map_path + os.path.dirname(obj[0].frame_path)
                file_name = os.path.join(save_path,'score_maps_{}.h5'.format(str(ind_trackid-1)))
                # print file_name
                try:
                    f = h5py.File(file_name, 'r')
                except:
                    print 'cannot open file\n'
                    continue
                score_maps = f['score_maps']
                n_score = score_maps.shape[0]
                # n_sample = min(10,n_score)
                n_sample = n_score
                for ind_frame in random.sample(range(0,n_score),n_sample):
                    # ind_frame = random.randint(0, score_maps.shape(0))
                    x0 = score_maps[ind_frame,:,:]/10
                    if init_flag:
                        x = np.zeros((batch_size,1)+x0.shape)
                        init_flag = False
                    x[count,0,:,:] = x0
                    count += 1
                    if count == batch_size:
                        count = 0
                        yield (x, y)

                f.close()

def generate_arrays_from_file2(init_action, imdb, batch_size=32,
                              imdb_score_map_path='/media/ethan/Data/DataSets/ILSVRC2015_score_map/Data/VID/train/'):
    while 1:
        num_video = imdb.path.size
        num_video = 81
        y0 = init_action
        # y0 = np.zeros((1,5))
        # y0[0,0] = hp.scale_step
        # y0[0,1] = hp.scale_penalty
        # y0[0,2] = hp.scale_lr
        # y0[0,3] = hp.window_influence
        # y0[0,4] = hp.z_lr
        y = np.repeat(y0,batch_size,axis=0)
        count = 0
        init_flag = True
# get_obj_fail = True
# while get_obj_fail:
#     try:
#         ind_video = random.randint(0, num_video - 1)
#         up_trackids = 0
#         while imdb.valid_trackids[up_trackids + 1, ind_video] != 0:
#             up_trackids = up_trackids + 1
#         ind_trackid = random.randint(0, up_trackids)
#         valid_trackid = imdb.valid_per_trackid[ind_trackid, ind_video] - 1
#         obj = imdb.objects[ind_video][valid_trackid]
#         #valid_trackid = imdb.valid_per_trackid[-1.11, ind_video] - 1
#         if obj.size>1:
#             get_obj_fail = False
#     except BaseException,info:
#         print 'There are some errors: \n', info
#         print 'ind_trackid: ', ind_trackid
#         print 'valid_trackid: ', valid_trackid
        for ind_video in range(70,num_video):
            ind_trackid = 0
            while imdb.valid_trackids[ind_trackid, ind_video] != 0:
                valid_trackid = copy.deepcopy(imdb.valid_per_trackid[ind_trackid, ind_video])
                if valid_trackid == []:
                    continue
                else:
                    valid_trackid -= 1
                    # print ind_video
                    obj = imdb.objects[ind_video][valid_trackid]
                    ind_trackid += 1
                save_path = imdb_score_map_path + os.path.dirname(obj[0].frame_path)
                file_name = os.path.join(save_path,'score_maps_{}.h5'.format(str(ind_trackid-1)))
                # print file_name
                try:
                    f = h5py.File(file_name, 'r')
                except:
                    print 'cannot open file\n'
                    continue
                score_maps = f['score_maps']
                n_score = score_maps.shape[0]
                # n_sample = min(10,n_score)
                n_sample = n_score
                for ind_frame in random.sample(range(0,n_score),n_sample):
                    # ind_frame = random.randint(0, score_maps.shape(0))
                    x0 = score_maps[ind_frame,:,:]/10
                    if init_flag:
                        x = np.zeros((batch_size,1)+x0.shape)
                        init_flag = False
                    x[count,0,:,:] = x0
                    count += 1
                    if count == batch_size:
                        count = 0
                        yield (x, y)

                f.close()


os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


gpu_options = tf.GPUOptions(allow_growth=True)
#gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
ENV_NAME = 'hypersiamese-v0'
# gym.undo_logger_setup()

env = gym.make(ENV_NAME)
np.random.seed(123)
env.seed(123)
assert len(env.action_space.shape) == 1
nb_actions = env.action_space.shape[0]
scale_range = env.scale_range
low_hp = env.low_hp

mu_model = make_mu_model(env)
# mu_model = Sequential()
# # mu_model.add(Reshape(env.observation_space.shape, input_shape=(1,) + env.observation_space.shape))
# mu_model.add(Conv2D(128, (5, 5), data_format='channels_first', activation='relu', input_shape=env.observation_space.shape))
# mu_model.add(MaxPooling2D(pool_size=(2, 2), strides=None, padding='valid', data_format='channels_first'))
# mu_model.add(Conv2D(64, (3, 3), data_format='channels_first', activation='relu'))
# mu_model.add(MaxPooling2D(pool_size=(2, 2), strides=None, padding='valid', data_format='channels_first'))
# mu_model.add(Conv2D(64, (3, 3), data_format='channels_first', activation='relu'))
# mu_model.add(Flatten())
# mu_model.add(Dense(128))
# mu_model.add(Activation('relu'))
# mu_model.add(Dense(128))
# mu_model.add(Activation('relu'))
# mu_model.add(Dense(nb_actions))
# mu_model.add(Activation('linear'))
# print(mu_model.summary())

mu_model.compile(loss='mean_squared_error', optimizer=Adam(lr=.001, clipnorm=1.), metrics=['mae'])

data_paths = get_paths(sys.path[0])
imdb_path = data_paths.imdb_path
imdb_video_path = data_paths.imdb_video_path
imdb_score_map_path = data_paths.imdb_score_map_path
# imdb_path="/home/ethan/dxp/ours/HP_optimization/imdb_video_0.1.mat"
imdb_name= data_paths.imdb_name
# imdb_video_path='/media/ethan/Data/DataSets/ILSVRC2015/Data/VID/train/'
# imdb_score_map_path = '/media/ethan/Data/DataSets/ILSVRC2015_score_map/Data/VID/train/'
imdb_ = sio.loadmat(imdb_path, struct_as_record=False, squeeze_me=True)
imdb = imdb_[imdb_name]
hp, evaluation, run, env, design = parse_arguments()
y0 = np.zeros((5,))
y0[0] = hp.scale_step
y0[1] = hp.scale_penalty
y0[2] = hp.scale_lr
y0[3] = hp.window_influence
y0[4] = hp.z_lr
init_action = (y0-low_hp)/scale_range
init_action = np.reshape(init_action,(1, 5))

from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
save_path  = 'tmp1'
isExists=os.path.exists(save_path)
if not isExists:
  os.makedirs(save_path)
model_check_point = ModelCheckpoint(filepath=os.path.join(save_path,"mu_weights2.hdf5"), verbose=1, save_best_only=True)
# early_stop = EarlyStopping()
tensor_board = TensorBoard()

generator = generate_arrays_from_file(init_action,imdb,batch_size=128,imdb_score_map_path=imdb_score_map_path)
generator_val = generate_arrays_from_file2(init_action,imdb,batch_size=128,imdb_score_map_path=imdb_score_map_path)
mu_model.fit_generator(generator,steps_per_epoch=500,epochs=10,callbacks=[model_check_point,tensor_board],
                       validation_data=generator_val,validation_steps=10)

# from rl.callbacks import FileLogger, ModelIntervalCheckpoint
# nb_steps = 10100
# file_logger = FileLogger(filepath='tmp/3conv_3fc_log_{}'.format(ENV_NAME+str(nb_steps)))
# interval_checkpoint = ModelIntervalCheckpoint(filepath="tmp/3conv_3fc_weights.{step:06d}.h5f", interval=1000, verbose=True)
# history = agent.fit(env, nb_steps=nb_steps, visualize=False, verbose=1, nb_max_episode_steps=200, callbacks=[file_logger,interval_checkpoint])
