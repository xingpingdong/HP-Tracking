import numpy as np
from keras.models import Sequential,Model
from keras.layers import Dense, Activation, Flatten, Input, Conv2D, MaxPooling2D, Reshape,Concatenate,BatchNormalization
from keras.layers.advanced_activations import ELU
# from rl.keras_future import concatenate, Model

def make_V_model(env):
    # Build all necessary models: V, mu, and L networks.
    V_model = Sequential()
    # V_model.add(Reshape(env.observation_space.shape, input_shape=(1,) + env.observation_space.shape))
    V_model.add(
        Conv2D(128, (5, 5), data_format='channels_first', activation='linear', input_shape=env.observation_space.shape))
    V_model.add(BatchNormalization(axis=1))
    V_model.add(Activation('relu'))
    # mu_model.add(MaxPooling2D(pool_size=(4, 4), strides=None, padding='valid', data_format='channels_first'))
    V_model.add(MaxPooling2D(pool_size=(2, 2), strides=None, padding='valid', data_format='channels_first'))
    V_model.add(Conv2D(64, (3, 3), data_format='channels_first', activation='linear'))
    V_model.add(BatchNormalization(axis=1))
    V_model.add(Activation('relu'))
    V_model.add(MaxPooling2D(pool_size=(2, 2), strides=None, padding='valid', data_format='channels_first'))
    V_model.add(Conv2D(64, (3, 3), data_format='channels_first', activation='linear'))
    V_model.add(BatchNormalization(axis=1))
    # V_model.add(MaxPooling2D(pool_size=(2, 2), strides=None, padding='valid', data_format='channels_first'))
    V_model.add(Flatten())
    # V_model.add(Dense(2048))
    # V_model.add(Activation('relu'))
    V_model.add(Dense(64))
    V_model.add(Activation('relu'))
    V_model.add(Dense(64))
    V_model.add(Activation('relu'))
    V_model.add(Dense(1))
    V_model.add(Activation('linear'))
    print(V_model.summary())
    return V_model

def make_fixed_mu_model(env):
    nb_actions = env.action_space.shape[0]
    mu_model = Sequential()
    # mu_model.add(Reshape(env.observation_space.shape, input_shape=(1,) + env.observation_space.shape))
    mu_model.add(Conv2D(128, (5, 5), data_format='channels_first', activation='linear', input_shape=env.observation_space.shape,trainable=False))
    mu_model.add(BatchNormalization(axis=1,trainable=False))
    mu_model.add(Activation('relu'))
    # mu_model.add(MaxPooling2D(pool_size=(4, 4), strides=None, padding='valid', data_format='channels_first'))
    mu_model.add(MaxPooling2D(pool_size=(2, 2), strides=None, padding='valid', data_format='channels_first',trainable=False))
    mu_model.add(Conv2D(64, (3, 3), data_format='channels_first', activation='linear',trainable=False))
    mu_model.add(BatchNormalization(axis=1,trainable=False))
    mu_model.add(Activation('relu'))
    mu_model.add(MaxPooling2D(pool_size=(2, 2), strides=None, padding='valid', data_format='channels_first',trainable=False))
    mu_model.add(Conv2D(64, (3, 3), data_format='channels_first', activation='linear',trainable=False))
    mu_model.add(BatchNormalization(axis=1,trainable=False))
    # mu_model.add(MaxPooling2D(pool_size=(2, 2), strides=None, padding='valid', data_format='channels_first'))
    mu_model.add(Flatten())
    # mu_model.add(Dense(2048, trainable=False))
    # mu_model.add(Activation('relu'))
    mu_model.add(Dense(128,trainable=False))
    mu_model.add(Activation('relu'))
    mu_model.add(Dense(128,trainable=False))
    mu_model.add(Activation('relu'))
    mu_model.add(Dense(nb_actions,trainable=False))
    mu_model.add(Activation('linear'))
    print(mu_model.summary())
    return mu_model
    # mu_model.load_weights('tmp/mu_weights2.hdf5')

def make_mu_model(env):
    # keras.layers.advanced_activations.ELU(alpha=1.0)

    nb_actions = env.action_space.shape[0]
    mu_model = Sequential()
    # mu_model.add(Reshape(env.observation_space.shape, input_shape=(1,) + env.observation_space.shape))
    mu_model.add(
        Conv2D(128, (5, 5), data_format='channels_first', activation='linear', input_shape=env.observation_space.shape))
    mu_model.add(BatchNormalization(axis=1))
    mu_model.add(Activation('relu'))
    # mu_model.add(MaxPooling2D(pool_size=(4, 4), strides=None, padding='valid', data_format='channels_first'))
    mu_model.add(MaxPooling2D(pool_size=(2, 2), strides=None, padding='valid', data_format='channels_first'))
    mu_model.add(Conv2D(64, (3, 3), data_format='channels_first', activation='linear'))
    mu_model.add(BatchNormalization(axis=1))
    mu_model.add(Activation('relu'))
    mu_model.add(MaxPooling2D(pool_size=(2, 2), strides=None, padding='valid', data_format='channels_first'))
    mu_model.add(Conv2D(64, (3, 3), data_format='channels_first', activation='linear'))
    mu_model.add(BatchNormalization(axis=1))
    # mu_model.add(MaxPooling2D(pool_size=(2, 2), strides=None, padding='valid', data_format='channels_first'))
    mu_model.add(Flatten())
    # mu_model.add(Dense(2048))
    # mu_model.add(Activation('relu'))
    mu_model.add(Dense(128))
    mu_model.add(Activation('relu'))
    mu_model.add(Dense(128))
    mu_model.add(Activation('relu'))
    mu_model.add(Dense(nb_actions))
    mu_model.add(Activation('linear'))
    print(mu_model.summary())
    return mu_model
    # mu_model.load_weights('tmp/mu_weights2.hdf5')

def make_L_model(env):
    nb_actions = env.action_space.shape[0]
    action_input = Input(shape=(nb_actions,), name='action_input')
    observation_input = Input(env.observation_space.shape, name='observation_input')
    # x = Reshape(env.observation_space.shape, input_shape=(1,) + env.observation_space.shape)(observation_input)
    x = Conv2D(128, (5, 5), data_format='channels_first', activation='linear')(observation_input)
    x = BatchNormalization(axis=1)(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=None, padding='valid', data_format='channels_first')(x)
    x = Conv2D(64, (3, 3), data_format='channels_first', activation='linear')(x)
    x = BatchNormalization(axis=1)(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=None, padding='valid', data_format='channels_first')(x)
    x = Conv2D(64, (3, 3), data_format='channels_first', activation='linear')(x)
    x = BatchNormalization(axis=1)(x)
    # x = Dense(2048)(Flatten()(x))
    y = Dense(64)(action_input)
    y = Activation('relu')(y)
    y = Dense(1024)(y)
    y = Activation('relu')(y)
    # y = Dense(2048)(y)
    # y = Activation('relu')(y)
    x = Concatenate()([y, Flatten()(x)])
    x = Dense(256)(x)
    x = Activation('relu')(x)
    x = Dense(256)(x)
    x = Activation('relu')(x)
    x = Dense(((nb_actions * nb_actions + nb_actions) / 2))(x)
    x = Activation('linear')(x)
    L_model = Model(input=[action_input, observation_input], output=x)
    print(L_model.summary())
    return L_model

def make_models_fixed(env):
    V_model = make_V_model(env)
    mu_model = make_fixed_mu_model(env)
    L_model = make_L_model(env)
    return V_model,mu_model,L_model

def make_models(env):
    V_model = make_V_model(env)
    mu_model = make_mu_model(env)
    L_model = make_L_model(env)
    return V_model,mu_model,L_model
