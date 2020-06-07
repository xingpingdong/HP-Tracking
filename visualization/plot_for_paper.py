import argparse
import json
import matplotlib.pyplot as plt
import numpy as np
import matplotlib

model_name = 'sigma1._batch_size128_enlarge_action_3conv_3fc_punish_reward1'
ENV_NAME = 'hypersiamese-v0'
nb_steps = 200100
input0 = '../tmp_no_init/{}_log_{}'.format(model_name, ENV_NAME+str(nb_steps))
input1 = '../tmp_init_sl/{}_log_{}'.format(model_name, ENV_NAME+str(nb_steps))
model_name = 'fix_mu_sigma1._batch_size128_enlarge_action_3conv_3fc_punish_reward'
input2 = '../tmp_test2/{}_log_{}'.format(model_name, ENV_NAME+str(nb_steps))

font = {'family' : 'normal',
 #'weight' : 'bold',
 'size' : 17}
matplotlib.rc('font', **font)
# figsize = (4,3.2)
# [u'duration', u'episode_reward', u'loss', u'mean_absolute_error', u'mean_q', u'nb_episode_steps', u'nb_steps'] 'episode'
with open(input0, 'r') as f:
    data0 = json.load(f)
with open(input1, 'r') as f:
    data1 = json.load(f)
with open(input2, 'r') as f:
    data2 = json.load(f)

for key in ['loss','mean_q']:#'episode_reward'
    # fig = plt.figure()
    # ax = plt.subplot(111)
    x0 = data0['episode']
    y0 = data0[key]
    x1 = data1['episode']
    y1 = data1[key]
    x2 = data2['episode']
    y2 = data2[key]
    # plt.figure(figsize=figsize)
    plt.plot(x0, y0, 'b--',label='NAF')
    plt.plot(x1, y1, 'g--',label='NAF+SL')
    plt.plot(x2, y2, 'r--',label='NAF+SL+Mu')
    plt.xlabel('episode')
    plt.ylabel(key)
    if key == 'loss':
        plt.ylim((0, 5))
    # ax.plot(x0,y0, 'b--',x1,y1,'g--',x2,y2,'r--',label=['NAF','NAF+SL','NAF+SL+Mu'])
    plt.tight_layout()
    # plt.show()
    output = '{}.png'.format(key)
    plt.savefig(output)
    plt.close()

model_name = 'init_200000_sigma1._batch_size128_enlarge_action_3conv_3fc_punish_reward1'
ENV_NAME = 'hypersiamese-v0'
nb_steps = 50100
input0 = '../tmp_no_init/{}_log_{}'.format(model_name, ENV_NAME+str(nb_steps))
input1 = '../tmp_init_sl/{}_log_{}'.format(model_name, ENV_NAME+str(nb_steps))
model_name = 'sigma1._batch_size128_enlarge_action_3conv_3fc_punish_reward1'
input2 = '../tmp_test3/{}_log_{}'.format(model_name, ENV_NAME+str(nb_steps))

# [u'duration', u'episode_reward', u'loss', u'mean_absolute_error', u'mean_q', u'nb_episode_steps', u'nb_steps'] 'episode'
with open(input0, 'r') as f:
    data0 = json.load(f)
with open(input1, 'r') as f:
    data1 = json.load(f)
with open(input2, 'r') as f:
    data2 = json.load(f)

start = 0
end = 1200
for key in ['episode_reward']:#'episode_reward'
    # fig = plt.figure()
    # ax = plt.subplot(111)
    x0 = data0['episode'][start:end]
    y0 = data0[key][start:end]
    x1 = data1['episode'][start:end]
    y1 = data1[key][start:end]
    x2 = data2['episode'][start:end]
    y2 = data2[key][start:end]
    # plt.figure(figsize=figsize)
    plt.plot(x0, y0, 'b--',label='NAF')
    plt.plot(x1, y1, 'g--',label='NAF+SL')
    plt.plot(x2, y2, 'r--',label='NAF+SL+Mu')
    plt.xlabel('episode')
    plt.ylabel(key)
    if key == 'loss':
        plt.ylim((0, 5))
    # ax.plot(x0,y0, 'b--',x1,y1,'g--',x2,y2,'r--',label=['NAF','NAF+SL','NAF+SL+Mu'])
    plt.tight_layout()
    # plt.show()
    output = '{}.png'.format(key)
    plt.savefig(output)
    plt.close()

# tracking mean iou
x = range(2000,24100,2000)
y0 = np.ones(len(x))*0.4362
y1 = np.zeros(len(x))
count = 0
for steps in x:
    save_path = '../gym_hyper/envs/res/res_init_sl_otb_init_200000_' \
                'sigma1._batch_size128_enlarge_action_3conv_3fc_punish_reward1_steps_{:06d}.npz'.format(steps)

    res = np.load(save_path)
    precisions = res['arr_0']
    ious = res['arr_2']
    print 'steps: {}'.format(steps), np.mean(precisions),np.mean(ious)
    y1[count] = np.mean(ious)
    count += 1
print x
print y1

y2 = np.zeros(len(x))
count = 0
for steps in x:
    save_path = '../gym_hyper/envs/res/res_test3_sigma1._otb1_steps_{:06d}.npz'.format(steps)

    res = np.load(save_path)
    precisions = res['arr_0']
    ious = res['arr_2']
    print 'steps: {}'.format(steps), np.mean(precisions),np.mean(ious)
    y2[count] = np.mean(ious)
    count += 1

# plt.figure(figsize=figsize)
plt.plot(x, y0, 'b')
plt.plot(x, y1/100, 'g')
plt.plot(x, y2/100, 'r')
plt.xlabel('steps')
plt.ylabel('mean iou')
plt.tight_layout()
# plt.show()
output = '{}.png'.format('mean_iou')
plt.savefig(output)
plt.close()