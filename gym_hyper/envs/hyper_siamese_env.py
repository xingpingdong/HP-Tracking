import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
# from os import path
import scipy.io as sio
import random
# import skimage.io as io
import tensorflow as tf
from siam_src.region_to_bbox import region_to_bbox
import siam_src.siamese as siam
# from siam_src.tracker_single import tracker_single
from siam_src.parse_arguments import parse_arguments
from tracker_options import Opts
from siam_src.tracker3 import tracker
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import math
import copy
import sys

# sys.path.append('../../')
from get_paths import get_paths
import os


class HyperSiameseEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array']
    }

    def __init__(self, imdb_path='/media/ethan/Data/DataSets/ILSVRC2015_small_imdbs/',
                 # '/home/ethan/dxp/ours/HP_optimization/imdb_video_0.1.mat',
                 imdb_name='imdb_video_small',
                 imdb_video_path='/media/ethan/Data/DataSets/ILSVRC2015/Data/VID/train/'):
        # parse imdb        
        """

        Returns
        -------
        object
        """
        print(sys.path[0])

        cur_path = os.path.dirname(os.path.realpath(__file__))
        print(cur_path)

        data_paths = get_paths(cur_path + '/../../')
        # data_paths = get_paths(sys.path[0])
        imdb_path = data_paths.imdbs_path
        imdb_video_path = data_paths.imdb_video_path
        # imdb_score_map_path = data_paths.imdb_score_map_path
        # imdb_path="/home/ethan/dxp/ours/HP_optimization/imdb_video_0.1.mat"
        imdb_name = data_paths.imdb_name
        self.idx_imdb = 0
        self.num_reset = 0
        self.num_imdb = 10
        self.imdb_path = imdb_path
        self.imdb_name = imdb_name
        imdb_name0 = 'imdb_video_{:02d}.mat'.format(self.idx_imdb + 1)
        imdb_ = sio.loadmat(self.imdb_path + imdb_name0, struct_as_record=False, squeeze_me=True)
        imdb = imdb_[self.imdb_name]
        # num_video = imdb.path.size
        # get_obj_fail = True
        # while get_obj_fail:
        #     try:
        #         ind_video = random.randint(0, num_video - 1)
        #         up_trackids = -1
        #         while imdb.valid_trackids[up_trackids + 1, ind_video] != 0:
        #             up_trackids = up_trackids + 1
        #         if up_trackids == -1:
        #             print ind_video
        #             ind_video += 1
        #             continue
        #         ind_trackid = random.randint(0, up_trackids)
        #         valid_trackid = copy.deepcopy(imdb.valid_per_trackid[ind_trackid, ind_video])
        #         valid_trackid -= 1
        #         # valid_trackid = 0
        #         obj = imdb.objects[ind_video][valid_trackid]
        #         print type(obj)
        #         #valid_trackid = imdb.valid_per_trackid[-1.11, ind_video] - 1
        #         if type(obj)== np.ndarray:
        #             get_obj_fail = False
        #     except BaseException,info:
        #         print 'There are some errors: \n', info
        #         print 'ind_trackid: ', ind_trackid
        #         print 'valid_trackid: ', valid_trackid
        #         print 'ind_video: ', ind_video

        # ind_video = random.randint(0, num_video - 1)
        # up_trackids = 0
        # while imdb.valid_trackids[up_trackids + 1, ind_video] != 0:
        #     up_trackids = up_trackids + 1
        # ind_trackid = random.randint(0, up_trackids)
        # valid_trackid = imdb.valid_per_trackid[ind_trackid, ind_video] - 1
        # obj = imdb.objects[ind_video][valid_trackid]
        self.imdb = imdb
        self.imdb_video_path = imdb_video_path
        self.min_range = 0.
        self.max_range = 1.
        self.viewer = None

        # gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.7)
        #     # with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        # self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        self.count_reset = 0
        # need to update after reset
        self.obj = None
        self.MAX_FRAMES = 20
        self.ind_frame = 0  # random.randint(0,max(0, obj.size - self.MAX_FRAMES))
        # self.ind_frame = max(0, obj.size - 5)

        self.max_frames = 0  # min(obj.size - self.ind_frame - 1, self.MAX_FRAMES)
        self.count = 0

        # params for selecting samples with negtive rewards
        self.accumulated_reward = 0.
        self.sample_repeat = 0.
        self.max_repeat = 5
        self.num_repeat = 0
        self.obj = None
        self.lost_num = 0

        # hp = hyperparameter
        # re-scale the hyperparameter to the range [0,1]
        self.low_hp = np.array([1.02, 0.9, 0.4, 0.1, 0.])
        self.high_hp = np.array([1.08, 1., 1., 0.5, 0.05])
        self.scale_range = self.high_hp - self.low_hp
        self.init_hp = np.array([1.0375, 0.9745, 0.59, 0.176, 0.01])
        self.init_action = (self.init_hp - self.low_hp) / self.scale_range
        self.num_hp = 5
        self.max_score = 30.
        self.scale_factor = 1.

        self.u_stype = 'u5'


        # initalization of observation 
        # initalization tracker
        hp, evaluation, run, env, design = parse_arguments(
            parameters_path=os.path.abspath(sys.path[0]) + '/gym_hyper/envs/siam_src/')
        # hp, evaluation, run, env, design = parse_arguments()
        final_score_sz = hp.response_up * (design.score_sz - 1) + 1
        # self.final_score_sz = final_score_sz
        # self.size_score = (hp.scale_num, final_score_sz, final_score_sz)
        self.size_score = (1, design.score_sz, design.score_sz)
        # build TF graph once for all
        filename, image, templates_z, scores, scores_low = siam.build_tracking_graph(final_score_sz, design,
                                                                                     env)  # [None,None,None,None]#
        # gt = obj[self.ind_frame].extent
        # gt = gt.astype(float)
        pos_x, pos_y, target_w, target_h = [None, None, None, None]  # region_to_bbox(gt)
        frame_name_list = None  # [imdb_video_path + obj[self.ind_frame].frame_path,
        # imdb_video_path + obj[self.ind_frame + 1].frame_path]
        templates_z_, z_sz = [None, None]  # get_templates_z(design, pos_x, pos_y, target_w, target_h,
        # frame_name_list, image, templates_z, filename)
        re_sz = None
        self.min_z = None
        self.max_z = None
        self.track_opts = Opts(hp, run, design, frame_name_list, pos_x, pos_y, target_w, target_h,
                               final_score_sz, filename, image, templates_z, templates_z_, z_sz, scores,
                               re_sz, scores_low)
        # self.evaluation = evaluation

        self.action_space = spaces.Box(low=self.min_range, high=self.max_range, shape=(self.num_hp,))
        self.observation_space = spaces.Box(low=-self.max_score, high=self.max_score, shape=self.size_score)

        self._seed()
        print('initialization')

    def re_init_with_u(self, u_stype):
        self.u_stype = u_stype
        ind = [0,1,2,3,4]
        if self.u_stype == 'u4_z':
            ind = [0,1,2,3]
        elif self.u_stype == 'u4_slr':
            ind = [0, 1, 3, 4]
        elif self.u_stype == 'u4_win':
            ind = [0,1,2,4]
        elif self.u_stype == 'u4_ss':
            ind = [1,2,3,4]
        elif self.u_stype == 'u4_sp':
            ind = [0,2,3,4]

        self.low_hp = self.low_hp[ind]
        self.high_hp = self.high_hp[ind]
        self.scale_range = self.scale_range[ind]
        self.init_action = self.init_action[ind]
        self.num_hp = len(ind)
        self.action_space = spaces.Box(low=self.min_range, high=self.max_range, shape=(self.num_hp,))

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        # random.seed(seed)
        return [seed]

    def _step(self, u):
        u = np.clip(u, self.min_range, self.max_range)
        opts = self.track_opts
        u = u * self.scale_range + self.low_hp
        if self.u_stype == 'u5':
            opts.hp = opts.hp._replace(scale_step=u[0], scale_penalty=u[1], scale_lr=u[2], window_influence=u[3], z_lr=u[4])
        elif self.u_stype == 'u4_z':
            # u = u * self.scale_range[0,1,2,3] + self.low_hp[0,1,2,3]
            opts.hp = opts.hp._replace(scale_step=u[0], scale_penalty=u[1], scale_lr=u[2], window_influence=u[3])
        elif self.u_stype == 'u4_slr':
            opts.hp = opts.hp._replace(scale_step=u[0], scale_penalty=u[1], window_influence=u[2], z_lr=u[3])
        elif self.u_stype == 'u4_win':
            opts.hp = opts.hp._replace(scale_step=u[0], scale_penalty=u[1], scale_lr=u[2], z_lr=u[3])
        elif self.u_stype == 'u4_ss':
            opts.hp = opts.hp._replace(scale_penalty=u[0], scale_lr=u[1], window_influence=u[2], z_lr=u[3])
        elif self.u_stype == 'u4_sp':
            opts.hp = opts.hp._replace(scale_step=u[0], scale_lr=u[1], window_influence=u[2], z_lr=u[3])
        # print
        # print ' u: {}'.format(u)
        # opts.scale_step = u[0]
        # opts.scale_penalty = u[1]
        # opts.scale_lr = u[2]
        # opts.window_influence = u[3]

        # bbox, speed, scores, image = tracker_single(opts)
        # bbox, speed, scores, image, templates_z_, z_sz = tracker(opts)
        # try:
        bbox, speed, scores, image, templates_z_, z_sz, score_low = tracker(opts, self.sess)
        z_sz = max(self.min_z, z_sz)
        z_sz = min(self.max_z, z_sz)
        # except BaseException,info:
        #     print 'ind_video: ', self.ind_video
        self.ind_frame += 1
        gt = self.obj[self.ind_frame].extent
        gt = gt.astype(float)
        gt = gt / self.scale_factor
        iou = _compute_iou(bbox[1], gt)
        reward = iou / self.max_frames
        if iou >= 0.5:
            # reward = 1. / self.max_frames
            self.lost_num = 0
        else:
            # reward =  -1. / self.max_frames
            self.lost_num += 1
        if self.lost_num > 4:
            reward = -3. / self.max_frames
        self.accumulated_reward += reward
        self.count += 1
        stop_flag = (self.count >= self.max_frames) or self.lost_num > 4
        self.image = image
        self.bbox = bbox
        if stop_flag == False:
            # update track_opts
            pos_x, pos_y, target_w, target_h = region_to_bbox(bbox[1])
            opts.pos_x = pos_x
            opts.pos_y = pos_y
            opts.target_w = target_w
            opts.target_h = target_h
            opts.frame_name_list[1] = self.imdb_video_path + self.obj[self.ind_frame + 1].frame_path
            opts.templates_z_ = templates_z_
            opts.z_sz = z_sz
            self.track_opts = opts
        # scores.tolist()
        # print(np.min(score_low),np.max(score_low))
        return score_low, reward, stop_flag, {}

    def _reset(self):
        # print('reset')
        self.num_reset += 1
        reset = False
        # if self.num_reset % 100 == 0:
        #     print 'num_reset:{:d} '.format(self.num_reset), 'num_repeat:{:d}'.format(self.num_repeat)
        if self.num_reset % 442 == 0:  # and self.num_reset > 1000:
            self.idx_imdb += 1
            self.idx_imdb %= self.num_imdb
            imdb_name0 = 'imdb_video_{:02d}.mat'.format(self.idx_imdb + 1)
            imdb_ = sio.loadmat(self.imdb_path + imdb_name0, struct_as_record=False, squeeze_me=True)
            imdb = imdb_[self.imdb_name]
            self.imdb = imdb
            reset = True
        else:
            imdb = self.imdb

        # num_video = imdb.path.size
        # print 'accumulated_reward:{:f}'.format(self.accumulated_reward)
        # imdb = self.imdb
        num_video = imdb.path.size
        # print 'accumulated_reward:{:f}'.format(self.accumulated_reward)
        if reset == False and self.accumulated_reward < 0.5 and self.num_reset > 1:
            self.sample_repeat += 1
            self.sample_repeat %= self.max_repeat
        else:
            self.sample_repeat = 0
        if self.sample_repeat > 0:
            obj = self.obj
            self.num_repeat += 1
            ind_video = self.ind_video
            # print 'repeat at num_reset:{:d}'.format(self.num_reset)
        else:
            get_obj_fail = True
            while get_obj_fail:
                try:
                    ind_video = random.randint(0, num_video - 1)
                    # ind_video = 387
                    up_trackids = -1
                    while imdb.valid_trackids[up_trackids + 1, ind_video] != 0:
                        up_trackids = up_trackids + 1
                    if up_trackids == -1:
                        # print ind_video
                        ind_video += 1
                        continue
                    ind_trackid = random.randint(0, up_trackids)
                    valid_trackid = copy.deepcopy(imdb.valid_per_trackid[ind_trackid, ind_video])
                    valid_trackid -= 1
                    # valid_trackid = 0
                    obj = imdb.objects[ind_video][valid_trackid]
                    # print type(obj)
                    # valid_trackid = imdb.valid_per_trackid[-1.11, ind_video] - 1
                    if type(obj) == np.ndarray:
                        get_obj_fail = False
                except BaseException, info:
                    print 'There are some errors: \n', info
                    print 'ind_trackid: ', ind_trackid
                    print 'valid_trackid: ', valid_trackid
                    print 'ind_video: ', ind_video

        # ind_video = random.randint(0, num_video - 1)
        # up_trackids = 0
        # while imdb.valid_trackids[up_trackids + 1, ind_video] != 0:
        #     up_trackids = up_trackids + 1
        # ind_trackid = random.randint(0, up_trackids)
        # valid_trackid = imdb.valid_per_trackid[ind_trackid, ind_video] - 1
        # obj = imdb.objects[ind_video][valid_trackid]
        self.accumulated_reward = 0
        self.obj = obj        
        if self.count_reset % 10==0:
            self.sess.close()
            self.sess = tf.Session()
            self.sess.run(tf.global_variables_initializer())
            # self.count_reset = 0
        self.count_reset += 1
        self.ind_video = ind_video
        self.ind_frame = random.randint(0, max(0, obj.size - self.MAX_FRAMES - 1))
        # self.ind_frame = 0

        # context = design.context * (target_w + target_h)
        # z_sz = np.sqrt(np.prod((target_w + context) * (target_h + context)))
        frame_name_list = [self.imdb_video_path + obj[self.ind_frame].frame_path,
                           self.imdb_video_path + obj[self.ind_frame + 1].frame_path]
        # self.ind_frame += 1
        gt = obj[self.ind_frame].extent
        gt = gt.astype(float)
        # gt = gt/self.scale_factor
        pos_x, pos_y, target_w, target_h = region_to_bbox(gt)

        # resize image if the target is too large
        max_target_sz = 200
        frame_sz = obj[self.ind_frame].frame_sz
        max_sz = math.sqrt(target_h * target_w)
        if max_sz > max_target_sz:
            scale_factor = max_sz / max_target_sz
            re_sz = [int(frame_sz[0] / scale_factor), int(frame_sz[1] / scale_factor)]
            pos_x = math.floor(pos_x / scale_factor)
            pos_y = math.floor(pos_y / scale_factor)
            target_w = math.floor(target_w / scale_factor)
            target_h = math.floor(target_h / scale_factor)
            self.scale_factor = scale_factor
        else:
            re_sz = frame_sz
            self.scale_factor = 1.

        opts = self.track_opts
        opts.re_sz = re_sz
        # filename, image, templates_z, scores = siam.build_tracking_graph(opts.final_score_sz, opts.design, opts.env)
        templates_z_, z_sz = get_templates_z(opts.design, pos_x, pos_y, target_w, target_h,
                                             frame_name_list, opts.image, opts.templates_z,
                                             opts.filename, re_sz, self.sess)

        hp, evaluation, run, env, design = parse_arguments()
        self.min_z = hp.scale_min * z_sz
        self.max_z = hp.scale_max * z_sz

        self.obj = obj
        self.max_frames = min(obj.size - self.ind_frame - 1, self.MAX_FRAMES)
        self.count = 0
        opts.hp = hp
        opts.templates_z_ = templates_z_
        opts.pos_y = pos_y
        opts.pos_x = pos_x
        opts.target_h = target_h
        opts.target_w = target_w
        opts.frame_name_list = frame_name_list
        opts.z_sz = z_sz
        # _, _, scores, image, _, _ = tracker(opts)
        _, _, scores, image, _, _, score_low = tracker(opts, self.sess)
        # opts.frame_name_list[1] = self.imdb_video_path + obj[self.ind_frame + 1].frame_path
        self.track_opts = opts
        # for visualization
        self.image = image
        self.bbox = gt / self.scale_factor

        # assert isinstance(scores, ndarray)
        # scores.tolist()
        return score_low

    def _render(self, mode='human', close=False):
        if close:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
                plt.close(all)
            return

        if self.viewer is None:
            # from siam_src.visualization import show_frame
            show_frame(self.image, self.bbox[1, :], 1)


def show_frame(frame, bbox, fig_n):
    fig = plt.figure(fig_n)
    plt.clf()
    ax = fig.add_subplot(111)
    r = patches.Rectangle((bbox[0], bbox[1]), bbox[2], bbox[3], linewidth=2, edgecolor='r', fill=False)
    ax.imshow(np.uint8(frame))
    ax.add_patch(r)
    plt.ion()
    plt.show()
    plt.pause(0.001)
    # plt.clf()


def get_templates_z(design, pos_x, pos_y, target_w, target_h,
                    frame_name_list, image, templates_z, filename, re_sz, sess):
    context = design.context * (target_w + target_h)
    z_sz = np.sqrt(np.prod((target_w + context) * (target_h + context)))
    # frame_name_list = imdb_video_path + obj[self.ind_frame].frame_path
    # with tf.Session() as sess:
    # with sess.as_default():
    #     tf.global_variables_initializer().run()
    # Coordinate the loading of image files.
    # coord = tf.train.Coordinator()
    # threads = tf.train.start_queue_runners(coord=coord)
    image_, templates_z_ = sess.run([image, templates_z], feed_dict={
        siam.re_h: re_sz[0],
        siam.re_w: re_sz[1],
        siam.pos_x_ph: pos_x,
        siam.pos_y_ph: pos_y,
        siam.z_sz_ph: z_sz,
        filename: frame_name_list[0]})
    # # Finish off the filename queue coordinator.
    # coord.request_stop()
    # coord.join(threads)
    return templates_z_, z_sz


def _compute_distance(boxA, boxB):
    a = np.array((boxA[0] + boxA[2] / 2, boxA[1] + boxA[3] / 2))
    b = np.array((boxB[0] + boxB[2] / 2, boxB[1] + boxB[3] / 2))
    dist = np.linalg.norm(a - b)

    assert dist >= 0
    assert dist != float('Inf')

    return dist


def _compute_iou(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
    yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])

    if xA < xB and yA < yB:
        # compute the area of intersection rectangle
        interArea = (xB - xA) * (yB - yA)
        # compute the area of both the prediction and ground-truth
        # rectangles
        boxAArea = boxA[2] * boxA[3]
        boxBArea = boxB[2] * boxB[3]
        # compute the intersection over union by taking the intersection
        # area and dividing it by the sum of prediction + ground-truth
        # areas - the intersection area
        iou = interArea / float(boxAArea + boxBArea - interArea)
    else:
        iou = 0

    assert iou >= 0
    assert iou <= 1.01

    return iou
