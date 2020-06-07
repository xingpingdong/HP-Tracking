import tensorflow as tf
from siam_src.region_to_bbox import region_to_bbox
import siam_src.siamese as siam
# from siam_src.tracker_single import tracker_single
from siam_src.parse_arguments import parse_arguments
from tracker_options import Opts
from siam_src.tracker3 import tracker
import scipy.io as sio
import random
import math
import numpy as np
import h5py
import os
import sys
sys.path.append('../../')
from get_paths import get_paths



os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def get_templates_z(design, pos_x, pos_y, target_w, target_h,
                    frame_name_list, image, templates_z, filename, re_sz, sess):
    context = design.context * (target_w + target_h)
    z_sz = np.sqrt(np.prod((target_w + context) * (target_h + context)))
    # frame_name_list = imdb_video_path + obj[self.ind_frame].frame_path
    # with tf.Session() as sess:
    with sess.as_default():
        tf.global_variables_initializer().run()
        # Coordinate the loading of image files.
        # coord = tf.train.Coordinator()
        # threads = tf.train.start_queue_runners(coord=coord)
        image_, templates_z_ = sess.run([image, templates_z], feed_dict={
            siam.re_h:re_sz[0],
            siam.re_w:re_sz[1],
            siam.pos_x_ph: pos_x,
            siam.pos_y_ph: pos_y,
            siam.z_sz_ph: z_sz,
            filename: frame_name_list[0]})
    # # Finish off the filename queue coordinator.
    # coord.request_stop()
    # coord.join(threads)
    return templates_z_, z_sz
def compute_iou(boxA, boxB):
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
# cur_path = os.path.realpath(__file__).mysplit('/',-1)
cur_path = os.path.dirname(os.path.realpath(__file__))
print(cur_path)

data_paths = get_paths(cur_path +'/../../')
imdb_path = data_paths.imdb_path
imdb_video_path = data_paths.imdb_video_path
imdb_score_map_path = data_paths.imdb_score_map_path
# imdb_path="/home/ethan/dxp/ours/HP_optimization/imdb_video_0.1.mat"
imdb_name= data_paths.imdb_name
# imdb_video_path='/media/ethan/Data/DataSets/ILSVRC2015/Data/VID/train/'
# imdb_score_map_path = '/media/ethan/Data/DataSets/ILSVRC2015_score_map/Data/VID/train/'
imdb_ = sio.loadmat(imdb_path, struct_as_record=False, squeeze_me=True)
imdb = imdb_[imdb_name]
# num_video = imdb.path.size
num_video = 81
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
# print(os.path.realpath(__file__))

# gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.4)
# config.gpu_options.allow_growth = True
gpu_options = tf.GPUOptions(allow_growth = True)
            # with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
# up_path = os.path.abspath(os.path.join(os.getcwd(), os.path.pardir))
hp, evaluation, run, env, design = parse_arguments(parameters_path=cur_path+'/siam_src/')
final_score_sz = hp.response_up * (design.score_sz - 1) + 1
# build TF graph once for all
filename, image, templates_z, scores, scores_low = siam.build_tracking_graph(final_score_sz, design, env)#[None,None,None,None]#
pos_x, pos_y, target_w, target_h = [None,None,None,None]# region_to_bbox(gt)
frame_name_list = None #[imdb_video_path + obj[self.ind_frame].frame_path,
                   #imdb_video_path + obj[self.ind_frame + 1].frame_path]
templates_z_, z_sz = [None,None] # get_templates_z(design, pos_x, pos_y, target_w, target_h,
                                     #frame_name_list, image, templates_z, filename)
re_sz = None
track_opts = Opts(hp, run, design, frame_name_list, pos_x, pos_y, target_w, target_h,
                       final_score_sz, filename, image, templates_z, templates_z_, z_sz, scores,
                       re_sz, scores_low)

for ind_video in range(0,num_video):
    ind_trackid = 0
    while imdb.valid_trackids[ind_trackid, ind_video] != 0:
        valid_trackid = imdb.valid_per_trackid[ind_trackid, ind_video]
        if valid_trackid == []:
            continue
        else:
            valid_trackid -= 1
            obj = imdb.objects[ind_video][valid_trackid]
            ind_trackid += 1
            if ind_video==0:
                t_obj = type(obj)
            # print type(obj)
            if type(obj) != t_obj:
                continue
        save_path = imdb_score_map_path + os.path.dirname(obj[0].frame_path)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        file_name = os.path.join(save_path,'score_maps_{}.h5'.format(str(ind_trackid-1)))
        if os.path.exists(file_name):
            print('finished ind_video: ', ind_video, ' ind_trackid: ', ind_trackid)
            continue
        score_maps = np.zeros(shape=(obj.size, design.score_sz, design.score_sz),dtype=float)
        ious = np.zeros(shape=(obj.size, ),dtype=float)
        count_frame = 0
        for ind_frame in range(0,obj.size-1):
            frame_name_list = [imdb_video_path + obj[ind_frame].frame_path,
                           imdb_video_path + obj[ind_frame+1].frame_path]
            # ind_frame += 1
            gt = obj[ind_frame].extent
            gt = gt.astype(float)
            # gt = gt/scale_factor
            pos_x, pos_y, target_w, target_h = region_to_bbox(gt)

            # resize image if the target is too large
            max_target_sz = 200
            frame_sz = obj[ind_frame].frame_sz
            max_sz = max(target_h, target_w)
            if max_sz > max_target_sz:
                scale_factor = max_sz/max_target_sz
                re_sz = [int(frame_sz[0]/scale_factor),int(frame_sz[1]/scale_factor)]
                pos_x = math.floor(pos_x/scale_factor)
                pos_y = math.floor(pos_y/scale_factor)
                target_w = math.floor(target_w/scale_factor)
                target_h = math.floor(target_h/scale_factor)
                scale_factor = scale_factor
            else:
                re_sz = frame_sz
                scale_factor = 1.

            opts = track_opts
            opts.re_sz = re_sz
            #filename, image, templates_z, scores = siam.build_tracking_graph(opts.final_score_sz, opts.design, opts.env)
            templates_z_, z_sz = get_templates_z(opts.design, pos_x, pos_y, target_w, target_h,
                                                 frame_name_list, opts.image, opts.templates_z,
                                                 opts.filename, re_sz, sess)
            count = 0
            opts.hp = hp
            opts.templates_z_ = templates_z_
            opts.pos_y = pos_y
            opts.pos_x = pos_x
            opts.target_h = target_h
            opts.target_w = target_w
            opts.frame_name_list = frame_name_list
            opts.z_sz = z_sz
            # _, _, scores, image, _, _ = tracker(opts)
            bbox, _, scores, image, _, _, score_map = tracker(opts,sess)
            gt = gt/scale_factor
            iou = compute_iou(bbox[1], gt)
            score_maps[ind_frame,:,:] = score_map
            ious[ind_frame] = iou
            count_frame += 1
            if count_frame > 100:
                sess.close()
                sess = tf.Session()
                count_frame = 0
        sess.close()
        sess = tf.Session()

        # save_path = imdb_score_map_path + os.path.dirname(obj[0].frame_path)
        # file_name = os.path.join(save_path,'score_maps_{}.h5'.format(str(ind_trackid-1)))
        # Create a new file
        f = h5py.File(file_name, 'w')
        f.create_dataset('score_maps', data=score_maps)
        f.create_dataset('ious', data=ious)
        f.close()
        print('ind_video: ', ind_video, ' ind_trackid: ', ind_trackid, ' num_frames: ', obj.size)
        print('save_path: ',file_name)

        # # Load hdf5 dataset
        # f = h5py.File('data.h5', 'r')
        # X = f['X_train']
        # Y = f['y_train']
        # f.close()


