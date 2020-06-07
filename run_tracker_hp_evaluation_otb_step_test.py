from __future__ import division
import sys
import os
import numpy as np
from PIL import Image

up_path0 = os.path.abspath(os.path.dirname(os.getcwd()))
# up_path = os.path.join(up_path0,'gym_hyper/envs')
# sys.path.insert(0, up_path0)
import gym_hyper.envs.siam_src.siamese0 as siam
# from src.tracker import tracker
from gym_hyper.envs.siam_src.parse_arguments import parse_arguments
from gym_hyper.envs.siam_src.region_to_bbox import region_to_bbox
from gym_hyper.envs.siam_src.tracker_hp_adjust_norm import tracker
# import hp_opt.load_models as load_models
# import gym
# import gym_hyper
from keras.models import load_model
from keras.backend import manual_variable_initialization
import tensorflow as tf
from keras import backend as K


from get_paths import get_paths
data_paths = get_paths('./')
dataset_folder = data_paths.OTB_path
# sys.path.insert(0,up_path0)
# print(sys.path[0])

def main():
    # avoid printing TF debugging information
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    K.set_learning_phase(0)
    # gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)
    gpu_options = tf.GPUOptions(allow_growth=True)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    # TODO: allow parameters from command line or leave everything in json files?
    hp, evaluation, run, env, design = parse_arguments()
    # Set size for use with tf.image.resize_images with align_corners=True.
    # For example,
    #   [1 4 7] =>   [1 2 3 4 5 6 7]    (length 3*(3-1)+1)
    # instead of
    # [1 4 7] => [1 1 2 3 4 5 6 7 7]  (length 3*3)
    # ENV_NAME = 'hypersiamese-v0'
    # gym.undo_logger_setup()
    # # Get the environment and extract the number of actions.
    # gym_env = gym.make(ENV_NAME)
    manual_variable_initialization(True)
    final_score_sz = hp.response_up * (design.score_sz - 1) + 1
    # build TF graph once for all
    filename, image, templates_z, scores, score_low = siam.build_tracking_graph(final_score_sz, design, env)
    # filename, image, templates_z, scores = [gym_env.track_opts.filename, gym_env.track_opts.image, gym_env.track_opts.templates_z, gym_env.track_opts.scores]

    # model_path = "/home/ethan/dxp/ours/HP_optimization/python_interface/gym-hyper/cdqn_hypersiamese-v0_large_conv_weights.h5f"
    # mu_model = load_models.load_mu_model_large(hp=hp,design=design,model_path=model_path)
    # mu_model = load_model("/home/ethan/dxp/ours/HP_optimization/python_interface/gym-hyper/backup/mu_model_50500.h5")
    # mu_model = load_model("/home/ethan/dxp/ours/HP_optimization/python_interface/gym-hyper/models/batch_size16_sgd_3conv_3fc_mu_model.h5")
    # root_path = "/media/dxp/DATA/ourworks/hp_backup/tmp_ok/"
    root_path = './tmp/'
    model_name = 'sigma1.0_batch_size128_enlarge_action_3conv_3fc_punish_reward1'
    i = 2000
    # mu_path = "tmp_test/no_fix_mu_{}".format(model_name)+"_mu_model_steps_{:06d}.h5".format(i)
    # mu_model = load_model(root_path+mu_path)
    low_hp = np.array([1.02, 0.9, 0.4, 0.1, 0.])
    high_hp = np.array([1.08, 1., 1., 0.5, 0.05])
    scale_range = high_hp - low_hp
    min_range = 0.
    max_range = 1.
    save_path = 'res'
    isExists = os.path.exists(save_path)
    if not isExists:
        os.makedirs(save_path)
    from collections import namedtuple
    u_stype = 'u5'
    Env = namedtuple('Env', 'low_hp, scale_range, min_range, max_range, u_stype')
    gym_env = Env(low_hp, scale_range, min_range, max_range,u_stype)
    res_text = '{}/otb_{}.txt'.format(save_path, model_name)
    # txt_file = open(res_text, 'a+')

    for steps in range(2000, 24100, i):
        mu_path = "{}".format(model_name) + "_mu_model_steps_{:06d}.h5".format(steps)
        mu_model = load_model(root_path + mu_path)
        save_path = 'res/res_otb_{}_steps_{:06d}.npz'.format(model_name, steps)
        print 'steps_{:06d}'.format(steps)
        # iterate through all videos of evaluation.dataset
        if evaluation.video == 'all':
            # dataset_folder = os.path.join(env.root_dataset, evaluation.dataset)
            # dataset_folder = '/media/dxp/DATA/datasets/OTB/'
            # videos_list = [v for v in os.listdir(dataset_folder) if v != "Jogging"]
            videos_list = [v for v in os.listdir(dataset_folder) if '.' not in v]
            # videos_list.append("Jogging1")
            # videos_list.append("Jogging2")
            videos_list.sort()
            nv = np.size(videos_list)
            speed = np.zeros(nv * evaluation.n_subseq)
            precisions = np.zeros(nv * evaluation.n_subseq)
            precisions_auc = np.zeros(nv * evaluation.n_subseq)
            ious = np.zeros(nv * evaluation.n_subseq)
            lengths = np.zeros(nv * evaluation.n_subseq)
            for i in range(nv):
                # print videos_list[i]
                gt, frame_name_list, frame_sz, n_frames = _init_video(env, evaluation, videos_list[i])
                starts = np.rint(np.linspace(0, n_frames - 1, evaluation.n_subseq + 1))
                starts = starts[0:evaluation.n_subseq]
                for j in range(evaluation.n_subseq):
                    start_frame = int(starts[j])
                    gt_ = gt[start_frame:, :]
                    frame_name_list_ = frame_name_list[start_frame:]
                    pos_x, pos_y, target_w, target_h = region_to_bbox(gt_[0])
                    idx = i * evaluation.n_subseq + j
                    bboxes, speed[idx] = tracker(hp, run, design, frame_name_list_, pos_x, pos_y,
                                                 target_w, target_h, final_score_sz, filename,
                                                 image, templates_z, scores, start_frame,
                                                 mu_model, gym_env, score_low)
                    lengths[idx], precisions[idx], precisions_auc[idx], ious[idx] = _compile_results(gt_, bboxes,
                                                                                                     evaluation.dist_threshold)
                    print str(i) + ' -- ' + videos_list[i] + \
                          ' -- Precision: ' + "%.2f" % precisions[idx] + \
                          ' -- Precisions AUC: ' + "%.2f" % precisions_auc[idx] + \
                          ' -- IOU: ' + "%.2f" % ious[idx] + \
                          ' -- Speed: ' + "%.2f" % speed[idx] + ' --'
                    print
                model_path = os.path.join('results', 'OTB100', 'step_{:06d}'.format(steps))
                if not os.path.isdir(model_path):
                    os.makedirs(model_path)
                result_path = os.path.join(model_path, '{}.txt'.format(videos_list[i]))
                with open(result_path, 'w') as f:
                    for x in bboxes:
                        f.write(','.join([str(i) for i in x]) + '\n')

            tot_frames = np.sum(lengths)
            mean_precision = np.sum(precisions * lengths) / tot_frames
            mean_precision_auc = np.sum(precisions_auc * lengths) / tot_frames
            mean_iou = np.sum(ious * lengths) / tot_frames
            mean_speed = np.sum(speed * lengths) / tot_frames
            mp = np.mean(precisions)
            m_iou = np.mean(ious)
            print '-- Overall stats (averaged per frame) on ' + str(nv) + ' videos (' + str(tot_frames) + ' frames) --'
            print ' -- Precision ' + "(%d px)" % evaluation.dist_threshold + ': ' + "%.2f" % mean_precision + \
                  ' -- Precisions AUC: ' + "%.2f" % mean_precision_auc + \
                  ' -- IOU: ' + "%.2f" % mean_iou + \
                  ' -- mp: ' + "%.2f" % mp + \
                  ' -- m_iou: ' + "%.2f" % m_iou + \
                  ' -- Speed: ' + "%.2f" % mean_speed + ' --'
            print
            with open(save_path, 'wb+') as fh:
                np.savez(save_path, precisions, precisions_auc, ious, mp, m_iou, speed)
            # print(txt_file,'-- Overall stats (averaged per frame) on ' + str(nv) + ' videos (' + str(tot_frames) + ' frames) --\n')
            with open(res_text, 'a+') as txt_file:
                if steps == 2000:
                    txt_file.write('-- Overall stats (averaged per frame) on ' + str(nv) + ' videos (' + str(
                        tot_frames) + ' frames) --\n')
                txt_file.write("steps_{:06d}\n".format(steps))
                txt_file.write(' -- Precision ' + "(%d px)" % evaluation.dist_threshold + ': ' + "%.2f" % mean_precision + \
                               ' -- Precisions AUC: ' + "%.2f" % mean_precision_auc + \
                               ' -- IOU: ' + "%.2f" % mean_iou + \
                               ' -- mp: ' + "%.2f" % mp + \
                               ' -- m_iou: ' + "%.2f" % m_iou + \
                               ' -- Speed: ' + "%.2f" % mean_speed + ' --\n')

        else:
            gt, frame_name_list, _, _ = _init_video(env, evaluation, evaluation.video)
            pos_x, pos_y, target_w, target_h = region_to_bbox(gt[evaluation.start_frame])
            bboxes, speed = tracker(hp, run, design, frame_name_list, pos_x, pos_y, target_w, target_h, final_score_sz,
                                    filename, image, templates_z, scores, evaluation.start_frame,
                                    mu_model, gym_env)
            _, precision, precision_auc, iou = _compile_results(gt, bboxes, evaluation.dist_threshold)
            print evaluation.video + \
                  ' -- Precision ' + "(%d px)" % evaluation.dist_threshold + ': ' + "%.2f" % precision + \
                  ' -- Precision AUC: ' + "%.2f" % precision_auc + \
                  ' -- IOU: ' + "%.2f" % iou + \
                  ' -- Speed: ' + "%.2f" % speed + ' --'
            print


def _compile_results(gt, bboxes, dist_threshold):
    l = np.size(bboxes, 0)
    gt4 = np.zeros((l, 4))
    new_distances = np.zeros(l)
    new_ious = np.zeros(l)
    n_thresholds = 50
    precisions_ths = np.zeros(n_thresholds)

    for i in range(l):
        gt4[i, :] = region_to_bbox(gt[i, :], center=False)
        new_distances[i] = _compute_distance(bboxes[i, :], gt4[i, :])
        new_ious[i] = _compute_iou(bboxes[i, :], gt4[i, :])

    # what's the percentage of frame in which center displacement is inferior to given threshold? (OTB metric)
    precision = sum(new_distances < dist_threshold) / np.size(new_distances) * 100

    # find above result for many thresholds, then report the AUC
    thresholds = np.linspace(0, 25, n_thresholds + 1)
    thresholds = thresholds[-n_thresholds:]
    # reverse it so that higher values of precision goes at the beginning
    thresholds = thresholds[::-1]
    for i in range(n_thresholds):
        precisions_ths[i] = sum(new_distances < thresholds[i]) / np.size(new_distances)

    # integrate over the thresholds
    precision_auc = np.trapz(precisions_ths)

    # per frame averaged intersection over union (OTB metric)
    iou = np.mean(new_ious) * 100

    return l, precision, precision_auc, iou


def _init_video(env, evaluation, video):
    # dataset_folder = '/media/dxp/DATA/datasets/OTB/'
    # if video == "Jogging1":
    #     video = "Jogging"
    #     gt_file = os.path.join(dataset_folder, video, 'groundtruth_rect.1.txt')
    # elif video == "Jogging2":
    #     video = "Jogging"
    #     gt_file = os.path.join(dataset_folder, video, 'groundtruth_rect.2.txt')
    # else:
    #     gt_file = os.path.join(dataset_folder, video, 'groundtruth_rect.txt')
    gt_file = os.path.join(dataset_folder, video, 'groundtruth_rect.txt')
    video_folder = os.path.join(dataset_folder, video, 'img')
    # video_folder = os.path.join(env.root_dataset, evaluation.dataset, video)
    frame_name_list = [f for f in os.listdir(video_folder) if (f.endswith(".jpg") or f.endswith(".png"))]
    frame_name_list = [os.path.join(dataset_folder, video, 'img', '') + s for s in frame_name_list]
    frame_name_list.sort()
    if video == "David":
        frame_name_list = frame_name_list[299:]
    elif video == "Freeman3":
        frame_name_list = frame_name_list[:460]
    elif video == "Freeman4":
        frame_name_list = frame_name_list[:283]
    elif video == "Football1":
        frame_name_list = frame_name_list[:74]
    elif video == "Diving":
        frame_name_list = frame_name_list[:215]
    with Image.open(frame_name_list[0]) as img:
        frame_sz = np.asarray(img.size)
        frame_sz[1], frame_sz[0] = frame_sz[0], frame_sz[1]

    # read the initialization from ground truth

    # gt_file = os.path.join(video_folder, 'groundtruth_rect.txt')
    gt = np.genfromtxt(gt_file, delimiter=',')
    if gt.shape[0] == gt.size:
        gt = np.loadtxt(gt_file)
    n_frames = len(frame_name_list)
    assert n_frames == len(gt), 'Number of frames and number of GT lines should be equal.'

    return gt, frame_name_list, frame_sz, n_frames


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


if __name__ == '__main__':
    sys.exit(main())
