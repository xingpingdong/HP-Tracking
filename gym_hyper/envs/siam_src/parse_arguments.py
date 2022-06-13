import json
from collections import namedtuple
#import global_path
import sys
import os
#sys.path.append(sys.path[0])
# os.chdir(sys.path[0])
# print(sys.path[0])

def parse_arguments(parameters_path=os.path.abspath(sys.path[0])+'/gym_hyper/envs/siam_src/',
                    in_hp={}, in_evaluation={}, in_run={}):
    # up_path = os.path.abspath(os.path.join(os.getcwd(), os.path.pardir))
    # up_path += '/'
    # parameters_path = '/home/ethan/dxp/ours/HP_optimization/python_interface/gym-hyper/'#global_path.parameters_path
    # parameters_path = sys.path[0]
    # print(sys.path[0])
    with open(parameters_path + 'parameters/hyperparams.json') as json_file:
        hp = json.load(json_file)
    with open(parameters_path + 'parameters/evaluation.json') as json_file:
        evaluation = json.load(json_file)
    with open(parameters_path + 'parameters/run.json') as json_file:
        run = json.load(json_file)
    with open(parameters_path + 'parameters/environment.json') as json_file:
        env = json.load(json_file)
    with open(parameters_path + 'parameters/design.json') as json_file:
        design = json.load(json_file)

    for name, value in in_hp.iteritems():
        hp[name] = value
    for name, value in in_evaluation.iteritems():
        evaluation[name] = value
    for name, value in in_run.iteritems():
        run[name] = value

    hp = namedtuple('hp', hp.keys())(**hp)
    evaluation = namedtuple('evaluation', evaluation.keys())(**evaluation)
    run = namedtuple('run', run.keys())(**run)
    env = namedtuple('env', env.keys())(**env)
    design = namedtuple('design', design.keys())(**design)

    return hp, evaluation, run, env, design
