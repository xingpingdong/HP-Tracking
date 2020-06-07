import json
from collections import namedtuple
#import global_path


def get_paths(file_path = '.'):
    # up_path = os.path.abspath(os.path.join(os.getcwd(), os.path.pardir))
    # up_path += '/'
    in_paths={}
    with open(file_path+'/data_paths.json') as json_file:
        data_paths = json.load(json_file)


    for name, value in in_paths.iteritems():
        data_paths[name] = value

    data_paths = namedtuple('data_paths', data_paths.keys())(**data_paths)


    return data_paths
