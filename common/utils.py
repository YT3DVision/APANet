import time

import cv2
import torch
import os


def localtime():
    '''
    Get current time
    '''
    return time.strftime('%Y%m%d%H%M%S', time.localtime())


def save(net, name, state_dict=False, replace=False):
    """
    Save a network
    """
    root = "./saved_nets"
    if not os.path.exists(root):
        os.mkdir(root)
    if replace:
        if state_dict:
            torch.save(net.state_dict(),
                       f'./saved_nets/' + name + '.pkl')
        else:
            torch.save(net, f'./saved_nets/' + name + '.pkl')
    else:
        if state_dict:
            torch.save(net.state_dict(),
                       f'./saved_nets/net_state_{localtime()}_' + name + '.pkl')
        else:
            torch.save(net, f'./saved_nets/net_{localtime()}_' + name + '.pkl')


def restore(pkl_path, model_class=None):
    """
    Restore a network
    """
    if model_class is not None:
        try:
            model = model_class()
            return model.load_state_dict(torch.load(pkl_path))
        except:
            raise ValueError(
                'model_class must match with the model you want to restore')

    else:
        return torch.load(pkl_path)


def load_img(filepath):
    return cv2.cvtColor(cv2.imread(filepath), cv2.COLOR_BGR2RGB)


def save_img(filepath, img):
    cv2.imwrite(filepath, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
