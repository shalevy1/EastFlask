# -*-coding:utf-8 -*-
# Reference:**********************************************
# @Time     : 2020-04-03 21:13
# @Author   : Fabrice LI
# @File     : commons.py
# @User     : liyihao
# @Software : PyCharm
# @Description: todo
# Reference:**********************************************

import torch
from vgg import EAST as VGG
import config


def get_model(device):
    model_path = config.PATH_PTHS + config.MODEL
    model = VGG(pretrained=False).to(device)
    model.load_state_dict(torch.load(model_path, map_location=None if torch.cuda.is_available() else "cpu"))
    model.eval()
    return model


