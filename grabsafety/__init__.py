# -*- coding: UTF-8 -*-
"""
Grab safety repo
"""
import os


__author__ = 'yang zhengzhi'

PARENT_DIR = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
cutoff = 0.5

from .api import predict_danger
