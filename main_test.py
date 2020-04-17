#!/usr/bin/env python
#coding:utf-8
"""
  Author : Chao Wang -- (chaosimpler@gmail.com)
  License: GNU GPL v3
  Created: 2020/4/16
  Purpose:
    for test only
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from nilmtk.disaggregate.hmm import test_hmm
from nilmtk import DataSet, MeterGroup




redd_file = r'E:\Dataset\REDD\rd.h5'

if __name__ == '__main__':
    
    data_file = redd_file
    
    all_ds = DataSet(data_file)
    train_ds = DataSet(data_file)
    test_ds = DataSet(data_file)
    building = 1
    
    train_ds.set_window(end='2011-04-30')
    test_ds.set_window(start='2011-05-08')
    
    all_elec = all_ds.buildings[1].elec
    train_elec = train_ds.buildings[1].elec
    test_elec = test_ds.buildings[1].elec
    
    
    test_hmm(train_elec, test_elec)
    
    
    print('over')    
    print('over')