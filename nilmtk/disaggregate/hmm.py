#!/usr/bin/env python
#coding:utf-8
"""
  Author : Chao Wang -- (chaosimpler@gmail.com)
  License: GNU GPL v3
  Created: 2020/4/16
  Purpose:
    
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#from nilmtk import DataSet, TimeFrame, MeterGroup, HDFDataStore
from .. import DataSet, MeterGroup

redd_file = r'E:\Dataset\REDD\rd.h5'

#----------------------------------------------------------------------
def test_hmm(train_mg, test_mg):
    """
    Args:
        trans_mat (ndarray, shape=(sample_size,sample_size)): transition matrix;
        num_simulations (int, optional): max iteration number, default = 10000；
        eps (float, optional): convergence threshold, default = 1e-10;
        restart_vector (array, shape=(sample_size,), optional): restart vector, default = None;
        damping_factor (float, optional): damping factor, default = 0.15;
        Returns : 
        arr_vect (array, shape=(sample_size,)): visiting probabilitiy of each sample after the
        random walk process reaches equilibrium.
    """
    from hmmlearn import hmm
    from sklearn.metrics import mean_squared_error
    num_total_states = 10
    sample_period = 120
    
    train_elec = train_mg.submeters().select_top_k(k=1)
    test_elec = test_mg.submeters().select_top_k(k=1)
    
    train_power = train_elec.power_series_all_data(sample_period=sample_period).dropna()
    train_data =train_power.values.reshape((-1, 1))
    test_power = test_elec.power_series_all_data(sample_period=sample_period).dropna()
    test_data = test_power.values.reshape((-1, 1))
    
    m = hmm.GaussianHMM(num_total_states, "full", n_iter=1000)
    m.fit(train_data)
    #s = m.predict(train_data)
    
    def draw_predict(test_power):
        test_data = test_power.values.reshape((-1, 1))
        pred_state = m.predict(test_data)
        
        means = m.means_.reshape((num_total_states, ))
        means.sort()
        dmeans = dict(zip(set(pred_state), means))
        pred_value = [dmeans[i] for i in pred_state]
        
        pd_gt = pd.DataFrame(test_data, columns=['GT'], index=test_power.index)
        pd_pred = pd.DataFrame(pred_value, columns=['Pred'], index=test_power.index)
        
        plt.ioff()
        start = 0
        end = -1
        
        #plt.cla()
        #plt.clf()        
        ax = pd_gt[start:end].plot()
        pd_pred[start:end].plot(ax=ax, alpha=0.6)
        #plt.show()
        
        rmse = np.sqrt(mean_squared_error(test_data, pred_value))
        plt.xlabel('RMSE={}'.format(rmse))
    
    draw_predict(train_power)
    plt.title('train data')
    plt.show()
    draw_predict(test_power)
    plt.title('test data')
    plt.show()    
    ax = train_elec.plot()
    
    
    print('over')    
    

if __name__ == '__main__':
    
    train = DataSet(data_file)
    test = DataSet(data_file)
    building = 1
    
    train.set_window(end='2011-04-30')
    test.set_window(start='2011-04-30')
    
    train_elec = train.buildings[1].elec
    test_elec = test.buildings[1].elec
    
    
    test_hmm(train_elec.submeters().select_top_k(k=1), test_elec)
    
    
    print('over')



