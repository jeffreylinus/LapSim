'''
A demo script for Acc and LapSim

Updates:
- start accelerating from 0
- changed velocity calculation
- added electric car option

'''

import numpy as np
from LapSim_v3 import LapSim
from acceleration import Acc
import matplotlib.pyplot as plt
import engine_and_trans_data as data

name = 'ktm_250_SX_F'                       #'ktm_250_SX_F''ktm_duke_200''honda_cbr_250R''yamaha_yz250f''kawasaki_ninja_250R_EXJ''emrax_208'
hybrid = 1                                  # 1-hybrid, 0-electric
run = 'lapsim'                                 # lapsim or acc

if run == 'acc':                        # accleration event
    
    acc = Acc.init_straight(steps=200, name=name, EM=33, m=200, hybrid=hybrid, track_len=500)
    acc.acc_time()

    print('Track length:',str('{0:.2f}'.format(acc.track_len/1000)),'km')
    print('ICE energy consumption:', str('{0:.2f}'.format(np.sum(acc.energy,axis=0)[0]/1000)), 'kJ')
    print('EM energy consumption:', str('{0:.2f}'.format(np.sum(acc.energy,axis=0)[1]/1000)), 'kJ')
    print('Total energy consumption:', str('{0:.2f}'.format(np.sum(acc.energy)/1000)), 'kJ')

    plt.show() 

elif run == 'lapsim':                   # endurance event

    lapsim = LapSim.init_ellipse(resolution=50, steps=100, name=name, EM=33, m=200, hybrid=hybrid) 
    lapsim.lap_time()

    print('Track length:',str('{0:.2f}'.format(lapsim.track_len/1000)),'km')
    print('Total lap time:',str('{0:.2f}'.format(lapsim.time)),'s')
    print('Average speed:',str('{0:.2f}'.format(np.mean(lapsim.v)*2.23)),'mph')
    print('ICE energy consumption:', str('{0:.2f}'.format(np.sum(lapsim.energy,axis=0)[0]/1000)), 'kJ')
    print('EM energy consumption:', str('{0:.2f}'.format(np.sum(lapsim.energy,axis=0)[1]/1000)), 'kJ')
    print('Total energy consumption:', str('{0:.2f}'.format(np.sum(lapsim.energy)/1000)), 'kJ')

    plt.show()




