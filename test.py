'''
A demo script for Acc and LapSim

Updates:
- start accelerating from 0
- changed velocity calculation
- added electric car option

make cars outside of acc/lapsim
add car.hybrid -- powertrain type: hybrid, gas, electric
electric car has 2 motors

check EM power for lapsim

TODO:
+ elevation 
 - traction
 - acceleration
 - curvature
+ continuous sim 
 - add fuel capacity (4449 Wh)
 - gas: 2343 Wh/Liter
 - sim until run out of fuel
tyre data

chapter 14
chapter 17 (suspension)

torque splitting for efficiency

drag coefficient

series vs parallel
- FH total capacity: 
- fix fuel tank, fix accummulator capcity
Series:
- charging speed: generator, accummulator (cpacity, current)





'''

import numpy as np
from LapSim_v3 import LapSim
from acceleration import Acc
import matplotlib.pyplot as plt
import engine_and_trans_data as data

name = 'emrax_208_2'                       #'ktm_250_SX_F''ktm_duke_200''honda_cbr_250R''yamaha_yz250f''kawasaki_ninja_250R_EXJ''emrax_208'
hybrid = 0                                  # 1-hybrid, 0-electric
run = 'lapsim'                                 # lapsim or acc

if run == 'acc':                        # accleration event
    
    acc = Acc.init_straight(steps=200, name=name, EM=0, m=220, hybrid=hybrid, track_len=300)
    acc.acc_time()

    print('Track length:',str('{0:.2f}'.format(acc.track_len/1000)),'km')
    print('ICE energy consumption:', str('{0:.2f}'.format(np.sum(acc.energy,axis=0)[0]/1000)), 'kJ')
    print('EM energy consumption:', str('{0:.2f}'.format(np.sum(acc.energy,axis=0)[1]/1000)), 'kJ')
    print('Total energy consumption:', str('{0:.2f}'.format(np.sum(acc.energy)/1000)), 'kJ')

    plt.show() 

elif run == 'lapsim':                   # endurance event

    lapsim = LapSim.init_ellipse(resolution=50, steps=100, name=name, EM=0, m=250, hybrid=hybrid) 
    # lapsim = LapSim.init_data(track_data='Formula Hybrid Track Data.xlsx', steps=100, name=name, EM=0, m=500, hybrid=hybrid) 
    lapsim.lap_time()

    print('Track length:',str('{0:.2f}'.format(lapsim.track_len/1000)),'km')
    print('Total lap time:',str('{0:.2f}'.format(lapsim.time)),'s')
    print('Average speed:',str('{0:.2f}'.format(np.mean(lapsim.v)*2.23)),'mph')
    print('ICE energy consumption:', str('{0:.2f}'.format(np.sum(lapsim.energy,axis=0)[0]/1000)), 'kJ')
    print('EM energy consumption:', str('{0:.2f}'.format(np.sum(lapsim.energy,axis=0)[1]/1000)), 'kJ')
    print('Total energy consumption:', str('{0:.2f}'.format(np.sum(lapsim.energy)/1000)), 'kJ')

    plt.show()




