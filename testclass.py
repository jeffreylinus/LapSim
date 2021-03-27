'''
Testclass

1. Choose engine/motor pair: run engine/motor pairs with e.g. 10/90, 30/70, 50/50 em-ice split. For weights - go by nominal voltage of em/energy limits in rules


2. Choose torque control strategy - e.g. 10/90....
    - maybe also test strategies while varying accumulator weight? 
        - q: can we assume we can get any arbitrary weight
        - when varying weight -> how does voltage, and consequently motor performance vary?
    - to measure: total distance travelled, fastest/avg lap times


'''

from motor import Motor
from engine import Engine
from car import Car

import numpy as np
# from LapSim_v3 import LapSim                # 2D lapsim
from LapSim_v4 import LapSim                # 3D lapsim
from acceleration import Acc
import matplotlib.pyplot as plt

# car = Car.init_config(filepath='data\\Powertrain Part Options.xlsx', name_EM='Emrax 207', name_ICE='KTM 250 SX-F', hybrid=1)
# car = Car.init_config(filepath='data\\Powertrain Part Options.xlsx', name_EM='Emrax 228', name_ICE='KTM 250 SX-F', hybrid=1, mu=0.6)
car = Car.init_config(filepath='data\\Powertrain Part Options.xlsx', name_EM='Emrax 207', hybrid=0, mu=0.4)
# car = Car.init_config(filepath='data\\Powertrain Part Options.xlsx', name_EM='Saietta 119R', hybrid=0)

run = 'lapsim'                                 # lapsim or acc

if run == 'acc':                        # accleration event
    
    acc = Acc.init_straight(steps=100, EM=0, m=350, track_len=150, car=car)
    acc.acc_time()

    print('Track length:',str('{0:.2f}'.format(acc.track_len/1000)),'km')
    print('ICE energy consumption:', str('{0:.2f}'.format(np.sum(acc.energy,axis=0)[0]/1000)), 'kJ')
    print('EM energy consumption:', str('{0:.2f}'.format(np.sum(acc.energy,axis=0)[1]/1000)), 'kJ')
    print('Total energy consumption:', str('{0:.2f}'.format(np.sum(acc.energy)/1000)), 'kJ')

    plt.show() 

elif run == 'lapsim':                   # endurance event

    # lapsim = LapSim.init_ellipse(resolution=50, steps=100, car=car) 
    lapsim = LapSim.init_data(track_data='data\\Formula Hybrid Track Data.xlsx', steps=200, car=car) 
    lapsim.lap_time()

    print('Track length:',str('{0:.2f}'.format(lapsim.track_len/1000)),'km')
    print('Total lap time:',str('{0:.2f}'.format(lapsim.time)),'s')
    print('Average speed:',str('{0:.2f}'.format(np.mean(lapsim.v)*2.23)),'mph')
    print('ICE energy consumption:', str('{0:.2f}'.format(np.sum(lapsim.energy,axis=0)[0]/1000)), 'kJ')
    print('EM energy consumption:', str('{0:.2f}'.format(np.sum(lapsim.energy,axis=0)[1]/1000)), 'kJ')
    print('Total energy consumption:', str('{0:.2f}'.format(np.sum(lapsim.energy)/1000)), 'kJ')

    plt.show()

print('finished!')



