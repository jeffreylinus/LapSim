'''
Optimization over capacity split and fixed power split

'''

from car import Car
from LapSim_v4 import LapSim                
from acceleration import Acc

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import sys, os
import pandas as pd
import time

EM = "Agni 95"
ICE = "KTM 250 SX-F"
v = [0.6,0.6]
plot = 0


def score(v, EM, ICE, plot):

    sys.stdout = open(os.devnull, 'w')
    # data from parametric scan run
    min_acc_time = 6.182742085
    min_lap_time = 50.55854895
    maxavg = 105/44*60
    max_lapsum = 990

    # build car
    car = Car.init_config(filepath='data\\Powertrain Part Options.xlsx', name_EM=EM, name_ICE=ICE, hybrid=1, capacity_split=v[0], power_split=v[1], acc_type='bat')
    
    # run sims
    lapsim = LapSim.init_data(track_data='data\\Formula Hybrid Track Data.xlsx', steps=200, car=car) 
    lapsim.lap_time()
    acc = Acc.init_straight(steps=100, track_len=75, car=car)
    acc.acc_time()

    # calculate score
    E_engine = np.sum(lapsim.energy,axis=0)[0]
    E_motor = np.sum(lapsim.energy,axis=0)[1]
    lap_no_motor = car.motor.capacity*1E6/E_motor
    lap_no_engine = car.engine.capacity*1E6/E_engine
    lap_no = min([lap_no_motor, lap_no_engine])
    lapsum = [(np.round(i)+1)*np.round(i)/2 for i in lap_no]

    acc_time = acc.time
    lap_time = lapsim.time

    acc_score = 10+15+75*(min_acc_time/acc_time)
    endurance_score = 35+52.5+262.5*(lapsum/max_lapsum)*((maxavg/lap_time-1)/(maxavg/min_lap_time-1))
    total_score = acc_score + endurance_score

    sys.stdout = sys.__stdout__

    print('Capacity split =', str(v[0]), 'Power split =', str(v[1]), '; Acc. time =',str('{0:.3f}'.format(acc_time)),'; Lap time =',\
        str('{0:.3f}'.format(lap_time)),'; Lap number =',str('{0:.3f}'.format(lap_no)),'; Total Score =', total_score)

    if plot == 1:
        lapsim.plot_velocity()
        acc.plot_velocity()
        plt.show()

    return -total_score


init_score = score(v,EM, ICE, 1)

starttime = time.time()

# optimize
res = minimize(score, v, args=(EM, ICE, plot), method='TNC', options={'gtol':1, 'disp': True, 'eps': 1E-01}, bounds=((0.488, 0.9), (0.05, 0.95)))

print("Total time elapsed: ", time.time()-starttime," Seconds")

new_score = score(res.x, EM, ICE, 1)

print('finished! Final capacity/power splits =', res.x)



