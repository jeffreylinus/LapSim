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
# v = [0.5,0.14182157324195305]
v = [0.5,0.2]
plot = 0
score_list = np.array([])

def score(v, EM, ICE, plot):
    '''
    Run acceleration sim and lap sim, calculate event scores
    '''

    sys.stdout = open(os.devnull, 'w')
    # data from parametric scan run
    min_acc_time = 6.182742085
    min_lap_time = 46.8694933
    maxavg = 105/44*60
    max_lapsum = 990

    # build car
    car = Car.init_config(filepath='data\\Powertrain Part Options.xlsx', name_EM=EM, name_ICE=ICE, hybrid=1, capacity_split=v[0], power_split=v[1], acc_type='bat')

    if car.capacity_split<=0.488:
        return 0
    if car.power_split<=0:
        return 0
    if car.capacity_split>=1:
        return 0
    if car.power_split>=1:
        return 0

    # run sims
    acc = Acc.init_straight(steps=100, track_len=75, car=car)
    acc.acc_time()
    
    # car.motor.torque_con = car.motor.torque_con*0.5
    # car.motor.torque_max = car.motor.torque_max*0.5
    # car.motor.power_nom = car.motor.power_nom*0.5
    # car.motor.power_max = car.motor.power_max*0.5
    lapsim = LapSim.init_data(track_data='data\\Formula Hybrid Track Data.xlsx', steps=200, car=car) 
    lapsim.lap_time()
    

    # calculate score
    E_engine = np.sum(lapsim.energy,axis=0)[0]
    E_motor = np.sum(lapsim.energy,axis=0)[1]
    lap_no_motor = car.motor.capacity*1E6/E_motor
    lap_no_engine = car.engine.capacity*1E6/E_engine

    lap_no = car.capacity*1E6/(E_engine+E_motor)

    # if lap_no_motor > lap_no_engine:
    #     lap_no_0 = lap_no_engine
    #     E_motor_only = E_motor / (1-car.power_split)
    #     lap_1 = (car.capacity*1E6 - lap_no_0 * (E_engine+E_motor))/E_motor_only
    # elif lap_no_motor <= lap_no_engine:
    #     lap_no_0 = lap_no_motor
    #     E_engine_only = E_engine / car.power_split
    #     lap_1 = (car.capacity*1E6 - lap_no_0 * (E_engine+E_motor))/E_engine_only
    # lap_no = lap_no_0 + lap_1

    if lap_no > 44:
        r = 44/lap_no
        new_cap = car.capacity * r

        car = Car.init_config(filepath='data\\Powertrain Part Options.xlsx', name_EM=EM, name_ICE=ICE, hybrid=1, capacity = new_cap, capacity_split=v[0], power_split=v[1], acc_type='bat')

        # run sims
        acc = Acc.init_straight(steps=100, track_len=75, car=car)
        acc.acc_time()
        
        # car.motor.torque_con = car.motor.torque_con*0.5
        # car.motor.torque_max = car.motor.torque_max*0.5
        # car.motor.power_nom = car.motor.power_nom*0.5
        # car.motor.power_max = car.motor.power_max*0.5
        lapsim = LapSim.init_data(track_data='data\\Formula Hybrid Track Data.xlsx', steps=200, car=car) 
        lapsim.lap_time()
        

        # calculate score
        E_engine = np.sum(lapsim.energy,axis=0)[0]
        E_motor = np.sum(lapsim.energy,axis=0)[1]
        lap_no_motor = car.motor.capacity*1E6/E_motor
        lap_no_engine = car.engine.capacity*1E6/E_engine

        lap_no = car.capacity*1E6/(E_engine+E_motor)

        # if lap_no_motor > lap_no_engine:
        #     lap_no_0 = lap_no_engine
        #     E_motor_only = E_motor / (1-car.power_split)
        #     lap_1 = (car.capacity*1E6 - lap_no_0 * (E_engine+E_motor))/E_motor_only
        # elif lap_no_motor <= lap_no_engine:
        #     lap_no_0 = lap_no_motor
        #     E_engine_only = E_engine / car.power_split
        #     lap_1 = (car.capacity*1E6 - lap_no_0 * (E_engine+E_motor))/E_engine_only
        # lap_no = lap_no_0 + lap_1

    lapsum = (np.round(lap_no)+1)*np.round(lap_no)/2

    acc_time = acc.time
    lap_time = lapsim.time

    acc_score = 10+15+75*(min_acc_time/acc_time)
    endurance_score = 35+52.5+262.5*(lapsum/max_lapsum)*((maxavg/lap_time-1)/(maxavg/min_lap_time-1))
    total_score = acc_score + endurance_score

    sys.stdout = sys.__stdout__

    print('Capacity split =', str(v[0]), 'Power split =', str(v[1]), 'E_total =',str('{0:.3f}'.format(E_motor+E_engine)),'; Acc. time =',str('{0:.3f}'.format(acc_time)),'; Lap time =',\
        str('{0:.3f}'.format(lap_time)),'; Lap number =',str('{0:.3f}'.format(lap_no)),'; Total Score =', total_score)

    if plot == 1:
        lapsim.plot_velocity()
        acc.plot_velocity()


    return -total_score


eps = [1E-02, 1E-01]
mtd = 'TNC'

init_score = score(v, EM, ICE, 1)
# plt.show()

starttime = time.time()

# optimize
res = minimize(score, v, args=(EM, ICE, plot), method=mtd, options={'gtol':10, 'disp': True, 'eps': eps})

print("Total time elapsed: ", time.time()-starttime," Seconds")

new_score = score(res.x, EM, ICE, 1)

plt.show()

print('finished! Final capacity/power splits =', res.x)



