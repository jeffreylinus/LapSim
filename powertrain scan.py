'''
Iterate over all powertrain configurations

TODO:
Check lapsim shifting conditions
Rerun scan with correct score calculation
Rerun optimizer

'''

from motor import Motor
from engine import Engine
from car import Car

import numpy as np
from LapSim_v4 import LapSim                # 3D lapsim
from acceleration import Acc
import matplotlib.pyplot as plt
import sys, os
import pandas as pd
import time

scan_paramaters = np.array(['motors','engines','capacity_split','cell/cap'])

motors = pd.read_excel('data\\Powertrain Part Options.xlsx', sheet_name='EM Stats')
motor_name = motors['Engine Name'].values[:2]
engines = motors = pd.read_excel('data\\Powertrain Part Options.xlsx', sheet_name='ICE Stats')
engine_name = engines['Engine Name'].values[:2]
capacity_split = np.array([0.5])#,0.6,0.7,0.8])  # Total ICE energy: should be greater than 0.488
cell_cap = np.array(['bat','cap'])


# generate all configs
configs = np.array(np.meshgrid(motor_name, engine_name,capacity_split,cell_cap)).T.reshape(-1,4)
result = np.zeros((len(motor_name)*len(engine_name)*len(capacity_split)*len(cell_cap),3))         # results: acc time, lap time, lap no

for i,config in enumerate(configs):

    starttime = time.time()
    # sys.stdout = open(os.devnull, 'w')

    # get_power_split
    car0 = Car.init_config(filepath='data\\Powertrain Part Options.xlsx', name_EM=config[0], name_ICE=config[1], hybrid=1, capacity_split=config[2], acc_type=config[3])

    lapsim0 = LapSim.init_data(track_data='data\\Formula Hybrid Track Data.xlsx', steps=200, car=car0) 
    lapsim0.lap_time()

    lapsim0.plot_velocity()
    plt.show()

    # Power split calculation:
    # Actual E = r * total E * a (coefficient to be determined); for r0=0.5, r_new = a_ICE / (a_ICE+a_EM)
    E_engine = np.sum(lapsim0.energy,axis=0)[0]
    E_motor = np.sum(lapsim0.energy,axis=0)[1]
    # r = E_motor/(E_engine+E_motor)
    a_ICE = E_engine/((E_engine+E_motor)*car0.power_split)
    a_EM = E_motor/((E_engine+E_motor)*(1-car0.power_split))
    r = car0.capacity_split*a_EM/(a_ICE+car0.capacity_split*(a_EM-a_ICE))
    lap_no0 = car0.capacity*1E6/np.sum(lapsim0.energy)

    # simulate config
    car = Car.init_config(filepath='data\\Powertrain Part Options.xlsx', name_EM=config[0], name_ICE=config[1], hybrid=1, capacity_split=config[2], acc_type=config[3], power_split=r)

    # acceleration event
    acc = Acc.init_straight(steps=100, track_len=75, car=car)
    acc.acc_time()
    acc_time = acc.time

    # endurance event
    lapsim = LapSim.init_data(track_data='data\\Formula Hybrid Track Data.xlsx', steps=200, car=car) 
    lapsim.lap_time()
    lap_time = lapsim.time
    lap_no = car.capacity*1E6/np.sum(lapsim.energy)

    result[i] = [acc_time,lap_time,lap_no]

    E_engine = np.sum(lapsim.energy,axis=0)[0]
    E_motor = np.sum(lapsim.energy,axis=0)[1]
    r = E_motor/(E_engine+E_motor)

    sys.stdout = sys.__stdout__
    print('Car',i,'finished. Time elapsed:', str('{0:.3f}'.format(time.time()-starttime)),'Seconds. Acc. time ='\
        ,str('{0:.3f}'.format(acc_time)),'; Lap time =',str('{0:.3f}'.format(lap_time)),'; Lap number =',str('{0:.3f}'.format(lap_no)))

# score calculation
acc_time = result[:,0]
lap_time = result[:,1]
lap_no = result[:,2]
lap_no = np.clip(lap_no,a_min=0,a_max = 44)
lapsum = (np.round(lap_no)+1)*np.round(lap_no)/2

acc_score = 10+15+75*(np.min(acc_time)/acc_time)
maxavg = 105/44*60
endurance_score = 35+52.5+262.5*(lapsum/990)*((maxavg/lap_time-1)/(maxavg/np.min(lap_time)-1))
total_score = acc_score + endurance_score

df = pd.DataFrame({'Motor_Name': configs[:,0],'ICE_Name': configs[:,1],'ICE_Capacity_Ratio':configs[:,2], 'Accumulator_Type':configs[:,3],\
    'Acceleration_Time': acc_time,'Lap_Time': lap_time,'Lap_Number': lap_no, 'Score': total_score})
writer = pd.ExcelWriter('powertrains_event_scores_full.xlsx', engine='xlsxwriter')
df.to_excel(writer, sheet_name='Simulation')
writer.save()
print("Simulation saved.")

print('finished!')



