import numpy as np
from LapSim_v3 import LapSim
import matplotlib.pyplot as plt
import engine_and_trans_data as data

power_name = 'ktm_250_SX_F' #'ktm_250_SX_F''ktm_duke_200''honda_cbr_250R''yamaha_yz250f''kawasaki_ninja_250R_EXJ'
tran_name = 'ktm_250_SX_F_trans' #'ktm_250_SX_F_trans''ktm_duke_200_trans''honda_cbr_250R_trans''yamaha_yz250f_trans' 'kawasaki_ninja_250R_EXJ_trans'
power = data.get_power_data(power_name)
tran = data.get_trans_data(tran_name)
fuel = data.get_fuel_data()

lapsim = LapSim.init_ellipse(resolution=50, steps=100, power=power, EM=33, m=525, tran=tran, fuel=fuel) 

lapsim.lap_time()

print('Track length:',str('{0:.2f}'.format(lapsim.track_len/1000)),'km')
print('Total lap time:',str('{0:.2f}'.format(lapsim.time)),'s')
print('Average speed:',str('{0:.2f}'.format(np.mean(lapsim.v)*2.23)),'mph')
print('ICE energy consumption:', str('{0:.2f}'.format(np.sum(lapsim.energy,axis=0)[0]/1000)), 'kJ')
print('EM energy consumption:', str('{0:.2f}'.format(np.sum(lapsim.energy,axis=0)[1]/1000)), 'kJ')
print('Total energy consumption:', str('{0:.2f}'.format(np.sum(lapsim.energy)/1000)), 'kJ')

plt.show()




