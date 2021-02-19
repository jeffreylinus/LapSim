import numpy as np
from LapSim_v3 import LapSim
import matplotlib.pyplot as plt
import engine_and_trans_data as data

power = data.get_power_data('ktm_250_SX_F')
tran = data.get_trans_data('ktm_250_SX_F_trans')

lapsim = LapSim.init_ellipse(resolution=50, steps=100, power=power, EM=33, m=525, tran=tran) 

lapsim.lap_time()

print('Track length:',str(lapsim.track_len/1000),'km')
print('Total lap time:',str(lapsim.time),'s')
print('Average speed:',str(np.mean(lapsim.v)*2.23),'mph')

plt.show()




