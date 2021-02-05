import numpy as np
from LapSim_v3 import LapSim
import matplotlib.pyplot as plt

lapsim = LapSim.init_ellipse(resolution=50, steps=100) 

lapsim.lap_time()

print('Track length:',str(lapsim.track_len/1000),'km')
print('Total lap time:',str(lapsim.time),'s')
print('Average speed:',str(np.mean(lapsim.v)*2.23),'mph')

plt.show()




