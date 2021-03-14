'''
Testclass
'''

from motor import Motor
from engine import Engine
from car import Car

car = Car.init_config(filepath='Powertrain Part Options.xlsx', name_EM='Emrax 207', name_ICE='KTM 250 SX-F', hybrid=1)

# engine = Engine.init_from_file(engine_data='Powertrain Part Options.xlsx', name='KTM 250 SX-F')
# motor = Motor.init_from_file(motor_data='Powertrain Part Options.xlsx', name='Emrax 207')

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




