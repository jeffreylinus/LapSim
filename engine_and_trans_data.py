import numpy as np

def get_power_data(name):

    #revolutions per minute, horsepower 
    #power curve data
    ktm_250_SX_F = [(4000,7.5), (5000, 11), (6000, 17.5), (7000, 24), (8000, 27), (9000, 32.5), (10000, 35), (11000, 37.5), (12000, 40), (13000, 40)]
    ktm_duke_200 = [(3000, 5), (4000, 7.5), (5000, 9), (6000, 12), (7000, 16), (8000, 20), (9000, 20), (10000, 22.2)]
    honda_cbr_250R = [(2000, 2), (3000, 5), (4000, 9), (5000, 14), (6000, 19), (7000, 22), (8000, 26), (9000, 28)]
    yamaha_yz250f = [(4000, 10), (5000, 12.5), (6000, 15), (7000, 22.5), (8000, 26.5), (9000, 31), (10000, 34), (11000, 36), (12000, 37.5)]
    briggs_stratton_world_formula_204 = [(3000, 5.5), (4000, 7.5), (5000, 10), (6000, 12.5), (7000, 12.5)]
    kawasaki_ninja_250R_EXJ = [(3000, 4), (4000, 6), (5000, 11), (6000, 13), (7000, 16), (8000, 19), (9000, 22.5), (10000, 25)]
    subaru_EX_21 = [(2000, 4), (3000, 6), (4000, 7)]
    
    return eval(name)


def get_trans_data(name):

    #transmission ratios
    #as reduction ratios so divide  engine rpm = wheel rpm * gear ratio --> engine torque = wheel torque / gear ratio
    #[primary ratio, final ratio, 1st, 2nd, 3rd, 4th...] 
    #or single ratio if straight drive
    ktm_250_SX_F_trans = [2.8, 3.57, 2, 1.62, 1.33, 1.14, .95]
    ktm_duke_200_trans = [3.27, 3.07, 2.83, 2.06, 1.55, 1.23, 1.04, .91]
    honda_cbr_250R_trans = [2.9, 3, 3.33, 2.11, 1.3, 1.57, 1.11, .96]
    yamaha_yz250f_trans = [3, 3.57, 1.9, 1.5, 1.27, 1.09, .95]
    kawasaki_ninja_250R_EXJ_trans = [3.08, 3.21, 2.6, 1.78, 1.4, 1.16, 1, .89] 
    swiss_auto_trans = [8.18]

    return eval(name)


def get_fuel_data():
    '''
    x - rpm (400-1400)
    y - torque (15-30)
    fuel - fuel efficiency [%]
    '''

    y, x = np.mgrid[10:30:4j, 400:1400:9j]

    fuel = np.array([[13,13,13,13,13,13,13,13,13],[16,16,16,15,15,15,15,15,15],\
        [23,23,23,23,23,22.5,22.5,22.5,22.5],[0,30,32,32.5,32.5,32,32,28,0]])
    
    data = np.transpose([x.reshape(-1),y.reshape(-1),fuel.reshape(-1)])

    return data