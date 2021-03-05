import numpy as np
import engine_and_trans_data as data

class Car:
    '''
    Car configurations
    '''

    def __init__(self, **kwargs):
        """
        Init function
        """

        self.m = kwargs.pop('m',300)                        # mass of car [kg]
        self.mu = kwargs.pop('mu',0.5)                      # tyre frictional coefficient
        
        self.power_EM = kwargs.pop('EM',0)                  # electric motor power
        self.gear_ratio = kwargs.pop('tran',10)             # transmission gear ratio
        self.power = kwargs.pop('power',0)                  # power curve interpolation (rpm, power[hp])
        self.maxrpm = kwargs.pop('maxrpm',0)                # maximum rpm
        self.minrpm = kwargs.pop('minrpm',0)                # minimum rpm
        self.wheel_radius = kwargs.pop('wheel_radius', 10)  # wheel radius [inches]
        self.eta_EM = kwargs.pop('eta_EM',95)
        self.fuel = kwargs.pop('fuel',0)                    # fuel efficiency
        
        self.alim = kwargs.pop('alim',0)                    # accelration limit


    @classmethod
    def init_config(cls, **kwargs):
        '''
        Init from car configuration
        '''
        name = kwargs.pop('name',10)                     # name of car engine
        m = kwargs.get('m',300)                        # mass of car [kg]
        
        power_curve = data.get_power_data(name)
        tran = np.array(data.get_trans_data(name+'_trans'))
        fuel = data.get_fuel_data()

        from scipy.interpolate import interp1d
        rpm = np.array(power_curve).T[0]                # rpm data
        power = np.array(power_curve).T[1]              # power data [hp]
        pint = interp1d(rpm,power, kind='cubic')        # interpolation
        
        return cls(power=pint, maxrpm=np.max(rpm), minrpm=np.min(rpm),\
            tran=tran, fuel=fuel, **kwargs)

    