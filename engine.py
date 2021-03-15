import numpy as np

class Engine:
    '''
    ICE specs
    '''

    def __init__(self, **kwargs):
        """
        Init function
        """

        self.name = kwargs.pop('name_ICE',0)                    # ICE name
        self.m = kwargs.pop('m',30)                        # mass of ICE [kg]
        
        self.capacity = kwargs.pop('capacity',0)            # cc
        self.power = kwargs.pop('power',0)          # nominal electric motor power [kW]
        self.trans = kwargs.pop('trans',10)             # transmission gear ratio
        
        self.maxrpm = kwargs.pop('maxrpm',0)                # maximum rpm
        self.minrpm = kwargs.pop('minrpm',0)                # minimum rpm
        
        self.eta = kwargs.pop('eta',0)                     # fuel efficiency
        
        
    @classmethod
    def init_from_file(cls, **kwargs):
        '''
        Init from car configuration
        '''
        engine_data = kwargs.pop('engine_data',0)                         # EM data file location
        sheet_name = kwargs.pop('sheet_name','ICE Stats')                       # EM data sheet name

        engine = cls(**kwargs)

        engine.get_data(engine_data=engine_data, sheet_name=sheet_name)

        return engine


    def get_data(self, engine_data=None, sheet_name='ICE Stats'):

        import pandas as pd
        import ast
        from scipy.interpolate import interp1d

        df = pd.read_excel(engine_data, sheet_name=sheet_name)                 # read track data file
        name = df['Engine Name'].values

        idx = np.where(name==self.name)
        if len(idx[0])==0:
            print('ICE not found!')
            return self
        
        self.capacity = df['cc'].values[idx[0][0]]
        self.m = df['Weight (lbs)'].values[idx[0][0]]*0.4536
        self.eta = self.get_fuel_data()

        trans = df['Transmission Data'].values[idx[0][0]]
        self.trans = np.array(ast.literal_eval(trans))

        power_curve= df['Interpolated Data (rpm, horse power)'].values[idx[0][0]]

        power_curve = np.array(ast.literal_eval(power_curve))

        rpm = np.array(power_curve).T[0]                # rpm data
        self.maxrpm = np.max(rpm)
        self.minrpm = np.min(rpm)
        
        power = np.array(power_curve).T[1]              # power data [hp]
        self.power = interp1d(rpm,power, kind='cubic')        # interpolation

        return 1
    

    @staticmethod
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