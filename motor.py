import numpy as np

class Motor:
    '''
    EM specs
    '''

    def __init__(self, **kwargs):
        """
        Init function
        """

        self.name = kwargs.pop('name_EM',0)                    # EM name
        self.m = kwargs.pop('m',30)                         # mass of EM [kg]
        self.m_acc = kwargs.pop('m_acc',0)                  # mass of accumulator [kg]
        self.m_MC = kwargs.pop('m_MC',0)                    # motor controller mass [kg]
        
        self.power_nom = kwargs.pop('power_nom',0)          # nominal electric motor power [kW]
        self.power_max = kwargs.pop('power_max',0)          # max power [kW]
        self.trans = kwargs.pop('trans',10)             # transmission gear ratio
        
        self.maxrpm = kwargs.pop('maxrpm',0)                # maximum rpm
        
        self.voltage = kwargs.pop('voltage',0)               # voltage rating
        self.current_nom = kwargs.pop('current_nom',0)        # nominal current
        self.torque_con = kwargs.pop('torque_con',0)          # continuous torque
        self.torque_max = kwargs.pop('torque_max',0)          # max torque [Nm]

        self.eta = kwargs.pop('eta',95)                     # fuel efficiency

        self.capacity = kwargs.pop('capacity',0)            # accumulator capacity [MJ] 
        
        
    @classmethod
    def init_from_file(cls, **kwargs):
        '''
        Init from car configuration
        '''
        motor_data = kwargs.pop('motor_data',0)                         # EM data file location
        sheet_name = kwargs.pop('sheet_name','EM Stats')                       # EM data sheet name

        motor = cls(**kwargs)

        motor.m_MC = 30*0.4536                                          # motor controller mass [kg]

        motor.get_data(motor_data=motor_data, sheet_name=sheet_name)

        return motor


    def get_data(self, motor_data=None, sheet_name='EM Stats'):

        import pandas as pd

        df = pd.read_excel(motor_data, sheet_name=sheet_name)                 # read track data file
        name = df['Engine Name'].values

        idx = np.where(name==self.name)
        if len(idx[0])==0:
            print('EM not found!')
            return self
        
        self.eta = df['Efficiency (Peak)'].values[idx[0][0]]
        self.voltage = df['Voltage'].values[idx[0][0]]
        self.current_nom = df['Continuous Current'].values[idx[0][0]]
        self.torque_con = df['Torque (Nm)'].values[idx[0][0]]
        self.torque_max = df['Peak Torque (Nm)'].values[idx[0][0]]
        self.maxrpm = df['MAX RPM'].values[idx[0][0]]
        self.power_nom = df['Power (kW)'].values[idx[0][0]]
        self.power_max = df['Peak Power'].values[idx[0][0]]

        battery_no = self.voltage / 2.5                                         # number of batteries
        self.m_acc = battery_no*2*0.4536                                        # mass of accumulator [kg]

        self.m = df['Weight (lbs)'].values[idx[0][0]]*0.4536 + self.m_acc + self.m_MC
        self.trans = 4

        return 1
    