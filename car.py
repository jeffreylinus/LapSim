import numpy as np
from motor import Motor
from engine import Engine

class Car:
    '''
    Car configurations
    '''

    def __init__(self, **kwargs):
        """
        Init function
        """

        self.name_ICE = kwargs.pop('name_ICE',0)            # ICE name
        self.name_EM = kwargs.pop('name_EM',0)              # EM name
        self.engine = kwargs.pop('engine',0)                # ICE class
        self.motor= kwargs.pop('motor',0)                   # EM class

        self.m = kwargs.pop('m',300)                        # mass of car [kg]
        self.mu = kwargs.pop('mu',0.8)                      # tyre frictional coefficient
        
        self.alim = kwargs.pop('alim',0)                    # traction-limited acceleration (only used in Acc)
        self.wheel_radius = kwargs.pop('wheel_radius', 10)  # wheel radius [inches]
        self.hybrid = kwargs.pop('hybrid',0)                # 1-hybrid; 0-electric

        self.power_split = kwargs.pop('power_split', 0.5)    # fraction of power drawn from EM (max: 0.512 from the rules)
        self.fuel_capacity = kwargs.pop('fuel_capacity', 31.25) # total fuel capacity [MJ]

        self.a = kwargs.pop('frontal_area', 0.85)           # cross sectional area
        self.cd = 0.95                                      # air drag coefficient


    @classmethod
    def init_config(cls, **kwargs):
        '''
        Init from car configuration
        '''
        filepath = kwargs.pop('filepath',0)                      # data file location
        name_EM = kwargs.get('name_EM',0)                        # name of EM
        name_ICE = kwargs.get('name_ICE',0)                      # name of ICE
        
        car = cls(**kwargs)

        capacity_EM = car.fuel_capacity * car.power_split           # fuel capacity of EM 
        capacity_ICE = car.fuel_capacity * (1-car.power_split)      # fuel capacity of ICE

        car.motor = Motor.init_from_file(motor_data=filepath, name_EM=name_EM, capacity=capacity_EM)

        if car.hybrid == 1:
            car.engine = Engine.init_from_file(engine_data=filepath, name_ICE=name_ICE, capacity=capacity_ICE)

        car.calc_mass()
        
        return car

    
    def calc_mass(self):
        '''
        Mass calculation (kg)
        '''
        base_mass = (120+20+24+160)*0.4536                          # chassis, tires, uprights, driver

        # self.engine.m_fuel =  
        # self.motor.m_acc = 

        if self.hybrid == 1:
            self.m = base_mass + self.motor.m + self.engine.m
        elif self.hybrid == 0:
            self.m = base_mass + self.motor.m


    