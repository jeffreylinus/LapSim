import numpy as np
import matplotlib.pyplot as plt
from car import Car

class Acc:
    '''
    A straight track acceleration simulator (point-mass)
    Forward integration until reaching traction/power limit

    Change gears when rpm exceeds range

    TODO:
    - start accelerating from 0
    - test our car
    - report
    
    '''

    def __init__(self, **kwargs):
        """
        Init function
        """

        self.g = 9.81                                       # gravitational acceleration
        self.steps = kwargs.pop('steps', 50)                # number of discretized points
        self.alim = kwargs.pop('alim',0)                    # traction limit

        self.pts = kwargs.pop('pts',0)                  # input track data
        self.pts_interp = kwargs.pop('pts_interp',0)    # interpolated track data
        self.track_len = kwargs.pop('track_len',0)      # total track length

        self.ds = kwargs.pop('ds',0)                    # differential arc length
        self.r = kwargs.pop('r',0)                      # radius of curvature
        self.apex = kwargs.pop('apex',0)                # apex points location
        self.brake = kwargs.pop('brake',0)              # brake points location
        self.v = kwargs.pop('v',0)                      # velocity at each discretized point on track

        self.time = kwargs.pop('time',0)                # total lap time

        self.car_init_args = {
            'name':kwargs.pop('name',0),
            'm':kwargs.pop('m',50),
            'mu':kwargs.pop('mu',0.5),
            'EM':kwargs.pop('EM',0)
        }


    @classmethod
    def init_straight(cls, **kwargs):
        '''
        Init from ellipse
        '''
        res = kwargs.get('steps',10)               # resolution of initial track data

        # input track data
        s = np.linspace(0,500,res,endpoint=True)
        pts = np.vstack((s,np.zeros(res)))

        return cls(pts=pts, **kwargs)


    def acc_time(self):
        '''
        Calculates lap time
        '''
        # init car
        self.car = Car.init_config(**self.car_init_args)

        # interpolate equidistant points on the track
        self.ds = self.pts[0,1] - self.pts[0,0]
        self.track_len = np.max(self.pts)

        # calculate traction-limited velocity at each point
        self.v, self.gear, self.energy = self.get_velocity_list()

        # find brake points
        self.brake = self.find_brake_pts()

        # calculate lap time
        self.time = np.sum(1/(self.v/self.ds))

        self.plot_velocity()

        return 1   


    def get_velocity_list(self):
        '''
        Calculates traction-limited velocity at each point
        m*ap = mv^2/r
        a = sqrt(ap^2+at^2)
        a = mu * N
        v_{i+1} = ap*(dt/ds)*ds + v_i = ap*(1/v_i)*ds + v_i     for increasing roc
        Repeat calculation until losing traction, then jump to the next apex and integrate backwards to find the brake point.
        '''

        self.car.alim = self.g * self.car.mu                            # might want to split lateral/longitudinal traction limit
        v = np.zeros(self.steps)
        energy_list = np.zeros((self.steps,2))
        gear = np.zeros(self.steps)
        
        i = 0
        gear[0] = 1
        v[0] = 8
        energy_list[0] = self.calc_fuel(int(gear[0]), v[0])

        for i in np.arange(self.steps-1):
            v[i+1], gear[i+1], energy_list[i+1]= self.calc_velocity(vin=v[i],gear=int(gear[i]))

        return v, gear, energy_list


    def calc_velocity(self, vin=0, gear=1):
        '''
        Calculates velocity at the next discretized step
        - Integrate for traction-limited velocity 
        - Calculate maximum acceleration allowed at the current power output and integrate for power-limited velocity
        - Compare and return the lower value as the velocity at the next step
        - Check rpm at each step and determine whether to shift gear
        '''

        # calculate rpm and check for shifting conditions
        r = 0.9                                             # set the max rpm
        rpm0 = vin/(self.car.wheel_radius*0.0254*2*np.pi)*60    # rpm of wheels
        rpm_list = rpm0*self.car.gear_ratio[2:]*self.car.gear_ratio[0]*self.car.gear_ratio[1]   # rpm at current gear

        rpm_idx = np.where((self.car.maxrpm*r>rpm_list) & (self.car.minrpm<rpm_list))       # index of possible rpm
        if len(rpm_idx[0]) == 0:
            print('No gear available. Current gear:',gear,', Current rpm:', rpm_list[gear-1])
            energy = self.calc_fuel(gear, vin)
            return vin, gear, energy
        else:
            gear_curr = rpm_idx[0][0]+1                                                    # gear chosen for next step
        
        if gear != gear_curr:
            print('shifting; current gear:', gear_curr)

        Power = self.car.power(rpm_list[rpm_idx[0][0]])                                 

        # calculate velocities and compare   
        v_trac = vin + self.car.alim*np.abs(1/vin)*self.ds                                     # traction-limited velocity

        # Power/rpm -> torque at the engine output (*gear ratio) -> torque at the wheel -> force at the wheel -> acceleration
        omega_rad_s = (rpm_list[rpm_idx[0][0]]/60)*(2*np.pi)                        # angular velocity [rad/s] revolution per minute / 60s * 2pi
        ae = ((Power+self.car.power_EM)*745.7/omega_rad_s)*self.car.gear_ratio[gear_curr+1]/(self.car.wheel_radius*0.0254*self.car.m)
        v_pow = vin + ae*np.abs(1/vin)*self.ds                                      # traction-limited velocity

        v = np.min([v_trac,v_pow])
        if v == v_pow:
            print('power_limited! Power [hp] =', Power)
        else:
            print('traction limited')

        energy = self.calc_fuel(gear_curr, v)
        
        return v, gear_curr, energy


    def calc_fuel(self, gear, v):
        '''
        Calculates the total energy consumed at a discrete step
        ICE efficiency 2D interpolation of the fuel efficiency chart
        EM efficiency is assumed to be a constant
        '''

        rpm = v/(self.car.wheel_radius*0.0254*2*np.pi)*60*self.car.gear_ratio[gear+1]*self.car.gear_ratio[0]*self.car.gear_ratio[1]   # rpm at current gear
        Power = self.car.power(rpm)

        # calculate energy consumed from fuel efficiency
        x = rpm/60*2*np.pi                  # ICE angular velocity [rad/s]
        y = Power*745.7/x                                       # torque [Nm]
        from scipy.interpolate import griddata
        intmethod = 'cubic'
        eta = griddata(self.car.fuel[:,:2], self.car.fuel[:,2], (x,y), method=intmethod)

        P_ICE = Power*100/eta*745.7*(self.ds/v)                 # power consumed by ICE [J]
        P_EM = self.car.power_EM*100/self.car.eta_EM*745.7*(self.ds/v)  # power consumed by EM [J]

        if np.isnan(eta):
            print('WARNING: ICE speed and/or torque are outside of the interpolation range.')

        return [P_ICE, P_EM]


    def find_brake_pts(self):
        '''
        Find brake points from velocity list
        '''

        v_diff = np.sign(self.v - np.roll(self.v, 1, axis=0))
        sign_flip = v_diff - np.roll(v_diff,-1,axis=0)
        brake = np.where(sign_flip == np.max(sign_flip))

        return brake

    
    def plot_velocity(self):

        from mpl_toolkits.axes_grid1 import make_axes_locatable
        import matplotlib.cm as cmx
        import matplotlib.colors

        t = [np.sum(1/(self.v[:i+1]/self.ds)) for i in np.arange(self.steps)]

        v = self.v*2.237                        # convert to [mph]

        fig2 = plt.figure(figsize=(8,6))

        g = self.gear
        cm = plt.get_cmap('viridis')
        cNorm = matplotlib.colors.Normalize(vmin = min(g),vmax = max(g))
        scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cm)
        
        ax = fig2.add_subplot(212)
        ax.scatter(t,self.pts[0],c=scalarMap.to_rgba(g),s=5)
        plt.xlabel('time [s]', fontsize=10)
        plt.ylabel('distance [m]', fontsize=10)
        plt.draw()

        ax2 = fig2.add_subplot(211)
        ax2.scatter(t,v,c=scalarMap.to_rgba(g),s=5)
        plt.xlabel('time [s]', fontsize=10)
        plt.ylabel('speed [mph]', fontsize=10)
        plt.title('Average speed:'+str('{0:.2f}'.format(np.mean(self.v)*2.23))+'mph'+\
            '\nTotal energy consumption:'+str('{0:.2f}'.format(np.sum(self.energy)/1000))+'kJ', fontsize=12)
        plt.draw()

        return 1