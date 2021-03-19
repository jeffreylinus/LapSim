import numpy as np
import matplotlib.pyplot as plt
from car import Car

class Acc:
    '''
    A straight track acceleration simulator (point-mass)
    Forward integration until reaching traction/power limit

    Change gears when rpm exceeds range
    
    '''

    def __init__(self, **kwargs):
        """
        Init function
        """

        self.g = 9.81                                       # gravitational acceleration
        self.steps = kwargs.pop('steps', 50)                # number of discretized points
        
        self.pts = kwargs.pop('pts',0)                      # input track data
        self.pts_interp = kwargs.pop('pts_interp',0)        # interpolated track data
        self.track_len = kwargs.pop('track_len',0)          # total track length

        self.ds = kwargs.pop('ds',0)                    # differential arc length
        self.r = kwargs.pop('r',0)                      # radius of curvature
        self.apex = kwargs.pop('apex',0)                # apex points location
        self.brake = kwargs.pop('brake',0)              # brake points location
        self.v = kwargs.pop('v',0)                      # velocity at each discretized point on track

        self.time = kwargs.pop('time',0)                # total lap time

        self.car = kwargs.pop('car',0)                  # Car class

    @classmethod
    def init_straight(cls, **kwargs):
        '''
        Init from ellipse
        '''
        res = kwargs.get('steps',10)               # resolution of initial track data
        track_len = kwargs.get('track_len', 1000)  # length of straight track

        # input track data
        s = np.linspace(0,track_len,res,endpoint=True)
        pts = np.vstack((s,np.zeros(res)))

        return cls(pts=pts, **kwargs)


    def acc_time(self):
        '''
        Calculates lap time
        '''

        # interpolate equidistant points on the track
        self.ds = self.pts[0,1] - self.pts[0,0]
        
        # calculate traction-limited velocity at each point
        self.v, self.gear, self.energy, self.time_list = self.get_velocity_list()

        # find brake points
        self.brake = self.find_brake_pts()

        # calculate lap time
        self.time = np.sum(self.time_list)

        self.plot_velocity()

        return 1   


    def get_velocity_list(self):
        '''
        Calculates traction-limited velocity at each point
        m*ap = mv^2/r
        a = sqrt(ap^2+at^2)
        a = mu * g
        v_{i+1} = ap*(dt/ds)*ds + v_i = ap*(1/v_i)*ds + v_i     for increasing roc
        Repeat calculation until losing traction, then jump to the next apex and integrate backwards to find the brake point.
        '''

        self.car.alim = self.g * self.car.mu                            # might want to split lateral/longitudinal traction limit
        v = np.zeros(self.steps)
        energy_list = np.zeros((self.steps,2))
        gear = np.zeros(self.steps)
        time = np.zeros(self.steps)
        
        i = 0
        gear[0] = 1
        v[0] = 0

        for i in np.arange(self.steps-1):
            v[i+1], gear[i+1], energy_list[i+1], time[i+1]= self.calc_velocity(vin=v[i],gear=int(gear[i]))

        return v, gear, energy_list, time

    def calc_velocity (self,vin=0,gear=1):

        # calculate rpm and check for shifting conditions
        rpm0 = vin/(self.car.wheel_radius*0.0254*2*np.pi)*60    # rpm of wheels at current velocity
        
        if self.car.hybrid == 1:
            a_tor, maxrpm, gear_new, p_EM, p_ICE = self.v_lim_hybrid(vin, gear, rpm0)
        else:
            a_tor, maxrpm, p_EM, p_ICE = self.v_lim_electric(vin, rpm0)
            gear_new = 1

        # torque-limited velocity [m/s]
        v_tor = np.sqrt(2*a_tor*self.ds+vin**2)    # v^2-vi^2 = 2a*ds
        t_tor = (v_tor-vin)/a_tor

        # traction-limited velocity [m/s]                   
        v_trac = np.sqrt(2*self.car.alim*self.ds+vin**2)  
        t_trac = (v_trac-vin)/self.car.alim                                       

        # rpm-limited velocity [m/s]
        v_rpm = maxrpm/60*(self.car.wheel_radius*0.0254*2*np.pi)
        t_rpm = self.ds/v_rpm

        v = np.min([v_trac,v_tor,v_rpm])
        if v == v_tor:
            t = t_tor
            print('Power limited. ICE Power [hp] =', str('{0:.2f}'.format(p_ICE)), 'EM Power [hp] =', str('{0:.2f}'.format(p_EM/745.7)))
        elif v == v_trac:
            t = t_trac
            print('Traction limited. ICE Power [hp] =', str('{0:.2f}'.format(p_ICE)), 'EM Power [hp] =', str('{0:.2f}'.format(p_EM/745.7)))
        elif v == v_rpm:
            t = t_rpm
            print('RPM limited. RPM at EM =', maxrpm*self.car.motor.trans, 'max EM RPM =', self.car.motor.maxrpm)

        e_ICE = self.calc_fuel(gear_new, v, p_ICE, t)
        e_EM = p_EM*t/(self.car.motor.eta/100)
        
        return v, gear_new, [e_ICE, e_EM], t
    

    def v_lim_hybrid(self, vin=0, gear=1, rpm0=0):
        '''
        Calculates velocity at the next discretized step (hybrid vehicles only)
        - Integrate for traction-limited velocity 
        - Calculate maximum acceleration allowed at the current power output and integrate for power-limited velocity
        - Compare and return the lower value as the velocity at the next step
        - Check rpm at each step and determine whether to shift gear
        ICE+EM
        '''
        
        # calculate rpm and check for shifting conditions
        r = 0.95                                             # set the max rpm
        rpm_list = rpm0*self.car.engine.trans[2:]*self.car.engine.trans[0]*self.car.engine.trans[1]   # rpm at all gears

        # calculate Power output
        if (gear == 1 and rpm_list[0]<self.car.engine.minrpm):
            rpm_at_gear_new = rpm_list[0]                                                  
            gear_new = gear
            p_ICE = self.car.engine.power(self.car.engine.minrpm)                                       # use constant extrapolation for v near 0        
        else:
            rpm_idx = np.where((self.car.engine.maxrpm*r>rpm_list) & (self.car.engine.minrpm<rpm_list))       # index of possible rpm
            if len(rpm_idx[0]) == 0:
                print('No higher gear available. Current gear:',gear,', Current rpm:', rpm_list[gear-1])
                rpm_at_gear_new = self.car.engine.maxrpm
                gear_new = gear
            else:
                gear_new = rpm_idx[0][0]+1                                                     # gear chosen for next step
                rpm_at_gear_new = rpm_list[rpm_idx[0][0]]
            p_ICE = self.car.engine.power(rpm_at_gear_new)                                          # ICE power output after shifting                              

        # Power/rpm -> torque at the engine output (*gear ratio) -> torque at the wheel -> force at the wheel -> acceleration
        omega_ICE = (rpm_at_gear_new/60)*(2*np.pi)                                           # angular velocity [rad/s] revolution per minute / 60s * 2pi
        if omega_ICE != 0:
            torque_ICE_at_wheel = (p_ICE*745.7/omega_ICE)*self.car.engine.trans[gear_new+1]  # always use maximum torque during acceleration
        else:
            torque_ICE_at_wheel = 0
            p_ICE = 0

        # torque limited acceleration
        rpm_at_EM = vin/(self.car.wheel_radius*0.0254*2*np.pi)*60*self.car.motor.trans
        omega_EM = (rpm_at_EM/60)*(2*np.pi)
        torque_EM_at_wheel = self.car.motor.torque_max*1.356*self.car.motor.trans
        a_tor = (torque_EM_at_wheel+torque_ICE_at_wheel)/(self.car.wheel_radius*0.0254*self.car.m)
        
        # maxrpm determined by transmission
        wheel_maxrpm_ICE = self.car.engine.maxrpm/(self.car.engine.trans[gear_new+1]*self.car.engine.trans[0]*self.car.engine.trans[1])     
        wheel_maxrpm_EM = self.car.motor.maxrpm/self.car.motor.trans      
        maxrpm = np.min([wheel_maxrpm_EM,wheel_maxrpm_ICE])
        
        p_EM = omega_EM*self.car.motor.torque_max              # Power = torque * omega

        if gear != gear_new:
            print('Shifting...... Current gear:', gear_new)
        
        return a_tor, maxrpm, gear_new, p_EM, p_ICE


    def v_lim_electric(self, vin=0, rpm0=0):
        '''
        Calculates velocity at the next discretized step
        - Integrate for traction-limited velocity 
        - Calculate maximum acceleration allowed at the current power output and integrate for power-limited velocity
        - Compare and return the lower value as the velocity at the next step
        - Check rpm at each step and determine whether to shift gear
        EM only
        '''

        rpm = rpm0*self.car.motor.trans                         # rpm at motor
        omega = (rpm/60)*(2*np.pi)                              # angular velocity [rad/s]                       # angular velocity [rad/s] revolution per minute / 60s * 2pi
        
        # torque-limited velocity [m/s]
        torque_EM_at_wheel = self.car.motor.power_max*1.356*self.car.motor.trans
        a_tor = torque_EM_at_wheel/(self.car.wheel_radius*0.0254*self.car.m)               # torque-limited acceleration
        
        # rpm-limited velocity [m/s]
        maxrpm = self.car.motor.maxrpm/self.car.motor.trans
        
        p_EM = omega*self.car.motor.torque_max                  # power consumed by EM [J]
        p_ICE = 0

        return a_tor, maxrpm, p_EM, p_ICE


    def calc_fuel(self, gear, v, Power, t):
        '''
        ONLY FOR HYBRID VEHICLES
        Calculates the total energy consumed at a discrete step
        ICE efficiency 2D interpolation of the fuel efficiency chart
        EM efficiency is assumed to be a constant
        '''

        if Power == 0:
            return 0

        rpm = v/(self.car.wheel_radius*0.0254*2*np.pi)*60*self.car.engine.trans[gear+1]*self.car.engine.trans[0]*self.car.engine.trans[1]   # rpm at current gear
        
        # calculate energy consumed from fuel efficiency
        x = rpm/60*2*np.pi                  # ICE angular velocity [rad/s]
        if x<self.car.engine.eta[0,0]:
            x = self.car.engine.eta[0,0]                        # for low v, use constant interpolation for fuel efficiency
        y = Power*745.7/x                                       # torque [Nm]
        if y<self.car.engine.eta[0,1]:
            y = self.car.engine.eta[0,1]                        # for low v, use constant interpolation for fuel efficiency
        from scipy.interpolate import griddata
        intmethod = 'cubic'
        eta = griddata(self.car.engine.eta[:,:2], self.car.engine.eta[:,2], (x,y), method=intmethod)

        e_ICE = Power*100/eta*745.7*t                 # energy consumed by ICE [J]

        if np.isnan(eta):
            print('WARNING: ICE speed and/or torque are outside of the interpolation range.')

        return e_ICE


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

        t = [np.sum(self.time_list[:i+1]) for i in np.arange(self.steps)]

        v = self.v*2.237                        # convert to [mph]

        fig2 = plt.figure(figsize=(8,6))
        plt.subplots_adjust(right=0.85)

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
        plt.title('Top speed:'+str('{0:.2f}'.format(np.max(self.v)*2.23))+'mph'+\
            '\nTotal energy consumption:'+str('{0:.2f}'.format(np.sum(self.energy)/1000))+'kJ', fontsize=12)
        cbaxes = fig2.add_axes([0.87, 0.2, 0.02, 0.6])
        cbar = fig2.colorbar(scalarMap,cax=cbaxes)
        cbar.ax.set_ylabel('gear')
        plt.draw()

        return 1