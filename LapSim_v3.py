import numpy as np
import matplotlib.pyplot as plt

class LapSim:
    '''
    A lap time simulator (point-mass)
    Forward integration until losing traction
    Backward integration from next apex to find brake point

    energy consumption (endurance)
        split between gas and battery
    powertrain
        maximum acceleration/power output

    Change gears when rpm exceeds range
    
    UPDATES:
    - checked longer track; reached power limit
    - added fuel efficiency check; interpolation falls out of range of data so it's a scam rn


    TODO:
    make car class
    make a straight track
        - maybe can get tyre coeff results

    '''

    def __init__(self, **kwargs):
        """
        Init function
        """

        self.m = kwargs.pop('m',0)                          # mass of car [kg]
        self.mu = kwargs.pop('mu',0.5)                      # tyre frictional coefficient
        self.g = 9.81                                       # gravitational acceleration
        self.steps = kwargs.pop('steps', 50)                # number of discretized points
        self.alim = kwargs.pop('alim',0)                    # traction limit

        self.gear_ratio = kwargs.pop('tran',10)             # transmission gear ratio
        self.power = kwargs.pop('power',0)                  # power curve interpolation (rpm, power[hp])
        self.maxrpm = kwargs.pop('maxrpm',0)                # maximum rpm
        self.minrpm = kwargs.pop('minrpm',0)                # minimum rpm
        self.wheel_radius = kwargs.pop('wheel_radius', 14)  # wheel radius [inches]
        self.power_EM = kwargs.pop('EM',0)                  # electric motor (if any) power
        self.eta_EM = kwargs.pop('eta_EM',95)               # electric motor fuel efficiency (assumed constant)
        self.fuel = kwargs.pop('fuel',0)                    # ICE fuel efficiency chart

        self.pts = kwargs.pop('pts',0)                  # input track data
        self.pts_interp = kwargs.pop('pts_interp',0)    # interpolated track data
        self.track_len = kwargs.pop('track_len',0)      # total track length

        self.ds = kwargs.pop('ds',0)                    # differential arc length
        self.r = kwargs.pop('r',0)                      # radius of curvature
        self.apex = kwargs.pop('apex',0)                # apex points location
        self.brake = kwargs.pop('brake',0)              # brake points location
        self.v = kwargs.pop('v',0)                      # velocity at each discretized point on track

        self.time = kwargs.pop('time',0)                # total lap time


    @classmethod
    def init_ellipse(cls, **kwargs):
        '''
        Init from ellipse
        '''
        res = kwargs.pop('resolution',10)               # resolution of initial track data
        power_curve = kwargs.pop('power',0)             # (rpm, power[hp])
        tran = np.array(kwargs.pop('tran',0))           # transmission gear ratio
        
        # input track data
        s = np.linspace(0,2*np.pi,res,endpoint=False)
        pts = np.vstack((500*np.cos(s),200*np.sin(s)))

        from scipy.interpolate import interp1d
        rpm = np.array(power_curve).T[0]                # rpm data
        power = np.array(power_curve).T[1]              # power data [hp]
        pint = interp1d(rpm,power, kind='cubic')        # interpolation
        
        return cls(pts=pts, power=pint, maxrpm=np.max(rpm), minrpm=np.min(rpm), tran=tran, **kwargs)


    @classmethod
    def init_data(cls, **kwargs):
        '''
        2 straights + 2 circular corners
        Input arguments:
            l = length of straight section [m]
            r = radius of circular section [m]
        
        This is intended to be the general track init function with input data files.

        *******WIP*******

        '''
        res = kwargs.pop('resolution',10)               # resolution of initial track data
        power_curve = kwargs.pop('power',0)             # (rpm, power[hp])
        tran = np.array(kwargs.pop('tran',0))           # transmission gear ratio
        l = kwargs.pop('l',200)
        r = kwargs.pop('l',50)
        
        # input track data
        straight = np.linspace(-l,l,res/2,endpoint=False)
        # s = np.linspace(np.pi/2, ,res,endpoint=False)
        pts = np.vstack((400*np.cos(s),200*np.sin(s)))

        from scipy.interpolate import interp1d
        rpm = np.array(power_curve).T[0]                # rpm data
        power = np.array(power_curve).T[1]              # power data [hp]
        pint = interp1d(rpm,power, kind='cubic')        # interpolation
        
        return cls(pts=pts, power=pint, maxrpm=np.max(rpm), minrpm=np.min(rpm), tran=tran, **kwargs)


    def lap_time(self):
        '''
        Calculates lap time
        '''
        
        # interpolate equidistant points on the track
        self.pts_interp, self.ds, self.track_len = self.discretize()

        # calculate radius of curvature
        self.dpds, self.d2pds2, self.r = self.roc()

        # find apex locations
        self.apex = self.find_apex()

        # calculate traction-limited velocity at each point
        self.v, self.energy = self.get_velocity_list()

        # find brake points
        self.brake = self.find_brake_pts()

        self.plot_discretized_points(apex=1, brake=1)            # check apex location

        # calculate lap time
        self.time = np.sum(1/(self.v/self.ds))

        self.plot_velocity(apex=1)

        return 1


    def discretize(self):
        
        # Parametrize track by variable s. Assume that the input points are ordered and arbitrarily spaced.
        # Parametrization with respect to normalized arc length
        diff = self.pts - np.roll(self.pts,1,axis=1)
        arclen = np.linalg.norm(diff,axis=0)   # length of displacement vector
        track_len = np.sum(arclen)
        s = np.cumsum(arclen)/track_len

        # periodic boundary condition (x(0) == x(1), y(0) == y(1))
        s = np.append(0,s)
        x = np.append(self.pts[0],self.pts[0,0])
        y = np.append(self.pts[1],self.pts[1,0])

        # Discretize track by equidistant points in s
        from scipy.interpolate import interp1d
        snew = np.linspace(0,1,num=self.steps, endpoint=False)
        fx = interp1d(s,x, kind='cubic')
        fy = interp1d(s,y, kind='cubic')

        xnew = fx(snew)
        ynew = fy(snew)
        pts_interp = np.vstack((xnew,ynew))

        ds = track_len/self.steps

        return pts_interp, ds, track_len

    
    def roc(self):
        '''
        Calculates radius of curvature at each point
        r(s) = |(dxds^2+dyds^2)^(3/2)/(dxds*dyds2-dyds*dxds2)|
        '''

        diff = ((self.pts_interp - np.roll(self.pts_interp,1,axis=1))+(np.roll(self.pts_interp,-1,axis=1)-self.pts_interp))/2
        dpds = diff/self.ds

        diff2 = ((dpds - np.roll(dpds,1,axis=1))+(np.roll(dpds,-1,axis=1)-dpds))/2
        d2pds2 = diff2/self.ds

        num = np.linalg.norm(dpds,axis=0)**3
        den = np.absolute(np.cross(dpds,d2pds2,axis=0))
        r = num/den
        
        return dpds, d2pds2, r

    
    def find_apex(self):
        '''
        finds cornering apex list: look for sign change in dr
        shift the arrays such that the discretization starts with an apex

        '''

        dr = self.r - np.roll(self.r,1,axis=0)
        sign = np.sign(dr)
        sign_flip = sign - np.roll(sign,-1,axis=0)

        apex = np.where(sign_flip == np.min(sign_flip))
        # apex_min = np.argmin(self.r)
        idx_0 = self.r.shape[0]-apex[0][0]
        idx = np.arange(self.r.shape[0]) - idx_0

        # re-indexing to make apex the first point
        apex = apex-apex[0][0]
        self.r = self.r[idx]
        self.pts_interp = self.pts_interp[:,idx]

        return apex


    def get_velocity_list(self):
        '''
        Calculates traction-limited velocity at each point
        m*ap = mv^2/r
        a = sqrt(ap^2+at^2)
        a = mu * N
        v_{i+1} = ap*(dt/ds)*ds + v_i = ap*(1/v_i)*ds + v_i     for increasing roc
        Repeat calculation until losing traction, then jump to the next apex and integrate backwards to find the brake point.
        '''

        self.alim = self.g * self.mu                            # might want to split lateral/longitudinal traction limit
        v = np.zeros(self.steps)
        energy_list = np.zeros((self.steps,2))
        v[self.apex] = np.sqrt(self.mu * self.g * self.r[self.apex])     # velocity at apex
        

        i = 0
        apex_idx = 0
        state = 'f'
        gear = 1
        energy_list[0] = self.calc_fuel(gear, v[0])

        # get velocity list
        while i<self.steps:
            if state == 'f':                                                        # forward
                if v[np.remainder(i+1, self.steps)]==0:
                    ap = v[i]**2/self.r[np.remainder(i+1, self.steps)]
                    if self.alim>ap:                                                # below traction limit
                        v[np.remainder(i+1, self.steps)], gear, energy_list[np.remainder(i+1, self.steps)]= self.calc_velocity(vin=v[i],ap=ap, gear=gear)
                        i = np.remainder(i+1, self.steps)
                    else:                                                           # traction is lost
                        state = 'b'
                        apex_idx= np.remainder(apex_idx+1, len(self.apex[0]))
                        print('losing traction, jumping to apex '+str(apex_idx+1))
                        i = self.apex[0][apex_idx]
                else:                                                               # check if velocity at next apex can be achieved with the current gear
                    # self.check_velocity_at_apex(vin=v[i], vnext=v[np.remainder(i+1, self.steps)],ap=ap)
                    if np.min(v)==0:                                                  # reaching an apex without braking
                        i = np.remainder(i+1, self.steps)
                        apex_idx = np.remainder(apex_idx+1, len(self.apex[0]))
                    else:
                        print('reached end of track')
                        break
            elif state == 'b':                                                  # backward
                ap = v[i]**2/self.r[i-1]
                if v[i-1]==0:                                                   # if velocity is not yet calculated
                    v[i-1], gear, energy_list[i-1] = self.calc_velocity(vin=v[i],ap=ap, gear=gear)
                    i-=1
                else:                                                           # if velocity is calculated from forward integration
                    vback, gear, energy = self.calc_velocity(vin=v[i],ap=ap, gear=gear)
                    if vback < v[i-1]:                                          # continue backward integration
                        v[i-1] = vback
                        energy_list[i-1] = energy
                        i-=1
                    else:                                                       # found brake point 
                        print('reached break point, start integrating forward from apex '+str(apex_idx+1))
                        state = 'f'
                        i = self.apex[0][apex_idx]


        return v, energy_list


    def calc_velocity(self, vin=0, ap=0, gear=1):
        '''
        Calculates velocity at the next discretized step
        - Integrate for traction-limited velocity 
        - Calculate maximum acceleration allowed at the current power output and integrate for power-limited velocity
        - Compare and return the lower value as the velocity at the next step
        - Check rpm at each step and determine whether to shift gear
        '''

        # calculate rpm and check for shifting conditions
        r = 0.9                                             # set the max rpm
        rpm0 = vin/(self.wheel_radius*0.0254*2*np.pi)*60    # rpm of wheels
        rpm_list = rpm0*self.gear_ratio[2:]*self.gear_ratio[0]*self.gear_ratio[1]   # rpm at current gear

        rpm_idx = np.where((self.maxrpm*r>rpm_list) & (self.minrpm<rpm_list))       # index of possible rpm
        if len(rpm_idx[0]) == 0:
            print('No gear available. Current gear:',gear,', Current rpm:', rpm_list[gear-1])
        else:
            gear_curr = rpm_idx[0][0]+1                                                    # gear chosen for next step
        
        if gear != gear_curr:
            print('shifting; current gear:', gear_curr)

        Power = self.power(rpm_list[rpm_idx[0][0]])                                 

        # calculate velocities and compare   
        at = np.sqrt(self.alim**2-ap**2)                                            # (traction limited) tangential acceleration
        v_trac = vin + at*np.abs(1/vin)*self.ds                                     # traction-limited velocity

        # Power/rpm -> torque at the engine output (*gear ratio) -> torque at the wheel -> force at the wheel -> acceleration
        omega_rad_s = (rpm_list[rpm_idx[0][0]]/60)*(2*np.pi)                        # angular velocity [rad/s] revolution per minute / 60s * 2pi
        ae = (Power*745.7/omega_rad_s)*gear_curr/(self.wheel_radius*0.0254)/self.m
        v_pow = vin + ae*np.abs(1/vin)*self.ds                                      # traction-limited velocity

        v = np.min([v_trac,v_pow])
        if v == v_pow:
            print('power_limited!')

        energy = self.calc_fuel(gear_curr, v)
        
        return v, gear_curr, energy


    def calc_fuel(self, gear, v):
        '''
        Calculates the total energy consumed at a discrete step
        ICE efficiency 2D interpolation of the fuel efficiency chart
        EM efficiency is assumed to be a constant
        '''

        rpm = v/(self.wheel_radius*0.0254*2*np.pi)*60*self.gear_ratio[gear+1]*self.gear_ratio[0]*self.gear_ratio[1]   # rpm at current gear
        Power = self.power(rpm)

        # calculate energy consumed from fuel efficiency
        x = rpm/60*2*np.pi                  # ICE angular velocity [rad/s]
        y = Power*745.7/x                                       # torque [Nm]
        from scipy.interpolate import griddata
        intmethod = 'cubic'
        eta = griddata(self.fuel[:,:2], self.fuel[:,2], (x,y), method=intmethod)

        P_ICE = Power*100/eta*745.7*(self.ds/v)                 # power consumed by ICE [J]
        P_EM = self.power_EM*100/self.eta_EM*745.7*(self.ds/v)  # power consumed by EM [J]

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


    def plot_discretized_points(self, apex=0, brake=0):
        
        fig = plt.figure(figsize=(8,6))
        ax1 = fig.add_subplot(111)
        ax1.set_aspect('equal')
        plt.scatter(self.pts_interp[0], self.pts_interp[1],s=10,label='Interpolation')
        plt.scatter(self.pts[0],self.pts[1],s=2,label='Input')
        if apex==1:
            plt.scatter(self.pts_interp[0,self.apex],self.pts_interp[1,self.apex],c='g',marker='^',label='apex')
        if brake==1:
            plt.scatter(self.pts_interp[0,self.brake],self.pts_interp[1,self.brake],c='r',marker='x',label='brake')
        plt.title('Discretized Track points (equidistant in s)')
        plt.legend()
        plt.draw()
        
        return 1

    
    def plot_velocity(self, apex=0):

        from mpl_toolkits.axes_grid1 import make_axes_locatable
        import matplotlib.cm as cmx
        import matplotlib.colors

        v = self.v*2.237                        # convert to [mph]
        fig2 = plt.figure(figsize=(8,6))
        ax = fig2.add_subplot(111)
        plt.subplots_adjust(right=0.85)
        ax.set_aspect('equal')
        cm = plt.get_cmap('viridis')
        cNorm = matplotlib.colors.Normalize(vmin = min(v),vmax = max(v))
        scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cm)
        ax.scatter(self.pts_interp[0], self.pts_interp[1],c=scalarMap.to_rgba(v),s=10)
        # ax.scatter(self.pts[0],self.pts[1],c='k',s=2,label='Input')
        if apex==1:
            plt.scatter(self.pts_interp[0,self.apex],self.pts_interp[1,self.apex],c='r',marker='x',label='apex')
        plt.xlabel('X [m]', fontsize=10)
        plt.ylabel('Y [m]', fontsize=10)
        plt.legend(fontsize=10)
        plt.title('Average speed:'+str('{0:.2f}'.format(np.mean(self.v)*2.23))+'mph'+\
            '\nTotal energy consumption:'+str('{0:.2f}'.format(np.sum(self.energy)/1000))+'kJ', fontsize=12)
        cbaxes = fig2.add_axes([0.87, 0.2, 0.02, 0.6])
        cbar = fig2.colorbar(scalarMap,cax=cbaxes)
        cbar.ax.set_ylabel('velocity [mph]')
        plt.draw()

        return 1


    def plot_derivatives(self):
        '''
        check derivative vectors for curvature calculation
        '''

        fig = plt.figure(figsize=(8,6))
        ax1 = fig.add_subplot(111)
        ax1.set_aspect('equal')
        plt.scatter(self.pts_interp[0],self.pts_interp[1],label='Discretized points')
        plt.quiver(self.pts_interp[0],self.pts_interp[1],self.dpds[0],self.dpds[1],linewidth=0.5,label='dpds')
        plt.quiver(self.pts_interp[0],self.pts_interp[1],self.d2pds2[0],self.d2pds2[1],label='d2pds2')
        plt.title('Curvature')
        plt.legend()
        plt.draw()

        return 1