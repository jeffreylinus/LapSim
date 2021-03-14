import numpy as np
import matplotlib.pyplot as plt
from car import Car

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

    '''

    def __init__(self, **kwargs):
        """
        Init function
        """

        self.g = 9.81                                       # gravitational acceleration
        self.steps = kwargs.pop('steps', 50)                # number of discretized points
        self.alim = kwargs.pop('alim',0)                    # traction limit

        self.hybrid = kwargs.pop('hybrid',1)                # 1-hybrid car  0-electric car

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
            'm':kwargs.pop('m',300),
            'mu':kwargs.pop('mu',0.3),
            'EM':kwargs.pop('EM',0)
        }


    @classmethod
    def init_ellipse(cls, **kwargs):
        '''
        Init from ellipse
        '''
        res = kwargs.pop('resolution',10)               # resolution of initial track data

        # input track data
        s = np.linspace(0,2*np.pi,res,endpoint=False)
        pts = np.vstack((300*np.cos(s),200*np.sin(s)))

        return cls(pts=pts, **kwargs)


    @classmethod
    def init_data(cls, **kwargs):
        '''
        Reat track data from file

        '''
        track_data = kwargs.pop('track_data',0)         # track data file name

        import pandas as pd

        df1 = pd.read_excel(track_data)                 # read track data file
        X = df1['X'].values
        Y = df1['Y'].values
        Z = df1['Z'].values

        pts = np.vstack((X,Y,Z))
        # pts = np.vstack((X,Y))
        
        return cls(pts=pts, **kwargs)


    def lap_time(self):
        '''
        Calculates lap time
        '''
        # init car
        self.car = Car.init_config(**self.car_init_args)

        # interpolate equidistant points on the track
        self.pts_interp, self.ds, self.track_len = self.discretize()

        # calculate radius of curvature
        self.dpds, self.d2pds2, self.r = self.roc()

        # find apex locations
        self.apex = self.find_apex()

        # calculate traction-limited velocity at each point
        self.v, self.gear, self.energy, self.time_list = self.get_velocity_list()

        # find brake points
        self.brake = self.find_brake_pts()

        self.plot_discretized_points(apex=0, brake=0, elevation=0)            # check apex location

        # calculate lap time
        self.time = np.sum(self.time_list)

        self.plot_velocity(apex=0)

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
        snew = np.linspace(0,1,num=self.steps, endpoint=False)
        dim = len(self.pts)
        pts_interp = np.zeros((dim, self.steps))

        from scipy.interpolate import interp1d

        for i in np.arange(dim):
            x = np.append(self.pts[i],self.pts[i,0])
            fx = interp1d(s,x,kind='cubic', fill_value='extrapolate')
            xnew = fx(snew)
            pts_interp[i] = xnew
        
        if dim == 3:

            xdiff = pts_interp[0] - np.roll(pts_interp[0],1,axis=0)
            ydiff = pts_interp[1] - np.roll(pts_interp[1],1,axis=0)
            zdiff = pts_interp[2] - np.roll(pts_interp[2],1,axis=0)
            self.elevation = np.arctan2(zdiff,np.sqrt(xdiff**2+ydiff**2))
        else:
            self.elevation = np.zeros(len(pts_interp[0]))

        ds = track_len/self.steps

        return pts_interp, ds, track_len

    
    def roc(self):
        '''
        Calculates radius of curvature at each point
        r(s) = |(dxds^2+dyds^2)^(3/2)/(dxds*dyds2-dyds*dxds2)|
        '''

        diff = ((self.pts_interp - np.roll(self.pts_interp,1,axis=1))+(np.roll(self.pts_interp,-1,axis=1)-self.pts_interp))/2
        dpds = diff[:2]/self.ds

        diff2 = ((dpds - np.roll(dpds,1,axis=1))+(np.roll(dpds,-1,axis=1)-dpds))/2
        d2pds2 = diff2[:2]/self.ds

        num = np.linalg.norm(dpds,axis=0)**3
        den = np.absolute(np.cross(dpds,d2pds2,axis=0))
        r = num/den

        r[np.isnan(r)] = np.inf
        
        return dpds, d2pds2, r

    
    def find_apex(self):
        '''
        finds cornering apex list: look for sign change in dr
        shift the arrays such that the discretization starts with an apex

        '''

        dr = self.r - np.roll(self.r,1,axis=0)
        sign = np.sign(dr)
        sign_flip = sign - np.roll(sign,-1,axis=0)

        s = np.arange(self.steps)
        apex = np.where((sign_flip == np.min(sign_flip)))
        # apex = np.where(self.r[apex0]<np.median(self.r))

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
        a = mu * g
        v_{i+1} = ap*(dt/ds)*ds + v_i = ap*(1/v_i)*ds + v_i     for increasing roc
        Repeat calculation until losing traction, then jump to the next apex and integrate backwards to find the brake point.
        '''

        self.car.alim = self.g * self.car.mu                            # might want to split lateral/longitudinal traction limit
        v = np.zeros(self.steps)
        energy_list = np.zeros((self.steps,2))
        v[self.apex] = np.sqrt(self.car.mu * self.g * self.r[self.apex])     # velocity at apex
        gear = np.zeros(self.steps)
        time = np.zeros(self.steps)

        i = 0
        apex_idx = 0
        state = 'f'
        gear[0] = 1

        # get velocity list
        while i<self.steps:
            if state == 'f':                                                        # forward
                if v[np.remainder(i+1, self.steps)]==0:
                    ap = v[i]**2/self.r[np.remainder(i+1, self.steps)]#*np.cos(self.elevation[i])-self.car.m*self.g*np.sin(self.elevation[i])
                    if self.car.alim>ap:                                                # below traction limit
                        i1 = np.remainder(i+1, self.steps)                           # step forward
                        # v[i1], gear[i1], energy_list[i1],time[i1]= self.calc_velocity_3D(vin=v[i],ap=ap, gear=int(gear[i]), hybrid = self.hybrid, elevation = self.elevation[i])
                        v[i1], gear[i1], energy_list[i1],time[i1]= self.calc_velocity(vin=v[i],ap=ap, gear=int(gear[i]), hybrid = self.hybrid)
                        i = i1
                    else:                                                           # traction is lost
                        state = 'b'
                        apex_idx= np.remainder(apex_idx+1, len(self.apex[0]))
                        print('losing traction, jumping to apex '+str(apex_idx+1), ' at i=',self.apex[0][apex_idx], ', current i=',i)
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
                ap = v[i]**2/self.r[i-1]#*np.cos(self.elevation[i])-self.car.m*self.g*np.sin(self.elevation[i])
                if v[i-1]==0:                                                   # if velocity is not yet calculated
                    # v[i-1], gear[i-1], energy_list[i-1], time[i-1] = self.calc_velocity_3D(vin=v[i],ap=ap, gear=int(gear[i]), hybrid = self.hybrid, elevation = self.elevation[i])
                    v[i-1], gear[i-1], energy_list[i-1], time[i-1] = self.calc_velocity(vin=v[i],ap=ap, gear=int(gear[i]), hybrid = self.hybrid)
                    i-=1
                else:                                                           # if velocity is calculated from forward integration
                    # vb, gearb, energyb, timeb = self.calc_velocity_3D(vin=v[i],ap=ap, gear=int(gear[i]), hybrid = self.hybrid, elevation = self.elevation[i])
                    vb, gearb, energyb, timeb = self.calc_velocity(vin=v[i],ap=ap, gear=int(gear[i]), hybrid = self.hybrid)
                    if vb < v[i-1]:                                          # continue backward integration
                        v[i-1] = vb
                        energy_list[i-1] = energyb
                        time[i-1] = timeb
                        gear[i-1] = gearb
                        i-=1
                    else:                                                       # found brake point 
                        print('reached break point, start integrating forward from apex '+str(apex_idx+1))
                        state = 'f'
                        i = self.apex[0][apex_idx]


        return v, gear, energy_list, time


    def calc_velocity_3D(self, vin=0, ap=0, gear=1, hybrid=0, elevation=0):
        '''
        Calculates velocity at the next discretized step
        - Integrate for traction-limited velocity 
        - Calculate maximum acceleration allowed at the current power output and integrate for power-limited velocity
        - Compare and return the lower value as the velocity at the next step
        - Check rpm at each step and determine whether to shift gear
        hybrid = 0: electric car (EM only)
        hybrid = 1: hybrid car (ICE + EM)
        '''

        # calculate rpm and check for shifting conditions
        r = 0.95                                                # set the max rpm
        rpm0 = vin/(self.car.wheel_radius*0.0254*2*np.pi)*60    # rpm of wheels
        rpm_list = rpm0*self.car.gear_ratio[2:]*self.car.gear_ratio[0]*self.car.gear_ratio[1]   # rpm at all gears

        # calculate Power output
        if (gear == 1 and rpm_list[0]<self.car.minrpm):
            rpm_at_gear_curr = self.car.minrpm                                                  # use constant extrapolation for v near 0
            gear_curr = gear
        else:
            rpm_idx = np.where((self.car.maxrpm*r>rpm_list) & (self.car.minrpm<rpm_list))       # index of possible rpm
            if len(rpm_idx[0]) == 0:
                print('No higher gear available. Current gear:',gear,', Current rpm:', rpm_list[gear-1])
                rpm_at_gear_curr = self.car.maxrpm
                gear_curr = gear
            else:
                gear_curr = rpm_idx[0][0]+1                                                     # gear chosen for next step
                rpm_at_gear_curr = rpm_list[rpm_idx[0][0]]                                              

        Power = self.car.power(rpm_at_gear_curr)                                          # ICE power output after shifting                                   

        # Power/rpm -> torque at the engine output (*gear ratio) -> torque at the wheel -> force at the wheel -> acceleration
        omega_rad_s = (rpm_at_gear_curr/60)*(2*np.pi)                                           # angular velocity [rad/s] revolution per minute / 60s * 2pi
        P_effective = (Power+self.car.power_EM)*745.7-self.car.m*self.g*vin*np.sin(elevation)       # effective P for acceleration parallel to the track
        ae = (P_effective/omega_rad_s)*self.car.gear_ratio[gear_curr+1]/(self.car.wheel_radius*0.0254*self.car.m)
        
        # power-limited velocity [m/s]                       
        v_pow = np.sqrt(2*ae*self.ds+vin**2)
        t_pow = (v_pow-vin)/ae

        # traction-limited velocity [m/s]                   
        at = np.sqrt(self.car.alim**2-ap**2)                                            # (traction limited) tangential acceleration
        v_trac = np.sqrt(2*at*self.ds+vin**2)  
        t_trac = (v_trac-vin)/self.car.alim
        
        # rpm-limited velocity [m/s]
        wheel_maxrpm = self.car.maxrpm/(self.car.gear_ratio[gear_curr+1]*self.car.gear_ratio[0]*self.car.gear_ratio[1])      # rpm at wheels
        v_rpm = wheel_maxrpm/60*(self.car.wheel_radius*0.0254*2*np.pi)
        t_rpm = self.ds/v_rpm

        v = np.min([v_trac,v_pow,v_rpm])
        if v == v_pow:
            t = t_pow
            print('Power limited. Power [hp] =', Power)
        elif v == v_trac:
            t = t_trac
            print('Traction limited. Power [hp] =', Power)
        elif v == v_rpm:
            t = t_rpm
            print('RPM limited. Power [hp] =', Power, '. Current gear:', gear_curr)

        if hybrid == 1:
            energy = self.calc_fuel(gear_curr, v, Power, t)
        else:
            energy = [0, Power*100/self.car.eta_EM*745.7*t]

        if gear != gear_curr:
            print('Shifting...... Current gear:', gear_curr)

        return v, gear_curr, energy, t


    def calc_velocity(self, vin=0, ap=0, gear=1, hybrid=0):
        '''
        Calculates velocity at the next discretized step
        - Integrate for traction-limited velocity 
        - Calculate maximum acceleration allowed at the current power output and integrate for power-limited velocity
        - Compare and return the lower value as the velocity at the next step
        - Check rpm at each step and determine whether to shift gear
        hybrid = 0: electric car (EM only)
        hybrid = 1: hybrid car (ICE + EM)
        '''

        # calculate rpm and check for shifting conditions
        r = 0.95                                                # set the max rpm
        rpm0 = vin/(self.car.wheel_radius*0.0254*2*np.pi)*60    # rpm of wheels
        rpm_list = rpm0*self.car.gear_ratio[2:]*self.car.gear_ratio[0]*self.car.gear_ratio[1]   # rpm at all gears

        # calculate Power output
        if (gear == 1 and rpm_list[0]<self.car.minrpm):
            rpm_at_gear_curr = self.car.minrpm                                                  # use constant extrapolation for v near 0
            gear_curr = gear
        else:
            rpm_idx = np.where((self.car.maxrpm*r>rpm_list) & (self.car.minrpm<rpm_list))       # index of possible rpm
            if len(rpm_idx[0]) == 0:
                print('No higher gear available. Current gear:',gear,', Current rpm:', rpm_list[gear-1])
                rpm_at_gear_curr = self.car.maxrpm
                gear_curr = gear
            else:
                gear_curr = rpm_idx[0][0]+1                                                     # gear chosen for next step
                rpm_at_gear_curr = rpm_list[rpm_idx[0][0]]                                              

        Power = self.car.power(rpm_at_gear_curr)                                          # ICE power output after shifting                                   

        # Power/rpm -> torque at the engine output (*gear ratio) -> torque at the wheel -> force at the wheel -> acceleration
        omega_rad_s = (rpm_at_gear_curr/60)*(2*np.pi)                                           # angular velocity [rad/s] revolution per minute / 60s * 2pi
        ae = ((Power+self.car.power_EM)*745.7/omega_rad_s)*self.car.gear_ratio[gear_curr+1]/(self.car.wheel_radius*0.0254*self.car.m)
        
        # power-limited velocity [m/s]                       
        v_pow = np.sqrt(2*ae*self.ds+vin**2)
        t_pow = (v_pow-vin)/ae

        # traction-limited velocity [m/s]                   
        at = np.sqrt(self.car.alim**2-ap**2)                                            # (traction limited) tangential acceleration
        v_trac = np.sqrt(2*at*self.ds+vin**2)  
        t_trac = (v_trac-vin)/self.car.alim
        
        # rpm-limited velocity [m/s]
        wheel_maxrpm = self.car.maxrpm/(self.car.gear_ratio[gear_curr+1]*self.car.gear_ratio[0]*self.car.gear_ratio[1])      # rpm at wheels
        v_rpm = wheel_maxrpm/60*(self.car.wheel_radius*0.0254*2*np.pi)
        t_rpm = self.ds/v_rpm

        v = np.min([v_trac,v_pow,v_rpm])
        if v == v_pow:
            t = t_pow
            print('Power limited. Power [hp] =', Power)
        elif v == v_trac:
            t = t_trac
            print('Traction limited. Power [hp] =', Power)
        elif v == v_rpm:
            t = t_rpm
            print('RPM limited. Power [hp] =', Power, '. Current gear:', gear_curr)

        if hybrid == 1:
            energy = self.calc_fuel(gear_curr, v, Power, t)
        else:
            energy = [0, Power*100/self.car.eta_EM*745.7*t]

        if gear != gear_curr:
            print('Shifting...... Current gear:', gear_curr)

        return v, gear_curr, energy, t


    def calc_fuel(self, gear, v, Power, t):
        '''
        Calculates the total energy consumed at a discrete step
        ICE efficiency 2D interpolation of the fuel efficiency chart
        EM efficiency is assumed to be a constant
        '''

        rpm = v/(self.car.wheel_radius*0.0254*2*np.pi)*60*self.car.gear_ratio[gear+1]*self.car.gear_ratio[0]*self.car.gear_ratio[1]   # rpm at current gear
        
        # calculate energy consumed from fuel efficiency
        x = rpm/60*2*np.pi                  # ICE angular velocity [rad/s]
        if x<self.car.fuel[0,0]:
            x = self.car.fuel[0,0]                              # for low v, use constant interpolation for fuel efficiency
        y = Power*745.7/x                                       # torque [Nm]
        from scipy.interpolate import griddata
        intmethod = 'cubic'
        eta = griddata(self.car.fuel[:,:2], self.car.fuel[:,2], (x,y), method=intmethod)

        P_ICE = Power*100/eta*745.7*t                 # power consumed by ICE [J]
        P_EM = self.car.power_EM*100/self.car.eta_EM*745.7*t  # power consumed by EM [J]

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


    def plot_discretized_points(self, apex=0, brake=0, elevation=0):
        
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        import matplotlib.cm as cmx
        import matplotlib.colors

        fig = plt.figure(figsize=(8,6))
        ax = fig.add_subplot(111)
        ax.set_aspect('equal')
        if elevation == 1:
            plt.subplots_adjust(right=0.85)
            cm = plt.get_cmap('plasma')
            cNorm = matplotlib.colors.Normalize(vmin = min(self.pts_interp[2]),vmax = max(self.pts_interp[2]))
            scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cm)
            ax.scatter(self.pts_interp[0], self.pts_interp[1],c=scalarMap.to_rgba(self.pts_interp[2]),s=10, label='interpolation')
        else:
            plt.scatter(self.pts_interp[0], self.pts_interp[1],s=10,label='Interpolation')
        plt.scatter(self.pts[0],self.pts[1],s=2,label='Input')
        if apex==1:
            plt.scatter(self.pts_interp[0,self.apex],self.pts_interp[1,self.apex],c='g',marker='^',label='apex')
        if brake==1:
            plt.scatter(self.pts_interp[0,self.brake],self.pts_interp[1,self.brake],c='r',marker='x',label='brake')
        plt.title('Discretized Track points (equidistant track length sampling)')
        plt.legend()
        if elevation == 1:
            cbaxes = fig.add_axes([0.87, 0.2, 0.02, 0.6])
            cbar = fig.colorbar(scalarMap,cax=cbaxes)
            cbar.ax.set_ylabel('Elevation [m]')
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