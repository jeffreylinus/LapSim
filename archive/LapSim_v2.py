import numpy as np
import matplotlib.pyplot as plt

class LapSim:
    '''
    A lap time simulator (point-mass)
    Forward integration until losing traction
    Backward integration from next apex to find brake point

    '''

    def __init__(self, **kwargs):
        """
        Init function
        """

        self.m = kwargs.pop('m',300)                    # mass of car [kg]
        self.mu = kwargs.pop('mu',0.5)                    # tyre frictional coefficient
        self.g = 9.81                                   # gravitational acceleration
        self.steps = kwargs.pop('steps', 50)            # number of discretized points
        self.alim = kwargs.pop('alim',0)                # traction limit

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
        # input track data
        s = np.linspace(0,2*np.pi,res,endpoint=False)
        pts = np.vstack((3*np.cos(s)+0.5*np.sin(3*s),2*np.sin(s)+0.2*np.sin(5*s)))
        
        return cls(pts=pts, **kwargs)


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
        self.v = self.get_velocity()

        # find brake points
        self.brake = self.find_brake_pts()

        self.plot_discretized_points(apex=1, brake=1)            # check apex location

        # calculate lap time
        self.time = np.sum(1/(self.v/self.ds))

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


    def get_velocity(self):
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
        v[self.apex] = np.sqrt(self.mu * self.g * self.r[self.apex])     # velocity at apex
        i = 0
        apex_idx = 0
        state = 'f'

        # get velocity list
        while i<self.steps:
            if state == 'f':                                                        # forward
                if v[i+1]==0:
                    ap = v[i]**2/self.r[i+1]
                    if self.alim>ap:                                                # below traction limit
                        v[i+1] = self.v_integrate(vin=v[i],ap=ap)
                        i+=1
                    else:                                                           # traction is lost
                        state = 'b'
                        apex_idx= np.remainder(apex_idx+1, len(self.apex[0]))
                        print('losing traction, jumping to apex '+str(apex_idx+1))
                        i = self.apex[0][apex_idx]
                elif np.min(v)==0:                                                  # reaching an apex without braking
                    i+=1
                    apex_idx = np.remainder(apex_idx+1, len(self.apex[0]))
                else:
                    print('reached end of track')
                    break
            elif state == 'b':                                                  # backward
                ap = v[i]**2/self.r[i-1]
                if v[i-1]==0:                                                   # if velocity is not yet calculated
                    v[i-1] = self.v_integrate(vin=v[i],ap=ap)
                    i-=1
                else:                                                           # if velocity is calculated from forward integration
                    vback = self.v_integrate(vin=v[i],ap=ap)
                    if vback < v[i-1]:                                          # continue backward integration
                        v[i-1] = vback
                        i-=1
                    else:                                                       # found brake point 
                        print('reached break point, start integrating forward from apex '+str(apex_idx+1))
                        state = 'f'
                        i = self.apex[0][apex_idx]


        return v


    def v_integrate(self, vin=0, ap=0):
        '''
        Integrate for velocity
        '''

        at = np.sqrt(self.alim**2-ap**2)
        v = vin + at*np.abs(1/vin)*self.ds

        return v


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