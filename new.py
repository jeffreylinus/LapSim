
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
            Power_ICE = self.car.engine.power(self.car.engine.minrpm)                                       # use constant extrapolation for v near 0        
        else:
            rpm_idx = np.where((self.car.engine.maxrpm*r>rpm_list) & (self.car.engine.minrpm<rpm_list))       # index of possible rpm
            if len(rpm_idx[0]) == 0:
                print('No higher gear available. Current gear:',gear,', Current rpm:', rpm_list[gear-1])
                rpm_at_gear_new = self.car.engine.maxrpm
                gear_new = gear
            else:
                gear_new = rpm_idx[0][0]+1                                                     # gear chosen for next step
                rpm_at_gear_new = rpm_list[rpm_idx[0][0]]
            Power_ICE = self.car.engine.power(rpm_at_gear_new)                                          # ICE power output after shifting                              

        # Power/rpm -> torque at the engine output (*gear ratio) -> torque at the wheel -> force at the wheel -> acceleration
        omega_ICE = (rpm_at_gear_new/60)*(2*np.pi)                                           # angular velocity [rad/s] revolution per minute / 60s * 2pi
        if omega_ICE != 0:
            torque_ICE_at_wheel = (Power_ICE*745.7/omega_ICE)*self.car.engine.trans[gear_new+1]  # always use maximum torque during acceleration
        else:
            torque_ICE_at_wheel = 0
            Power_ICE = 0

        # torque limited acceleration
        rpm_at_EM = vin/(self.car.wheel_radius*0.0254*2*np.pi)*60*self.car.motor.trans
        omega_EM = (rpm_at_EM/60)*(2*np.pi)
        torque_EM_at_wheel = self.car.motor.torque_max*1.356*self.car.motor.trans
        a_tor = (torque_EM_at_wheel+torque_ICE_at_wheel)/(self.car.wheel_radius*0.0254*self.car.m)
        
        # maxrpm determined by transmission
        wheel_maxrpm_ICE = self.car.engine.maxrpm/(self.car.engine.trans[gear_new+1]*self.car.engine.trans[0]*self.car.engine.trans[1])     
        wheel_maxrpm_EM = self.car.motor.maxrpm/self.car.motor.trans      
        maxrpm = np.min([wheel_maxrpm_EM,wheel_maxrpm_ICE])
        
        P_ICE = self.calc_fuel(gear_new, v, Power_ICE)
        P_EM = (omega_EM*self.car.motor.torque_max*100/self.car.motor.eta)               # Power = torque * omega
        
        power = [P_ICE, P_EM]

        if gear != gear_new:
            print('Shifting...... Current gear:', gear_new)
        
        return a_tor, maxrpm, gear_new, power


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
        
        e_EM = (omega*self.car.motor.torque_max/self.car.motor.eta)  # power consumed by EM [J]
        power = [0, e_EM]

        return a_tor, maxrpm, power