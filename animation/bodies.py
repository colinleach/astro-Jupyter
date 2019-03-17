import sys
import numpy as np

from astropy import units as u
from astropy.constants import G
from astropy.table import QTable

from scipy.integrate import solve_ivp

G = 6.7e-11 # Universal gravitational constant, SI

class Bodies:
    def __init__(self):
        self.posns = np.zeros((0,3))
        self.vs = np.zeros((0,3))
        self.ms = np.zeros((0))
        self.rs = np.zeros((0))
        self.sun = None
        self.planets = []
        self.nBodies = 0
        self.time = 0
        
#----------------------------------------   
# Some general utility methods

    def is_iterable(self, x):
        # a surprising omission from standard Python?
        try:
            iterator = iter(x)
        except TypeError:
            return False
        else:
            return True       
        
    def fix_units(self, val, unit):
        "Convert to SI if necessary and return value as a Python scalar"
        if isinstance(val, u.quantity.Quantity):
            val = val.to(unit).value
        return val

    def veclen(self, vector):
        # beware, units are lost and this just returns a number
        return np.linalg.norm(vector) 

    def vecperp(self, vector):
        "rotate 90 deg ccw, return normalised vector in x,y plane"
        v = np.array([-vector[1], vector[0], 0])
        return v/self.veclen(v)
    
    def get_time(self):
        "Cumulative integration time (s)"
        return self.time*u.s
    
    def CoM_velocity(self):
        "Reruns velocity of center of mass"
        return np.sum(self.vs * self.ms[:, np.newaxis], axis=0) / np.sum(self.ms)
    
    def fix_CoM(self):
        "Ensure CoM velocity is zero"
        self.vs -= self.CoM_velocity()

#-----------------------------------------------------
# Several methods to add bodies to the collection
    
    def add_sun(self, M, R=None):
        """
        For 1-body problems, a large mass fixed at the origin
           M = mass (kg or Quantity)
           R = radius (m or Quantity); only for collision detection
        """

        M = self.fix_units(M, u.kg)
        R = self.fix_units(R, u.m)

        self.sun = self.ms.size # index to this new body
        self.posns = np.concatenate((self.posns, np.zeros((1,3))))
        self.vs = np.concatenate((self.vs, np.zeros((1,3))))
        self.ms = np.concatenate((self.ms, [M,]))
        self.rs = np.concatenate((self.rs, [R,]))
        self.nBodies = self.ms.size
        
    def add_bodies(self, pos, v, m, R):
        """ 
        Can be one body or many
        single: need pos and v to be 3-item iterables
        many: pos and p have shape (N,3), m and R are 1-D array-like
        """
        
        if not self.is_iterable(m): # just have a single body
            # make sure the 3-vectors are numpy arrays
            # (this does nothing if they are already arrays)
            pos = np.array(pos)
            v = np.array(v)
            
            # get everything to a suitable shape for concatenation
            pos = pos[np.newaxis,:] # converts shape (3,) to (0,3)
            v = v[np.newaxis,:]
            m = [m,]
            R = [R,]
        
        self.posns = np.concatenate((self.posns, pos))
        self.vs = np.concatenate((self.vs, v))
        self.ms = np.concatenate((self.ms, [m,]))
        self.rs = np.concatenate((self.rs, [R,]))
        self.nBodies = self.ms.size
        
    def add_planet_at_pericenter(self, a, e, i=0, phi=0, m=None, R=None):
        """
        For setting up a 1-body Keplerian orbit.
           a = semimajor axis (m or Quantity)
           e = eccentricity
           i = inclination (deg)
           phi = orientation of perihelion (deg ccw from x-axis)
           m = mass (kg or Quantity); only req if an N-body calc will be run
           R = radius (m or Quantity); only for collision detection
        """

        if self.sun is None:
            display("Error: Please run add_sun() first")
            return
        else:
            M_sun = self.ms[self.sun]
            
        a = self.fix_units(a, u.m)
        m = self.fix_units(m, u.kg)
        R = self.fix_units(R, u.m)

        P = np.sqrt(4 * np.pi**2 * a**3/(G * M_sun))
        planet = {}
        planet['P'] = P
        planet['a'] = a
        planet['e'] = e
        planet['i'] = i
        self.planets.append(planet)

        # set starting position and velocity, at perihelion
        r_dir = np.array([np.cos(phi), np.sin(phi), np.sin(i)]) 
        rhat = r_dir/self.veclen(r_dir)
        r_vec = a*(1-e) * rhat

        vhat = self.vecperp(rhat) 
        v_vec = np.sqrt(G*M_sun/a*(1+e)/(1-e)) * vhat
        
        self.posns = np.concatenate((self.posns, r_vec[np.newaxis,:]))
        self.vs = np.concatenate((self.vs, v_vec[np.newaxis,:]))
        self.ms = np.concatenate((self.ms, [m,]))
        self.rs = np.concatenate((self.rs, [R,]))
        self.nBodies = self.ms.size

    def add_binary(self, masses, a, e, Rs=None):
        """
        For setting up a 1-body Keplerian orbit.
           masses = 2-tuple (kg or Quantity)
           a = semimajor axis (m or Quantity)
           e = eccentricity
           m = mass (kg or Quantity); only req if an N-body calc will be run
           Rs = radii (m or Quantity); only for collision detection
        """
        raise NotImplementedError
        
    def add_random(self, n, L=50*u.AU, power=1/3, masses=1*u.M_sun, vs=50*u.km/u.s, radii=5*u.R_earth):
        """
        Add n bodies at random directions within a sphere of radius L (m)
        Density will fall off as `power` from the origin
        Masses (kg) and radii (m) can each be:
            - a list/array which will be converted to length n as needed
            - a number giving the mid-point of a distribution
        """
        
        def random_direction():
            ra = 2*np.pi*np.random.rand(n)
            dec = np.arccos(2*np.random.rand(n) - 1) - np.pi/2
            return ra, dec
        
        def random_unit_vector():
            theta, phi = random_direction()
            x = np.cos(phi) * np.sin(theta)
            y = np.cos(phi) * np.cos(theta)
            z = np.sin(phi)
            return np.vstack((x, y, z)).transpose()
        
        def create_distribution(x, sd, shape=None):
            if self.is_iterable(x): # list, may need to adjust length
                if len(x) < n:
                    reps = n//len(x) + 1
                    x = np.tile(x, reps)
                x = x[:n]
            else: # x is a midpoint, create a Gaussian around it
                # but don't let the tail go negative!
                if shape:
                    x = np.abs(np.random.standard_normal(shape)*sd + 1) * x
                else:
                    x = np.abs(np.random.randn(n)*sd + 1) * x   
            return x

        L = self.fix_units(L, u.m)
        masses = self.fix_units(masses, u.kg)
        vs = self.fix_units(vs, u.m/u.s)
        radii = self.fix_units(radii, u.m)

        distances = L * np.random.rand(n)**power
        posns = random_unit_vector() * distances[:,np.newaxis]
        
        masses = create_distribution(masses, 0.2)
        vs = create_distribution(vs, 0.3, shape=(n,3))
        radii = create_distribution(radii, 0.1)
        
        self.posns = np.concatenate((self.posns, posns))
        self.vs = np.concatenate((self.vs, vs))
        self.ms = np.concatenate((self.ms, masses))
        self.rs = np.concatenate((self.rs, radii))
        self.nBodies = self.ms.size
        self.fix_CoM()
        
#-------------------------------------------------------------------
# Methods to integrate forward in time
        
    def integrate_1body(self, t_end=None, nOrbits=1, dt=None, 
                            points_per_orbit=200, inx=-1, resTbl=True):
        """
        Run a 1-body (Keplerian) orbit.
           nOrbits = number of full periods
           dt = time step (s or Quantity)
           points_per_orbit: used to calculate dt if necessary
           inx = index into self.planets to get orbit data (defaults to last)
        """

        if self.sun is None:
            display("Error: Please run add_sun() first")
            return
        else:
            M_sun = self.ms[self.sun]
            
        # Quantities are accepted as parameters but the integrator needs dimensionless values
        t_end = self.fix_units(t_end, u.s)
        dt = self.fix_units(dt, u.s)

        # get period and calculate (if necessary) time step
        P = self.planets[inx]['P']
        if dt is None:
            dt = P/points_per_orbit # seconds
        if t_end is None:
            t_end = nOrbits*P # seconds

        # set the total time range, and the intermediate time points
        t_span = (0, t_end) # (start, end) 2-tuple
        t_vals = np.arange(0, t_end, dt) # 1-D array of time points
        
        # the solver needs a 6-vector: [x,y,z,x',y',z']
        initial_value = np.r_[self.posns[inx,:], self.vs[inx,:]]
        
        # define a callable for the solver; returns dy/dt 6-vector
        def kepler(t, y):
            print(t, y)
            r_in = y[:3]
            v_in = y[3:]
            r = self.veclen(r_in)
            a = G*M_sun/r**3 * -r_in # scalar * reciprocal position vector
            return np.r_[v_in, a]

        # let scipy.integrate do the heavy work
        sol = solve_ivp(kepler, t_span, initial_value, t_eval=t_vals, max_step=dt)
        if not sol.success:
            display(sol.message)
        
        self.time += sol.t[-1]
        final = sol.y[:,-1]
        self.posns[inx,:] = final[:3]
        self.vs[inx,:] = final[3:]
        
        if resTbl:
            self.makeResultsTable(sol)
        
    def step_1body(self, tspan, inx=0):
        self.integrate_1body(t_end=tspan, inx=inx, resTbl=False)
        # results are in self.posns, self.vs
        return self.posns[inx]
            

    def makeResultsTable(self, sol):
        # put the results in an astropy table, with units
        results = QTable()
        results['t'] = (sol.t*u.s).to(u.year)
        results['x'] = (sol.y[0,:]*u.m).to(u.AU)
        results['y'] = (sol.y[1,:]*u.m).to(u.AU)
        results['z'] = (sol.y[2,:]*u.m).to(u.AU)
        results['vx'] = sol.y[3,:]/1000*u.km/u.s
        results['vy'] = sol.y[4,:]/1000*u.km/u.s
        results['vz'] = sol.y[5,:]/1000*u.km/u.s
        results['v_tot'] = np.sqrt(results['vx']**2 + results['vy']**2 + results['vz']**2)
        results['theta_rads'] = np.arctan2(results['y'], results['x'])

        # angles -180 < theta < 180 will mess up our later plot, so convert to 0 < theta < 360
        results['theta_rads'][results['theta_rads'] < 0 ] += 2*np.pi*u.rad 
        self.results = results 

    def integrate_Nbody(self, t_end, dt=1*u.day):
        """
        Run a general N-body simulation (needn't be Keplerian)
           t_end = time to elapse (s or Quantity)
           dt = time step (s or Quantity)
        """

        t_end = self.fix_units(t_end, u.s)
        dt = self.fix_units(dt, u.s)
            
        # set the total time range, and the intermediate time points
        t_span = (0, t_end) # (start, end) 2-tuple
        t_vals = np.arange(0, t_end, dt) # 1-D array of time points
        
        # the solver needs a 6N-vector: [x,y,z,x',y',z'...]
        initial_values = np.r_[self.posns, self.vs].flatten()
        n = len(self.ms)
        
        # define a callable for the solver; returns dy/dt Nx6 array
        def keplerN(t, y):
            y = np.reshape(y, (2*n, 3))

            r_in = y[:n, :]
            v_in = y[n:, :]
            
            mm = self.ms * self.ms[:,np.newaxis]
            r = r_in - r_in[:, np.newaxis]
            rmag = np.sqrt(np.sum(r**2, -1))
            for i in range(rmag.shape[0]):
                rmag[i,i] = 1e6
            if np.any(rmag==0):
#                 print(r, rmag)
                sys.exit()
            F = G * mm[:,:,np.newaxis] * r / rmag[:,:,np.newaxis]**3
            F = np.sum(F, 1)
            a = F / self.ms[:,np.newaxis]

            return np.r_[v_in, a].flatten()

        # let scipy.integrate do the heavy work
        sol = solve_ivp(keplerN, t_span, initial_values, 
                        t_eval=t_vals, max_step=dt)
        if not sol.success:
            display(sol.message)       
        
        # update the final positions of all bodies
        final = np.reshape(sol.y[:,-1], (2*n, 3))
        self.posns = final[:n, :]
        self.vs = final[n:, :]
        
        self.time += sol.t[-1]
        nTimesteps = sol.y.shape[-1]
        #after reshape: 1st axis (x,y,z); 2nd bodies; 3rd time
        vals = sol.y.reshape([3, 2*n, nTimesteps], order='F')
        posns = vals[:, :4, :]*u.m
        vs = vals[:, 4:, :]*u.m/u.s
        times = sol.t*u.s
        return times, posns, vs
    
    def step_Nbody(self, tspan, dt=1*u.day):
        self.integrate_Nbody(t_end=tspan, dt=dt)
        # results are in self.posns, self.vs
        return self.posns*u.m
            
