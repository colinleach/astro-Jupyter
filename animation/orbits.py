import numpy as np
from astropy import units as u
from astropy.constants import G
from astropy.table import QTable

from scipy.integrate import solve_ivp


def veclen(vector):
    # beware, units are lost and this just returns a number
    return np.linalg.norm(vector) 

def vecperp(vector):
    "rotate 90 deg ccw, return normalised vector in x,y plane"
    v = np.array([-vector[1], vector[0], 0])
    return v/veclen(v)

def makeResultsTable(sol):
    # put the results in an astropy table, with units
    results = QTable()
    results['t'] = (sol.t*u.s).to(u.year)
    results['x'] = (sol.y[0,:]*u.m).to(u.AU)
    results['y'] = (sol.y[1,:]*u.m).to(u.AU)
    results['z'] = (sol.y[2,:]*u.m).to(u.AU)
    results['vx'] = sol.y[3,:]/1000*u.km/u.s
    results['vy'] = sol.y[4,:]/1000*u.km/u.s
    results['vz'] = sol.y[5,:]/1000*u.km/u.s
    return results    
    
def calcOrbit(Mstar, a, e, i=0, phi=0, nOrbits=1, dt=None, points_per_orbit=200):
    """Parameters: Mstar = central mass, with units
                   a = semimajor axis, with units
                   e = eccentricity
                   i = inclination (deg)
                   phi = orientation of perihelion (deg ccw from x-axis)
                   nOrbits = number of full periods
                   dt = time step (s)
                   points_per_orbit: used to calculate dt if necessary """

    # get everything into SI and strip off the units
    G_val = 6.67408e-11 # SI units
    Mstar = Mstar.to(u.kg).value
    a = a.to(u.m).value
    i = i*np.pi/180 # to radians
    phi = phi*np.pi/180
    
    # calculate expected orbital period and time step
    P = np.sqrt(4 * np.pi**2 * a**3/(G_val * Mstar))
    if dt is not None:
        dt = dt.to(u.s).value
    else:
        dt = P/points_per_orbit # seconds

    # set the total time range, and the intermediate time points
    t_span = (0, P*nOrbits) # (start, end) 2-tuple
    t_vals = np.arange(0, P*nOrbits, dt) # 1-D array of time points

    # set starting position and velocity, at perihelion
    r_dir = np.array([np.cos(phi), np.sin(phi), np.sin(i)]) 
    rhat = r_dir/veclen(r_dir)
    r_vec = a*(1-e) * rhat
    
    vhat = vecperp(rhat) 
    v_vec = np.sqrt(G_val*Mstar/a*(1+e)/(1-e)) * vhat
    
    # the solver needs a 6-vector: [x,y,z,x',y',z']
    initial_value = np.r_[r_vec, v_vec]

    # define a callable for the solver; returns dy/dt 6-vector
    def kepler(t, y):
        r_in = y[:3]
        v_in = y[3:]
        r = veclen(r_in)
        a = G*Mstar/r**3 * -r_in # scalar * reciprocal position vector
        return np.r_[v_in, a]
    
    # let scipy.integrate do the heavy work
    sol = solve_ivp(kepler, t_span, initial_value, t_eval=t_vals, max_step=dt)
    if not sol.success:
        display(sol.message)
    
    return makeResultsTable(sol)
