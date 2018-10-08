#*********************************************************************
#
#  File: StellarBinary_suppl.py
#
#  Various function definitions to support the StellarBinary notebook
#  Colin Leach, Sept 2018
#
#*********************************************************************


import time

import numpy as np
import matplotlib.pyplot as plt

from IPython.display import Image, HTML, display, clear_output
from matplotlib.offsetbox import AnchoredText

from ipywidgets import interact, interactive, fixed, interact_manual, Layout, Output
import ipywidgets as w

from astropy import units as u
from astropy.constants import G, sigma_sb, L_sun



#*********************************************************************
#
#	transformation()
#
#	Converts orbital coordinates to plane-of-sky (primed) coords
#
#*********************************************************************

def transformation(x, y, i):
    xp = x*np.sin(i)
    yp = y
    zp = -x*np.cos(i)
    
    return (xp, yp, zp)
    

#*********************************************************************
#
#	limbDarkening(), F(), fluxIntegral(), eclipse()
#
#	All are functions used in predicting the light curve
#
#*********************************************************************

def limbDarkening(r_prime, R):
    """
    This routine calculates limb darkening for a specific value 
    of the distance from the center of the stellar disk, r_prime.
    R is the full radius of the disk.
    r_prime and R should have the same units (or none)
    Return value is a dimensionless coefficient

    Data are due to Van Hamme, W., Astronomical Journal, 
      Vol. 106, 1096, 1993. 
    """
        
    # Van Hamme model #109 (solar-like)
    x = 0.648
    y = 0.207
    ang = (2*r_prime/(R*np.pi))*u.rad
    mu = np.cos(ang)
    return 1 - x*(1-mu) - y*mu*np.log(mu)

def F(r_prime, R, T):
    """This routine calculates the flux for a specific value
        of the distance from the center of the stellar disk, r_prime"""
    
    flux = sigma_sb * T**4 * limbDarkening(r_prime, R)
    return flux

def fluxIntegral(R1, T1, R2, T2):
    """
    Flux varies with radius in a non-trivial way, 
    so divide the disk into concentric rings for numerical integration
    """
    
    Nr = 100 # number of rings
    
    # radii and dr values for iteration
    drS1 = R1/Nr
    drS2 = R2/Nr
    rS1 = 0 * u.m
    rS2 = 0 * u.m
    
    # cumulative brightness
    S1 = 0 * u.W
    S2 = 0 * u.W
    
    for j in range(Nr):
        rS1 += drS1
        rS2 += drS2
        f1 = F(rS1 - drS1/2, R1, T1)
        f2 = F(rS2 - drS2/2, R2, T2)
        S1 += 2*np.pi * f1 * (rS1 - drS1/2) * drS1
        S2 += 2*np.pi * f2 * (rS2 - drS2/2) * drS2
    
    return S1 + S2

def eclipse(x1, y1, R1, T1, x2, y2, R2, T2, i):
    "Computes the change in observed luminosity due to an eclipse"
    
    # number of steps for r', theta' integrations
    Nr = 100
    Ntheta = 500 # 500
    dtheta_prime = (np.pi / Ntheta) * u.rad
    
    dS = 0 * u.W
    
    # transform coordinates
    x1p, y1p, z1p = transformation(x1, y1, i)
    x2p, y2p, z2p = transformation(x2, y2, i)
    
    # Determine which star is in front (f) and which star is in back (b)
    if x1p > 0:
        xfp = x1p; yfp = y1p; zfp = z1p; Rf = R1; Tf = T1
        xbp = x2p; ybp = y2p; zbp = z2p; Rb = R2; Tb = T2
    else:
        xfp = x2p; yfp = y2p; zfp = z2p; Rf = R2; Tf = T2
        xbp = x1p; ybp = y1p; zbp = z1p; Rb = R1; Tb = T1
        
    # Are the two stars close enough for an eclipse?
    d = np.sqrt((yfp - ybp)**2 + (zfp - zbp)**2)
    if d <= Rf + Rb:
        
        # Find the angle between y' and the projected line between the 
        # centers of the stars in the (y',z') plane.  The polar coordinate
        # integration will be centered on this line to take advantage 
        # of spherical symmetry.
        theta0_prime = np.arctan2((zfp - zbp), (yfp - ybp))
        
        # Determine the starting radius for the integration
        if d < Rb - Rf:
            r_prime = d + Rf  # Foreground star disk entirely 
            r_stop  = d - Rf  # inside background star disk
            if r_stop < 0:
                r_stop = 0
        else:
            r_prime = Rb
            r_stop = 0
        dr_prime = r_prime / Nr
        
        # The surface integration loop
        while True:
            # Determine the limits of the angular integration for the current r_prime
            
            # force Quantity to copy value, not just pointer
            # (this was previously a hard bug to spot!)
            theta_prime = theta0_prime.value * u.rad
            finished = False
            while not finished:
                yp_dA = r_prime * np.cos(theta_prime + dtheta_prime) + ybp
                zp_dA = r_prime * np.sin(theta_prime + dtheta_prime) + zbp
                if np.sqrt((yp_dA - yfp)**2 + (zp_dA - zfp)**2) > Rf:
                    finished = True
        
                theta_prime += dtheta_prime
                if theta_prime - theta0_prime > np.pi*u.rad:
                    finished = True
                    
            dS += 2 * F(r_prime - dr_prime/2, Rb, Tb) * (r_prime - dr_prime/2) \
                * dr_prime * (theta_prime - theta0_prime)/u.rad
                
            # Check to see that there is no remaining overlap or if center of disk has been reached
            r_prime -= dr_prime
            
            if r_prime >= r_stop:
                break
    
    return (dS, x1p, x2p, y1p, z1p, y2p, z2p)

#*********************************************************************
#
#	runSim()
#
#	Main loop, populating the data dictionary
#
#*********************************************************************

def runSim(stars, steps=1000):
    "main stellar orbit loop"

    # initialize parameters
    N = steps # steps around the orbit
    t = 0 * u.s
    dt = stars['P']/N # time step [s]
    theta = 0 * u.rad
    
    # calculate angular momentum about center of mass
    L_ang = stars['mu'] * np.sqrt(G * stars['M'] * stars['a'] * (1 - stars['e']**2))
    
    # dA/dt, for calculating dtheta at each time step
    dAdt = L_ang / (2 * stars['mu'])
    
    # combined flux from the 2 stars when not eclipsed
    S = fluxIntegral(stars['R1'], stars['T1'], stars['R2'], stars['T2'])
    
    # bolometric magnitude of our sun - slightly arbitrary
    Mbol_sun = 4.74 
    
    # initialize output arrays, 
    # either creating new dict keys or overwriting existing ones
    stars['t_P'] = np.zeros(N) # t/P, fraction of the orbit [dimensionless]
    stars['v1r'] = np.zeros(N) * u.km/u.s # radial velocity, star 1
    stars['v2r'] = np.zeros(N) * u.km/u.s # radial velocity, star 2
    stars['Mbol'] = np.zeros(N) # bolometric magnitude [dimensionless]
    stars['dS'] = np.zeros(N) * u.W # dip in output
    stars['xp_diff'] = np.zeros(N) * u.AU # coord along line of sight, for deciding which star is in front
    stars['y1p'] = np.zeros(N) * u.AU # coords on plane of sky, star 1
    stars['z1p'] = np.zeros(N) * u.AU
    stars['y2p'] = np.zeros(N) * u.AU # star 2
    stars['z2p'] = np.zeros(N) * u.AU
    stars['x1'] = np.zeros(N) * u.AU # coords on orbital plane, star 1
    stars['y1'] = np.zeros(N) * u.AU
    stars['x2'] = np.zeros(N) * u.AU # star 2
    stars['y2'] = np.zeros(N) * u.AU
    #~ stars['ycmp'] = np.zeros(N) * u.AU # center of mass
    #~ stars['zcmp'] = np.zeros(N) * u.AU
    stars['theta'] = np.zeros(N) * u.deg
    
    # run the time-step loop
    for step in range(N):
        # Calculate orbit parameters in the CoM reference frame,
        # then translate to individual star positions
        
        # position
        r =  stars['a']*(1 - stars['e']**2)/(1 + stars['e']*np.cos(theta))
        
        # velocity
        v = np.sqrt(G * stars['M'] * (2/r - 1/stars['a']))
        
        # radial velocity (along our line of sight)
        vr  = -v * np.sin(stars['i']) * np.sin(theta + stars['phi'])
        # radial velocity for each star
        v1r = (stars['mu']/stars['m1']) * vr
        v2r = -(stars['mu']/stars['m2']) * vr
        
        # Determine (x,y) positions of star centers, i.e. in plane of orbit
        
        # CoM ref frame
        x   = r * np.cos(theta + stars['phi'])
        y   = r * np.sin(theta + stars['phi'])
        # each star
        x1 = (stars['mu']/stars['m1']) * x
        y1 = (stars['mu']/stars['m1']) * y
        x2 = -(stars['mu']/stars['m2']) * x
        y2 = -(stars['mu']/stars['m2']) * y
        
        # Check if there is an eclipse at this time step
        # As a by-product, this also returns plane-of-sky (primed) coordinates 
        dS, x1p, x2p, y1p, z1p, y2p, z2p = eclipse(x1, y1, stars['R1'], stars['T1'], 
                                         x2, y2, stars['R2'], stars['T2'], 
                                         stars['i'])
        
        # Calculate luminosity and bolometric magnitude
        # These should be constant, except for a dip during each eclipse (if any)
        Lt = stars['L'] * (1 - dS/S)
        Mbol = Mbol_sun - 5 * np.log10(Lt / L_sun)
            
        # store the results
        stars['t_P'][step] = t/stars['P']
        stars['v1r'][step] = v1r
        stars['v2r'][step] = v2r 
        stars['Mbol'][step] = Mbol
        stars['dS'][step] = Lt*dS/S
        stars['xp_diff'][step] = x1p - x2p
        stars['y1p'][step] = y1p
        stars['z1p'][step] = z1p
        stars['y2p'][step] = y2p
        stars['z2p'][step] = z2p
        stars['x1'][step] = x1
        stars['y1'][step] = y1
        stars['x2'][step] = x2
        stars['y2'][step] = y2
        stars['theta'][step] = theta
        
        # prepare for next step
        dtheta = 2 * dAdt/r**2 * dt * u.rad
        theta += dtheta
        t += dt

#*********************************************************************
#
#	saveData()
#
#	Rarely-used function to dump results to a text file
#
#*********************************************************************

def saveData(fname='stars.dat'):
	"Write data from the sim to a text file"

	fields = ['t_P','v1r','v2r','Mbol','dS','y1p','z1p','y2p','z2p','x1','y1','x2','y2','theta']
	headerline = ','.join(fields)
	vals = {}
	for f in fields:
		try:
			vals[f] = stars[f].value
		except:
			vals[f] = stars[f]
	stacked = (vals[f] for f in fields)
	alldata = np.transpose(np.stack(stacked))
	np.savetxt(fname, alldata, delimiter=',', header=headerline)

#*********************************************************************
#
#	lookup_temp()
#
#	A just-for-fun routine, converting blackbody temp to RGB color
#
#*********************************************************************

def lookup_temp(T):
    """
        Input is a temperature in K
        Must be a value, e.g. float, not an astropy Quantity
        
        For the data and discussion, see:
        http://www.vendian.org/mncharity/dir3/blackbody/
    """
    
    import csv
    reader = csv.reader(open('rgb_temp.csv', 'r'))
    tempdict = {}
    for row in reader:
       k, v = row
       tempdict[k] = v

    # lookup table only has data for 1k <= T <= 40k
    if T < 1000:
        T = 1000
    elif T > 40000:
        T = 40000
    tempkey = str(int(T/100)*100)
    return tempdict[tempkey];    
    
#*********************************************************************
#
#	plotOrbit()
#
#	Main plotting routine
#	Moved here because it is too verbose for the Jupyter notebook
#
#*********************************************************************

def plotOrbit(stars, figwidth = 15, figheight = 10):
    "Display sim results with Matplotlib"
    
    tic = time.time()
    # run the calculation
    runSim(stars)
    toc = time.time()
    print("sim took {:.3f} s".format(toc-tic))
    
    # cosmetic details: plot-marker-sizes for the stars
    R_min = min(stars['R1'], stars['R2'])
    R_max = max(stars['R1'], stars['R2'])
    msize_min = 10
    msize_max = int(msize_min*R_max/R_min)
    if stars['R1'] > stars['R2']:
        msize1 = msize_max
        msize2 = msize_min
    else:
        msize1 = msize_min
        msize2 = msize_max
    color1 = lookup_temp(stars['T1'].value)
    color2 = lookup_temp(stars['T2'].value)
        
    # make the plot
    plt.ioff()
    fig = plt.figure(figsize=(figwidth, figheight))

    # some labelling
    ax1 = fig.add_subplot(231)
    plt.axis('off')
    
    at1 = AnchoredText("m1:  {0:.2f}\nR1:  {1:.2f}\nT1:  {2:.0f}\na1: {3:.3f}\nL1: {4:.3g}"
                        .format(stars['m1'].to(u.M_sun), stars['R1'].to(u.R_sun), 
                                stars['T1'], stars['a1'].to(u.AU), stars['L1']),
                      prop=dict(size=13, color='r'), frameon=False,
                      loc=2,
                      )
    at1.patch.set_boxstyle("round,pad=0.1,rounding_size=0.2") # ,color='r'
    ax1.add_artist(at1)
    
    at2 = AnchoredText("m2:  {0:.2f}\nR2:  {1:.2f}\nT2:  {2:.0f}\na2: {3:.3f}\nL2: {4:.3g}"
                        .format(stars['m2'].to(u.M_sun), stars['R2'].to(u.R_sun), 
                                stars['T2'], stars['a2'].to(u.AU), stars['L2']),
                      prop=dict(size=13, color='g'), frameon=False,
                      loc=1,
                      )
    at2.patch.set_boxstyle("round,pad=0.1,rounding_size=0.2") # ,color='g'
    ax1.add_artist(at2)
    
    at = AnchoredText("e:  {0:.2f}\ni:  {1:.0f}\n$\phi$:  {2:.0f}\nPeriod: {3:.4f}" #\nMbol_max: {4:.3f}\nt_max: {5:.3f}"
                        .format(stars['e'], stars['i'], stars['phi'], 
                                stars['P'].to(u.year)), #, stars['Mbol_max'], stars['t_max'].to(u.year)),
                      prop=dict(size=13), frameon=False,
                      loc=6,
                      )
    at.patch.set_boxstyle("round,pad=0.1,rounding_size=0.2")
    ax1.add_artist(at)
    
    # add the plane-of-sky plot
    yrange = max(np.max(stars['y1p']) - np.min(stars['y1p']), np.max(stars['y2p']) - np.min(stars['y2p']))
    zrange = max(np.max(stars['z1p']) - np.min(stars['z1p']), np.max(stars['z2p']) - np.min(stars['z2p']))
    if np.abs(zrange/yrange) < 0.1: # problem scaling vertical axis if orbit edge-on
        axo = fig.add_subplot(232)
    else:
        axo = fig.add_subplot(232, aspect='equal')
    axo.plot(stars['y1p'], stars['z1p'], lw=2, color='r')
    axo.plot(stars['y2p'], stars['z2p'], lw=2, color='g')
    axo.plot(0,0,'b+',markersize=20)
    axo.plot(stars['y1p'][1], stars['z1p'][1], 'o', markersize=msize1, color=color1, markeredgecolor='k')
    axo.plot(stars['y2p'][1], stars['z2p'][1], 'o', markersize=msize2, color=color2, markeredgecolor='k')
    axo.set_xlabel("y' (AU)", fontsize=14) #, fontsize=18)
    axo.set_ylabel("z' (AU)", fontsize=14) #, fontsize=18)
    axo.set_title("Orbit on the plane of the sky", fontsize=16)

    # add the orbit plot
    axo2 = fig.add_subplot(233, aspect='equal')
    axo2.plot(stars['x1'], stars['y1'], lw=2, color='r')
    axo2.plot(stars['x2'], stars['y2'], lw=2, color='g')
    axo2.plot(0,0,'b+',markersize=20)
    axo2.plot(stars['x1'][1], stars['y1'][1], 'ro', markersize=msize1, color=color1, markeredgecolor='k')
    axo2.plot(stars['x2'][1], stars['y2'][1], 'go', markersize=msize2, color=color2, markeredgecolor='k')
    axo2.set_xlabel("x (AU)", fontsize=14) 
    axo2.set_ylabel("y (AU)", fontsize=14) 
    axo2.set_title("Orbital plane", fontsize=16)

    # add the light curve
    axl = fig.add_subplot(234)
    axl.plot(stars['t_P'], stars['dS'], lw=2)
    axl.set_xlabel("t/P", fontsize=14) #, fontsize=18)
    axl.set_ylabel("dS (W)", fontsize=14) #, fontsize=18)
    axl.set_title("Light Curve", fontsize=16)

    # add the radial velocity plot
    axv = fig.add_subplot(235)
    axv.plot(stars['t_P'], stars['v1r'], lw=2, color='r')
    axv.plot(stars['t_P'], stars['v2r'], lw=2, color='g')
    axv.set_xlabel("t/P", fontsize=14) #, fontsize=18)
    axv.set_ylabel("radial velocity (km/s)", fontsize=14) #, fontsize=18)
    axv.set_title("Radial Velocities", fontsize=16)

    # add the angle plot
    axv = fig.add_subplot(236)
    axv.plot(stars['t_P'], stars['theta'], lw=2, color='b')
    axv.set_xlabel("t/P", fontsize=14) #, fontsize=18)
    axv.set_ylabel("theta (deg)", fontsize=14) #, fontsize=18)
    axv.set_title("Orbit Angle", fontsize=16)

    plt.tight_layout()  # to stop the plots overlapping
    plt.show()                
    toc = time.time()
#     print("plot took {:.3f} s".format(toc-tic))


#***************************************************************************
#
#	widgetDict()
#
#	Defines an alrming number of interactive controls to set sim parameters
#
#***************************************************************************

def widgetDict(stars):
	
	style = {'description_width': 'initial'} # to avoid the labels getting truncated

	wd = dict(
		m1_Msun = w.FloatText(description="star 1 mass ($M_{\odot}$)", style=style,
								layout=Layout(width='20%'),
								continuous_update=False, 
								value=round(stars['m1'].to(u.M_sun).value,2)),
		R1_Rsun = w.FloatText(description="star 1 radius ($R_{\odot}$)", style=style,
								layout=Layout(width='20%'),
								continuous_update=False, 
								value=round(stars['R1'].to(u.R_sun).value,2)),
		T1 = w.FloatText(description="star 1 $T_e$ (K)", style=style,
								layout=Layout(width='20%'),
								continuous_update=False, 
								value=stars['T1'].value),
		m2_Msun = w.FloatText(description="star 2 mass ($M_{\odot}$)", style=style,
								layout=Layout(width='20%'),
								continuous_update=False, 
								value=round(stars['m2'].to(u.M_sun).value,2)),
		R2_Rsun = w.FloatText(description="star 2 radius ($R_{\odot}$)", style=style,
								layout=Layout(width='20%'),
								continuous_update=False, 
								value=round(stars['R2'].to(u.R_sun).value,2)),
		T2 = w.FloatText(description="star 2 $T_e$ (K)", style=style,
								layout=Layout(width='20%'),
								continuous_update=False, 
								value=stars['T2'].value),
		P_day = w.FloatText(description="Orbit period (days)", style=style,
								layout=Layout(width='20%'),
								continuous_update=False, 
								value=stars['P'].to(u.day).value),
		e = w.FloatText(description="eccentricity", style=style,
								layout=Layout(width='20%'),
								continuous_update=False, 
								value=stars['e']),
		i = w.FloatSlider(description="inclination (deg)", style=style,
								layout=Layout(width='80%'),
								continuous_update=False,
								min=0.0, max=90.0, 
								value=stars['i'].value), 
		phi = w.FloatSlider(description="$\phi$ (deg)", style=style,
								layout=Layout(width='80%'),
								continuous_update=False,
								min=0.0, max=90.0, 
								value=stars['phi'].value), 
		)
	return wd
