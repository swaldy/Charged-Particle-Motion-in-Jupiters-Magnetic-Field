#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 24 14:11:11 2022

@author: sarahwaldych
"""

import numpy as np
from scipy.integrate import ode
import matplotlib.pyplot as plt
import time


# parts adpoted from 
# Charged Particle Trajectories in Electric and Magnetic Fields
# F. Le Bourdais (2016)
# https://github.com/flothesof/posts/blob/master/20160128_chargedParticleMotions.ipynb

# some ideas were also adpoted from the particle motions of Earth and modified to fit Jupiter.

"""
Constants
------------------------------------------------------------------------------
"""
vel_mag_array = []
mass = 1.7e-27                     # mass of a proton (kg)
q = 1.6e-19                        # charge (C)
u0 = 4*np.pi*10**-7                # permabilitiy of free space (H/m)
c =299792458                       # speed of light (m/s)
m= [0,0,2.83*10**20]               # magnetic moment vector
m_mag = (m[2]**2)**0.5             # magnitude of magnetic moment of Jupiter (Am^2)
c = 299792458                      # speed of light (m/s)
vel_mag_array = []


def timestep(t, Y, q, mass, m, u0, m_mag):
    """
    A function that returns the new position of an electrons moving in the
    B-field.
    Approximates this as many tiny "constant acceleration" segments
    Parameters
    ----------
    t : float
    current time (s)
    dt : float
    value to step ahead in time (s)
    Y : ndarray
    state vector with position, time: [x, y, z, u, v, w]
    q : float
    charge of the particle, C (in this case, -1.6e-19)
    mass : float
    mass of the particle, kg (in this case, 9.1e-31)
    m : ndarray
    magnetic dipole vector
    u0: float
    permeability of free space
    m_mag: float
    magnitude of the moment of Earth's dipole
    Returns
    -------
    dY/dt (ndarray), the derivative of the state vector
    """

# Charged Particle Trajectories in Electric and Magnetic Fields
# F. Le Bourdais (2016)
# https://github.com/flothesof/posts/blob/master/20160128_chargedParticleMotions.ipynb
#separate out the parts of the state vector Y

    x, y, z = Y[0], Y[1], Y[2]
    u, v, w = Y[3], Y[4], Y[5]
    
#define radius r from origin (center of Jupter) in terms of x, y, z (Cartesian)

    r = (x**2+y**2+z**2)**0.5
    Bu = (3*m_mag*u0*x*z)/(4*np.pi*r**5)
    Bv = (3*m_mag*u0*y*z)/(4*np.pi*r**5)
    Bw = m_mag*u0*((z**2)-r**2)/(4*np.pi*r**5)
    B = np.array([Bu, Bv, Bw])
    
#calculate acceleration vector (dv/dt) of particle
    a = (q/mass)*np.cross(np.array([u, v, w]), B)
# Charged Particle Trajectories in Electric and Magnetic Fields
# F. Le Bourdais (2016)
# https://github.com/flothesof/posts/blob/master/20160128_chargedParticleMotions.ipynb
#return the derivative of the state vector dY/dt

    return np.array([u, v, w, a[0], a[1], a[2]])
"""
Initialize variables (change parameters, test name here!)
"""

#name of the test on the csv outputs for position and speed
test_name = '9'
#initialize initial time, end time, timestep reported
t0 = 0
t1 = 5000
dt = 0.1 


#initialize variables
# Charged Particle Trajectories in Electric and Magnetic Fields
# F. Le Bourdais (2016)
# https://github.com/flothesof/posts/blob/master/20160128_chargedParticleMotions.ipynb


#initial position
x0 = np.array([-6e9,-6e9,-6e9])
#initial velocity
v0 = np.array([0.009*c,0, 0])
state_initial = np.concatenate((x0, v0))


#path to save the positions csv output
csv_save_path = "./pos_outputs"
#path to save the speed output
vel_save_path = "./vel_outputs"
#plotting variables
#determines if a b_field will be plotted on this first plot
b_field = False
#determine the axis limits (will be a cube-shaped plot)
ax_limits = 7e9


"""
Initialize constants here
"""

#permeability of free space (H/m)
u0 = 4*np.pi*10**-7
#moment of Jupiters magnetic field (A/m^2), and its magnitude
m = [0, 0, 2.83e26 ]
m_mag = np.sqrt(m[0]**2 + m[1]**2 + m[2]**2)
"""
Initialize the integrator, desired error, etc. (ode set_integrator documentation)
# Charged Particle Trajectories in Electric and Magnetic Fields
# F. Le Bourdais (2016)
# https://github.com/flothesof/posts/blob/master/20160128_chargedParticleMotions.ipynb
"""
r = ode(timestep).set_integrator('dopri5')
r.set_initial_value(state_initial, t0).set_f_params(q, mass, m, u0, m_mag)
t = 0
state = state_initial


"""
Perform the test run of the particle
"""


#make a list to append position arrays for each timestep
positions = []
positions.append(state_initial[:3])
#see how long to code took to run
start = time.perf_counter()
while r.successful and r.t < t1:
    
    r.integrate(r.t+dt)
    positions.append(r.y[:3])
    vel_mag_array.append(np.linalg.norm(r.y[3:]))
#get the seconds it took to calculate the path
seconds = time.perf_counter() - start
#convert positions to numpy array (nx3 array)
positions = np.array(positions)

"""
Plot the image, including the B-field
"""

#create the figure
fig = plt.figure(figsize=(20,20))
ax = fig.add_subplot(111, projection='3d')
#plot the magnetic field vectors on top of plot, if desired
if b_field:
#make the grid
    spacing = np.linspace(-ax_limits, ax_limits, 11)
    gridaxis = np.meshgrid(spacing)
    x, y, z = np.meshgrid(gridaxis, gridaxis, gridaxis)
#define radius r from origin in terms of x, y, z (spherical --> Cartesian)
    r = (x**2+y**2+z**2)**0.5
#define magnetic field vector components in terms of Cartesian points
    Bu = (3*m_mag*x*z)/r**5
    Bv = (3*m_mag*y*z)/r**5
    Bw = m_mag*(3*(z**2)-r**2)/r**5
    ax.quiver(x, y, z, Bu, Bv, Bw, length=0.5*10**9, linewidth=0.5, normalize=True)
#plot the origin, where Earth is
ax.plot([0], [0], [0], markeredgecolor='r', marker='o', markersize=5, linewidth=0, label='origin')
#plot the path of the particle
ax.plot3D(positions[:, 0], positions[:, 1], positions[:, 2], c='red')
#set limits on the graph
plt.xlabel('x')
plt.ylabel('y')
ax.set_zlabel('z')
plt.xlim(-ax_limits, ax_limits)
plt.ylim(-ax_limits, ax_limits)
ax.set_zlim(-ax_limits, ax_limits)
#see how many seconds it took to run
print(seconds)
#put the speeds at each timestep into an array, to see how speed was conserved at each timestep
vel_mag_array = np.array(vel_mag_array)
#save these outputs as a CSV for later analysis, plotting, or animation creation
np.savetxt(csv_save_path+"/test"+test_name+".csv", positions, delimiter=",")
np.savetxt(vel_save_path+"/test"+test_name+"_vel.csv", vel_mag_array, delimiter=",")
