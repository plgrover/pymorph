#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import math
from copy import copy, deepcopy
from flux_solvers import *

#--------------------------------------------------------------------------
# Global variables and constants
#-------------------------------------------------------------------------

G = 9.81

def main():

    #--------------------------------------------------------------------------
    # Model run parameters - These are set by the user
    #--------------------------------------------------------------------------

    # Number of nodes
    number_nodes =100
    # max time
    t_max = 30


    # Domain Length
    L = 1000
    centre_cell = int(math.floor(number_nodes/2))
    dx = L/(number_nodes-1)
    # Initial epth settings
    depth_l = 10.
    depth_r = 2.
    S0 = 0.
    n = 0.0
    # Courant Number
    Cn=0.2

    ''' Initialize the domain '''
    x = np.zeros(number_nodes)
    for i in range(len(x)):
        x[i] = i*dx

    h = np.zeros(number_nodes)
    q = np.zeros(number_nodes)

    ''' Set the water elevations '''
    h[0:centre_cell+1] = depth_l
    h[centre_cell:] = depth_r


    h_int = deepcopy(h)
    q_int = deepcopy(q)


    ''' ---------------------------------------------'''
    ''' Starting main loop here '''
    ''' ---------------------------------------------'''
    try:
        t = 0
        cntr = 0
        while t < t_max:
            ''' calculate the courant number '''
            dt = Cn * dx / max(abs(q/h) + np.sqrt(G*h))

            cntr += 1
            t += dt
            print t,dt

            dhdx = np.zeros((number_nodes,2))
            dqdx = np.zeros((number_nodes,2))

            h_tmp = deepcopy(h)
            q_tmp = deepcopy(q)

            for stage in range(2):
                for i in range(3,number_nodes-2):

                    Flux_m  = get_hll(h[i-1], h[i],q[i-1],q[i])
                    Flux_p  = get_hll(h[i], h[i+1],q[i],q[i+1])

                    ''' Calculate the friction slope '''
                    Sf = (n**2.)*q[i]*abs(q[i])/(h[i]**(10./3.))
                    S = G*h[i]*(S0-Sf)

                    dhdx[i,stage] = (1./dx)*(Flux_p[0]-Flux_m[0])
                    dqdx[i,stage] = (1./dx)*(Flux_p[1]-Flux_m[1]) + S


                for i in range(3,number_nodes-2):
                    if stage == 0:
                        h[i] = h_tmp[i] - dt*dhdx[i,0]
                        q[i] = q_tmp[i] - dt*dqdx[i,0]
                    else:
                        h[i] = h_tmp[i] - 0.5 * dt*(dhdx[i,0] + dhdx[i,1])
                        q[i] = q_tmp[i] - 0.5 * dt*(dqdx[i,0] + dqdx[i,1])


        u = q/h
        plt.plot(x,q)
        plt.show()

        print 'OK'

    except Exception, err:
        print h
        print Exception, err








if __name__ == "__main__":
    main()
