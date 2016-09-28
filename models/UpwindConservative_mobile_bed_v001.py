#!/usr/bin/env python

'''
Notes

I applied a flux treatment to the bed
calculating bed load only. This is not how it is descrived in the book, rather
just a nieve approach. That said it does appear stable although I believe that
the bedload transport rate is underpredicted.
'''


import numpy as np
import matplotlib.pyplot as plt
import math
from copy import copy, deepcopy
from flux_solvers import *
from sed_trans import *

#--------------------------------------------------------------------------
# Global variables and constants
#-------------------------------------------------------------------------




def main_EULER():

    #--------------------------------------------------------------------------
    # Model run parameters - These are set by the user
    #--------------------------------------------------------------------------

    # Number of nodes
    number_nodes =100
    # max time
    t_max = 20


    # Domain Length
    L = 1000
    centre_cell = int(math.floor(number_nodes/2))
    dx = L/(number_nodes-1)
    # Initial epth settings
    depth_l = 4
    depth_r = 10

    D0 = 0.002
    pm = get_bed_porosity(D0)
    rho_particule=2650


    n = 0.03
    # Courant Number
    Cn=0.1
    width = 1.

    ''' Initialize the domain '''
    x = np.zeros(number_nodes)

    Zbed = np.ones(number_nodes)
    Zbed = Zbed*2
    Z = np.zeros(number_nodes)
    Q = np.zeros(number_nodes)
    Qb = np.zeros(number_nodes)

    Z[0:centre_cell+1] = depth_l
    Z[centre_cell:] = depth_r

    A = (Z-Zbed)*width
    h = (Z-Zbed)

    for i in range(len(x)):
        x[i] = i*dx

    ''' Set the water elevations '''
    Z_init = deepcopy(Z)
    Zbed_init = deepcopy(Zbed)
    Q_init = deepcopy(Q)

    dQbdx_final = None

    method= 'hll'


    ''' ---------------------------------------------'''
    ''' Starting main loop here '''
    ''' ---------------------------------------------'''
    t = 0
    cntr = 0
    while t < t_max:
        ''' calculate the courant number '''
        U=Q/A
        dt = Cn * dx / max(abs(U) + np.sqrt(G*A/width))

        cntr += 1
        t += dt
        print t,dt



        dAdx = np.zeros(number_nodes)
        dQdx = np.zeros(number_nodes)
        dQbdx = np.zeros(number_nodes)

        A_tmp = deepcopy(A)
        Q_tmp = deepcopy(Q)
        Qb_tmp = deepcopy(Qb)
        Zbed_tmp = deepcopy(Zbed)
        km = np.zeros(number_nodes)
        kp = np.zeros(number_nodes)

        for i in range(3,number_nodes-2):
            Flux_m=  np.zeros(2)
            Flux_p = np.zeros(2)

            if method == 'hll':
                Flux_m,km[i]  = get_hll(A[i-1], A[i], Q[i-1], Q[i],width)
                Flux_p,kp[i]  = get_hll(A[i], A[i+1], Q[i], Q[i+1],width)
            else:
                Flux_m,km[i]  = get_upwind_flux(A[i-1], A[i], Q[i-1], Q[i])
                Flux_p,kp[i]  = get_upwind_flux(A[i], A[i+1], Q[i], Q[i+1])

            dAdx[i] = (1./dx)*(Flux_p[0]-Flux_m[0])
            dQdx[i] = (1./dx)*(Flux_p[1]-Flux_m[1])

        for i in range(3,number_nodes-2):
            ''' Calculate the friction before updating the Area '''
            R = (A[i]/(2*Z[i] + width))
            Sf = G*(n**2.)*Q[i]*abs(Q[i])/(A[i]*R**(4./3.) )

            ''' Update the area '''
            A[i] = A_tmp[i] - dt*dAdx[i]

        ''' Update the depth'''

        Z = (A/width + Zbed)
        h = (Z-Zbed)
        U=Q/A

        for i in range(4,number_nodes-3):
            ''' Calculate the slope based on the update surface '''
            w1 = get_w1(U,x,i,km[i],dt)
            w2 = get_w2(U,x,i,kp[i],dt)

            dzdx_down =get_delta_zdown(Z, x, i, km[i])
            dzdx_up = get_delta_zup(Z, x, i, kp[i])

            ''' Water slope '''
            S = -G*A[i]*(w1*dzdx_down + w2*dzdx_up) + Sf
            Q[i] = Q_tmp[i] - dt*dQdx[i] + dt*S

            ''' Calculate the sediment transport fluxes '''
            Flux_Qsb_in = get_upwind_bedload_flux(h[i-1], h[i], U[i-1], U[i],D0,rho_particule)
            Flux_qsb_out = get_upwind_bedload_flux(h[i], h[i+1], U[i], U[i+1],D0,rho_particule)

            Qb[i] = get_unit_bed_load(h[i],U[i],D0,rho_particule)
            dQbdx[i] = (1./dx)*(Flux_qsb_out-Flux_Qsb_in)

            Zbed[i] = Zbed_tmp[i] - (dt)*(1./(1.-pm))*(Flux_qsb_out-Flux_Qsb_in)

        dQbdx_final = dQbdx
        ''' Update the bed parameters '''
        A = (Z-Zbed)*width
        h = (Z-Zbed)
        print 'Max dqb/dx: %s, min: %s' % (max(dQbdx), min(dQbdx))


    fig = plt.subplot(5,1,1)
    plt.plot(x,Z,x,Z_init,x,Zbed,x,Zbed_init)
    fig.set_title('Water Surface')

    fig = plt.subplot(5,1,2)
    plt.plot(x,Q)
    fig.set_title('Flow (m^2s-1)')

    fig = plt.subplot(5,1,3)
    plt.plot(x,U)
    fig.set_title('Velocity m/s')

    fig = plt.subplot(5,1,4)
    plt.plot(x,Qb)
    fig.set_title('Qbed')

    fig = plt.subplot(5,1,5)
    plt.plot(x,dQbdx_final)
    fig.set_title('grad(qbed)')
    plt.show()

    print 'OK'
if __name__ == "__main__":
    main_EULER()

