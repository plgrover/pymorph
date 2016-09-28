#!/usr/bin/env python

'''
Notes

Applied the proper bedload formula:

Qb/ub + dQb/dx = 1/l(Qbed* - Qbed)

Running against the test cases from Taipai and Louvain
the model becomes unstable.

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
    number_nodes =401

    sed_flag = True

    # Domain Length
    L = 2.
    centre_cell = int(math.floor(number_nodes/2))
    dx = L/(number_nodes-1)
    print dx
    # Initial epth settings
    depth_l = 0.1
    depth_r = 0.003

     # max time
    t0 = math.sqrt(depth_l/G)
    t_max = 1.5*t0

    case = 'Louvain'
    method = ''

    D0 = 0.0035
    rho_particule=1540.0

    if case == 'Louvain':
        D0 = 0.0035
        rho_particule=1540.0
    else:
        D0 = 0.0061
        rho_particule=1048.0

    pm = get_bed_porosity(D0)
    Ycr = get_Ycr(D0,rho_particule)


    n = 0.00
    # Courant Number
    Cn=0.1
    width = 1.

    ''' Initialize the domain '''
    x = np.zeros(number_nodes)

    Zbed = np.ones(number_nodes)
    Zbed = Zbed*2.0
    Z = np.zeros(number_nodes)
    Q = np.zeros(number_nodes)
    Qbu = np.zeros(number_nodes)
    Qbed = np.zeros(number_nodes)
    Qb_star = np.zeros(number_nodes)
    eta = np.zeros(number_nodes)

    Z[0:centre_cell+1] = depth_l + Zbed[0:centre_cell+1]
    Z[centre_cell:] = depth_r + Zbed[centre_cell:]

    A = (Z-Zbed)*width
    h = (Z-Zbed)

    for i in range(len(x)):
        x[i] = i*dx

    ''' Set the water elevations '''
    Z_init = deepcopy(Z)
    Zbed_init = deepcopy(Zbed)
    Q_init = deepcopy(Q)

    dQbdx_final = None


    ''' ---------------------------------------------'''
    ''' Starting main loop here '''
    ''' ---------------------------------------------'''
    t = 0
    cntr = 0
    try:
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
            Qbu_tmp = deepcopy(Qbu)
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

                U=Q/A

                ''' Calculate the sediment transport fluxes '''

                if sed_flag==True:
                    Flux_Qsb_in = get_upwind_bedload_flux_neq(Qbed[i-1], Qbed[i],Q_tmp[i])
                    Flux_qsb_out = get_upwind_bedload_flux_neq(Qbed[i], Qbed[i+1], Q_tmp[i])
                    dQbdx[i] = (1./dx)*(Flux_qsb_out-Flux_Qsb_in)

                    Qb_star[i] = get_unit_bed_load(h[i],U[i],D0,rho_particule)

                    #if abs(U[i]) > 0.0001:
                    #     Qbed[i] = Qbu_tmp[i]/U[i]
                    L = 6 * h[i]

                    Qbu[i] = Qbu_tmp[i] - (dt)*dQbdx[i] + (dt/L)*(Qb_star[i] - Qbed[i])

                    ''' Calculate ub '''
                    Qbed[i]=0
                    if abs(U[i]) > 0.0001:
                         Qbed[i] = Qbu[i]/U[i]

                    if i==200:
                        print 'A: %s, Q: %s, U: %s, Qbu: %s, Qbed: %s, h: %s' % (A[i], Q[i], U[i], Qbu[i],Qbed[i],h[i])
                        print 'dQbdx: %s at node: %s' %(dQbdx[i],i)

                    if Qbed[i]> 1000:
                        print 'Qbed is spiking: %s' % Qbed[i]
                        print 'Upwind: %s    Downwind: %s ' %(  Qbed[i-1], Qbed[i+1] )
                        print '------------------------------------------------------'
                        print 'A: %s, Q: %s, U: %s, Qbu: %s, Qbed: %s, h: %s' % (A[i], Q[i], U[i], Qbu[i],Qbed[i],h[i])
                        print 'dQbdx: %s at node: %s' %(dQbdx[i],i)

                        return
                    #Zbed[i] = Zbed_tmp[i] - (dt/dx)*(1./(1-pm))*(1./L)*(Qb_star[i] - Qbed[i])*0

                eta[i] = get_Y(h[i],U[i],D0,rho_particule)/Ycr
            ''' Update the bed parameters '''
            A = (Z-Zbed)*width
            h = (Z-Zbed)
    except :
        print 'Error'

    Qb = np.zeros(number_nodes)
    ''' Calculate the final bed load '''
    for i in range(4,number_nodes-3):
        if U[i] != 0:
            Qb[i] = Qbu[i]/U[i]



    fig = plt.subplot(5,1,1)
    plt.plot(x,Z,label='Final WS')
    plt.plot(x,Z_init,label='inital WS')
    plt.plot(x,Zbed_init,label='inital Zbed')
    plt.plot(x,Zbed,label='Final Zbed')
    ax = plt.gca()
    ax.set_ylim([1.95,2.1])

    fig.set_title('Water Surface')
    fig.legend()

    fig = plt.subplot(5,1,2)
    plt.plot(x,Q)
    fig.set_title('Flow (m^2s-1)')

    fig = plt.subplot(5,1,3)
    plt.plot(x,U)
    fig.set_title('Velocity m/s')

    fig = plt.subplot(5,1,4)
    plt.plot(x,Qbed,label='Qbed')
    #plt.plot(x,Qb_star,label='Qb*')
    fig.set_title('Qbed')
    fig.legend()

    fig = plt.subplot(5,1,5)
    plt.plot(x,eta)
    fig.set_title('eta')
    plt.show()

    print U

    print 'OK'
if __name__ == "__main__":
    main_EULER()

