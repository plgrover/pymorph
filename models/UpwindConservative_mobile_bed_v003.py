#!/usr/bin/env python

'''
Notes

THis version includes correct Boundary Conditions and is used to examine
a more normal, none-dambreak type flow scneario

'''


import numpy as np
import matplotlib.pyplot as plt
import math
from copy import copy, deepcopy
from flux_solvers import *
from sed_trans import *
from bc_utilities import *

#--------------------------------------------------------------------------
# Global variables and constants
#-------------------------------------------------------------------------




def main_EULER():

    #--------------------------------------------------------------------------
    # Model run parameters - These are set by the user
    #--------------------------------------------------------------------------

    ''' General Settings '''
    plotting_update_interval = 10

    ''' Domain Settings '''
    number_nodes =101
    L = 100.0
    S = 1.0/300.0
    dx = L/(number_nodes-1)
    width = 1.0
    Zbed_left = 1.0

    ''' Hydrualic Parameters '''
    n = 0.027

    ''' Initial Conditions and Boundary Conditions '''
    h_init = 1.00
    Q_ic = 0.40
    Qin_BC = 0.40

    ''' Time Steps and Courant Number '''
    t_max = 1200.0
    Cn=0.8

    ''' Solver '''
    method = 'hll'

    ''' Sed Transport Settings '''
    sed_flag = False
    case = 'Louvain'
    D0 = 0.0035
    rho_particule=1540.0

    if case == 'Louvain':
        D0 = 0.0035
        rho_particule=1540.0
    elif case == 'Taipai':
        D0 = 0.0061
        rho_particule=1048.0
    else:
        Do = 0.001
        rho_particule=2450.0

    pm = get_bed_porosity(D0)
    Ycr = get_Ycr(D0,rho_particule)




    ''' Initialize the domain '''
    x = np.zeros(number_nodes)

    Zbed = np.ones(number_nodes)
    Z = np.zeros(number_nodes)
    Q = np.zeros(number_nodes)


    for i in range(len(x)):
        x[i] = i*dx
        Zbed[i] = Zbed_left - S*(dx*float(i))
        Z[i] = h_init + Zbed_left


    Zbed[48] = Zbed[48] + h_init*0.1
    Zbed[49] = Zbed[49] + h_init*0.15
    Zbed[50] = Zbed[50] + h_init*0.20
    Zbed[51] = Zbed[51] + h_init*0.15
    Zbed[52] = Zbed[52] + h_init*0.1

    A = (Z-Zbed)*width
    h = (Z-Zbed)


    ''' Apply Boundary conditions '''
    Q[:] = Q_ic
    Q[0] = Q_ic

    ''' Set the water elevations '''
    Z_init = deepcopy(Z)
    Zbed_init = deepcopy(Zbed)
    Q_init = deepcopy(Q)

    ''' Set the inital velocity'''
    U=Q/A

    dQbdx_final = None


    '''----------------------------------------------------------------------
        Set up the real-time plotting
       ----------------------------------------------------------------------
    '''

    fig = plt.figure()
    ax1 = fig.add_subplot(5,1,1)
    q_line, = ax1.plot(x,Q)
    ax1.set_ylim([0, Q_ic+0.25*Q_ic])
    ax1.set_title('Flow Rate')

    ax2 = plt.subplot(5,1,2)
    ws_line, = ax2.plot(x,Z)
    ax2.plot(x,Z_init,label='inital WS')
    ax2.plot(x,Zbed_init,label='inital Zbed')
    zbed_line, = ax2.plot(x,Zbed,label='Final Zbed')
    ax2.set_ylim([0.5,max(Z)])
    ax2.set_title('Water Surface')

    ax3 = plt.subplot(5,1,3)
    u_line, = ax3.plot(x,U)
    ax3.set_title('Velocity m/s')

    ax4 = plt.subplot(5,1,4)
    qbed_line, = ax4.plot(x,Qbed,label='Qbed')
    ax4.set_title('Bed Load')

    ax5 = plt.subplot(5,1,5)
    eta_line, = ax5.plot(x,eta)
    ax5.set_title('eta')


    plt.show(block=False)


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
            if cntr == plotting_update_interval:
                cntr=0
                max_q = max(Q)
                ax1.set_ylim([0,max_q + max_q*0.25])
                q_line.set_ydata(Q)

                max_z = max(Z)
                ax2.set_ylim([0,max_z + max_z*0.25])
                ws_line.set_ydata(Z)
                zbed_line.set_ydata(Zbed)

                max_u = max(U)
                if max_u < 0.1:
                    max_u = 0.1
                ax3.set_ylim([0,max_u + max_u*0.25])
                u_line.set_ydata(U)

                max_qbed = max(Qbed)
                ax4.set_ylim([0,max_qbed+0.25*max_qbed])
                qbed_line.set_ydata(Qbed)

                max_eta = max(eta)
                ax5.set_ylim([0,max_eta + 0.25*max_eta])
                eta_line.set_ydata(eta)
                fig.canvas.draw()


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

            for i in range(1,number_nodes-1):
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

                ''' Calculate the friction before updating the Area '''
                R = (A[i]/(2*Z[i] + width))
                Sf = G*(n**2.)*Q[i]*abs(Q[i])/(A[i]*R**(4./3.) )

                ''' Update the area '''
                A[i] = A_tmp[i] - dt*dAdx[i]


            ''' Update the depth'''
            Z = (A/width + Zbed)
            h = (Z-Zbed)
            U=Q/A

            ''' Update the boundary conditions '''
            ''' Inlet height - extrapolated from the next two interior nodes'''
            A[0] = linear_extrapolation(A[1],A[2])
            h[0] = A[0]/width
            Z[0] = h[0] + Zbed[0]

            ''' At the open outlfow boundary, both the discharge and water level
            are calculated by linear extrapolation'''
            A[number_nodes-1] = linear_extrapolation(A[number_nodes-2],A[number_nodes-3])
            h[number_nodes-1] = A[number_nodes-1]/width
            Z[number_nodes-1] = h[number_nodes-1] + Zbed[number_nodes-1]


            for i in range(1,number_nodes-1):
                ''' Calculate the slope based on the update surface '''
                w1 = get_w1(U,x,i,km[i],dt)
                w2 = get_w2(U,x,i,kp[i],dt)

                dzdx_down =get_delta_zdown(Z, x, i, km[i])
                dzdx_up = get_delta_zup(Z, x, i, kp[i])

                ''' Water slope '''
                S = -G*A[i]*(w1*dzdx_down + w2*dzdx_up) - Sf
                Q[i] = Q_tmp[i] - dt*dQdx[i] + dt*S

                #if i == number_nodes-2:
                    #print 'Qn+1: %s   Qn: %s  dQdx: %s   S: %s' % (Q[i],Q_tmp[i], dQdx[i], S )

                U[i]=Q[i]/A[i]



            '''----------------------------------------------------------------------'''
            ''' Update the boundary conditions'''
            '''----------------------------------------------------------------------'''
            ''' Inlet height - extrapolated from the next two interior nodes'''
            A[0] = linear_extrapolation(A[1],A[2])
            h[0] = A[0]/width
            Z[0] = h[0] + Zbed[0]

            Qbu[0] = linear_extrapolation(Qbu[1],Qbu[2])
            Qbed[0] = Qbu[0]/U[0]

            ''' At the open outlfow boundary, both the discharge and water level
            are calculated by linear extrapolation'''
            A[number_nodes-1] = linear_extrapolation(A[number_nodes-2],A[number_nodes-3])
            h[number_nodes-1] = A[number_nodes-1]/width
            Z[number_nodes-1] = h[number_nodes-1] + Zbed[number_nodes-1]
            Q[number_nodes-1] = linear_extrapolation(Q[number_nodes-2],Q[number_nodes-3])

            Qbu[number_nodes-1] = linear_extrapolation(Qbu[number_nodes-2],Qbu[number_nodes-3])
            Qbed[number_nodes-1] = Qbu[number_nodes-1]/U[number_nodes-1]
            #print 'Linear extrapolation at outlet: Q1: %s, Q2: %s and got %s' % (Q[number_nodes-2],Q[number_nodes-3],Q[number_nodes-1])
    except :
        import traceback
        print traceback.format_exc()

    Qb = np.zeros(number_nodes)
    ''' Calculate the final bed load '''
    for i in range(4,number_nodes-3):
        if U[i] != 0:
            Qb[i] = Qbu[i]/U[i]


    ax2.legend()
    plt.show()

    print '----------------------------------------------'
    print '   Q (m3/s)'
    print Q
    print '----------------------------------------------'

    print '----------------------------------------------'
    print '   Z (m)'
    print Z
    print '----------------------------------------------'

    print 'OK'




if __name__ == "__main__":
    main_EULER()

