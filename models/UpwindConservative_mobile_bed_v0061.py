#!/usr/bin/env python

'''
Notes

This is based on version 3 but have moved the sediment tranport into a
separate class.

Seems to be a problem with the hll solver for the bump. Water surface is incorrect
The simple upwind approach does seem to work well here.

'''


import numpy as np
import matplotlib.pyplot as plt
import math
from copy import copy, deepcopy
from flux_solvers import *
from sed_trans import *
from sed_trans_models import *
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
    number_nodes =201
    L = 100.0
    S = 1.0/300.0
    dx = L/(number_nodes-1)
    width = 1.0
    Zbed_left = 1.0

    ''' Hydrualic Parameters '''
    n = 0.035

    ''' Initial Conditions and Boundary Conditions '''
    h_init = 2.00
    Q_ic = 0.05
    Qin_BC = 1.40
    tramp = 60.0

    ''' Time Steps and Courant Number '''
    t_max = 400.0
    Cn=0.8

    ''' Solver '''
    method = ''

    ''' Sed Transport Settings '''

    sed_flag = False
    case = 'Louvain'
    D50 = 0.0035
    rho_sediment=1540.0
    rho_water = 1000.0

    if case == 'Louvain':
        D50 = 0.0035
        rho_sediment=1540.0
    elif case == 'Taipai':
        D50 = 0.0061
        rho_sediment=1048.0
    else:
        D50 = 0.001
        rho_sediment=2450.0
    #sed_model = equlibrium_sed_transport_model(rho_sediment, D50)
    sed_model = quasi_equlibrium_sed_transport_model(rho_sediment, D50)
    ''' Somethine is wrong with the non-equilibrium model '''
    #sed_model = non_equlibrium_sed_transport_model(rho_sediment, D50)
    #sed_model = null_sed_transport_model(rho_sediment, D50)
    sed_model = non_equlibrium_sed_transport_model(rho_sediment, D50)
    apply_sediment_at_time = 60.0



    ''' Initialize the domain '''
    x = np.zeros(number_nodes)

    Zbed = np.ones(number_nodes)
    Z = np.zeros(number_nodes)
    Q = np.zeros(number_nodes)
    U = np.zeros(number_nodes)

    A0 = Zbed_left
    A1 = 0.5
    lam = 0.1*L

    for i in range(len(x)):
        x[i] = i*dx

        if x[i] >= 40.0 and x[i] <= 45:
            Zbed[i] =A0 + A1*math.sin(2.0*math.pi*x[i]/lam)- S*(dx*float(i))
        else:
            Zbed[i] = Zbed_left - S*(dx*float(i))
        Z[i] = h_init + Zbed_left

    A = (Z-Zbed)*width

    if 1==1:
        fig = plt.figure()
        ax1 = fig.add_subplot(3,1,1)
        q_line, = ax1.plot(x,Q)
        ax1.set_ylim([-0.1, Q_ic+0.25*Q_ic])
        ax1.set_title('Flow Rate')

        ax1 = fig.add_subplot(3,1,2)
        ax1.plot(x,Z, label = 'Water Surface')
        ax1.plot(x,Zbed, label = 'Bed Elevation')
        ax1.set_ylim([-0.1, 6.0])
        ax1.set_title('Water and Bed Elevation')

        ax1 = fig.add_subplot(3,1,3)
        ax1.plot(x,Z-Zbed, label = 'Water Depth')
        ax1.set_title('Water Depth')
        plt.show()

    ''' Apply Boundary conditions '''
    Q[:] = Q_ic
    Q[0] = Q_ic

    ''' Set the water elevations '''
    Z_init = deepcopy(Z)
    Zbed_init = deepcopy(Zbed)
    Q_init = deepcopy(Q)

    sed_model.set_flow_properties(Q,A,Zbed,width,x,dx)
    sed_model_update_interval = 50


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
    ws_line, = ax2.plot(x,Z, label='WS')
    ax2.plot(x,Z_init,label='inital WS')
    ax2.plot(x,Zbed_init,label='inital Zbed')
    zbed_line, = ax2.plot(x,Zbed,label='Final Zbed')
    ax2.set_ylim([0.5,max(Z)])
    ax2.set_title('Water Surface')

    ax3 = plt.subplot(5,1,3)
    u_line, = ax3.plot(x,Q/A)
    ax3.set_title('Velocity')

    ax4 = plt.subplot(5,1,4)
    qbed_line, = ax4.plot(x,sed_model.get_Qbed(),label='Qb')
    qbstar_line, = ax4.plot(x,sed_model.get_Qb_star(),label='Qb*')
    ax4.set_title('Bed Load')

    ax5 = plt.subplot(5,1,5)
    eta_line, = ax5.plot(x,sed_model.get_Eta())
    ax5.set_title('eta')


    plt.show(block=False)


    ''' ---------------------------------------------'''
    ''' Starting main loop here '''
    ''' ---------------------------------------------'''
    t = 0
    cntr = 0
    sed_cntr = 0
    dt_sed = 0
    try:
        while t < t_max:

            if t < tramp:
                Q[0] = 0.1 + Qin_BC*(t/tramp)
            else:
                Q[0] = Qin_BC

            ''' Step 1: Calculate the courant number '''
            dt = Cn * dx / max(abs(Q/A) + np.sqrt(G*A/width))

            ''' Step 2: Set up the temporary variables '''
            t += dt
            print t,dt

            dAdx = np.zeros(number_nodes)
            dQdx = np.zeros(number_nodes)
            dQbdx = np.zeros(number_nodes)

            A_tmp = deepcopy(A)
            Q_tmp = deepcopy(Q)
            Zbed_tmp = deepcopy(Zbed)
            km = np.zeros(number_nodes)
            kp = np.zeros(number_nodes)

            bed_exchange = sed_model.get_sed_exchange_terms()
            qbed = sed_model.get_Qbed()
            for i in range(1,number_nodes-1):
                Flux_m=  np.zeros(2)
                Flux_p = np.zeros(2)

                if method == 'hll':
                    Flux_m,km[i]  = get_hll(A_tmp[i-1], A_tmp[i], Q[i-1], Q[i],width)
                    Flux_p,kp[i]  = get_hll(A_tmp[i], A_tmp[i+1], Q[i], Q[i+1],width)
                else:
                    Flux_m,km[i]  = get_upwind_flux(A_tmp[i-1], A_tmp[i], Q[i-1], Q[i])
                    Flux_p,kp[i]  = get_upwind_flux(A_tmp[i], A_tmp[i+1], Q[i], Q[i+1])

                dAdx[i] = (1./dx)*(Flux_p[0]-Flux_m[0])
                dQdx[i] = (1./dx)*(Flux_p[1]-Flux_m[1])

                ''' Calculate the friction before updating the Area '''
                R = (A[i]/(2*A[i]/width + width))
                Sf = G*(n**2.)*Q[i]*abs(Q[i])/(A[i]*R**(4./3.) )

                ''' Update the area '''
                A[i] = A_tmp[i] - dt*dAdx[i] + (1.0/(1.0 - sed_model.get_pm()))*bed_exchange[i]

            ''' Update the depth'''
            Z = (A/width + Zbed)


            ''' Update the boundary conditions '''
            ''' Inlet height - extrapolated from the next two interior nodes'''
            A[0] = linear_extrapolation(A[1],A[2])
            Z[0] = A[0]/width + Zbed[0]

            ''' At the open outlfow boundary, both the discharge and water level
            are calculated by linear extrapolation'''
            A[number_nodes-1] = linear_extrapolation(A[number_nodes-2],A[number_nodes-3])
            Z[number_nodes-1] = A[number_nodes-1]/width + Zbed[number_nodes-1]


            for i in range(1,number_nodes-1):
                ''' Calculate the slope based on the update surface '''
                w1 = get_w1(Q/A,x,i,km[i],dt)
                w2 = get_w2(Q/A,x,i,kp[i],dt)

                dzdx_down =get_delta_zdown(Z, x, i, km[i])
                dzdx_up = get_delta_zup(Z, x, i, kp[i])

                ''' Water and friction slope '''
                S = -G*A[i]*(w1*dzdx_down + w2*dzdx_up) - Sf
                ''' Bed Transport term '''
                Bed_Transport= ( (rho_sediment-rho_water)/(rho_water) )*Q[i]/A[i]*( 1 -(qbed[i]/(1.0 - sed_model.get_pm()) ) )*bed_exchange[i]


                Q[i] = Q_tmp[i] - dt*dQdx[i] + dt*S -Bed_Transport


            '''----------------------------------------------------------------------'''
            ''' Update the boundary conditions'''
            '''----------------------------------------------------------------------'''
            ''' Inlet height - extrapolated from the next two interior nodes'''
            A[0] = linear_extrapolation(A[1],A[2])
            Z[0] = A[0]/width + Zbed[0]


            ''' At the open outlfow boundary, both the discharge and water level
            are calculated by linear extrapolation'''

            A[number_nodes-1] = linear_extrapolation(A[number_nodes-2],A[number_nodes-3])
            Z[number_nodes-1] = A[number_nodes-1]/width + Zbed[number_nodes-1]
            Q[number_nodes-1] = linear_extrapolation(Q[number_nodes-2],Q[number_nodes-3])

            if t > apply_sediment_at_time:
                if sed_cntr == sed_model_update_interval:
                    print 'Updating the sediment model'
                    Zbed = sed_model.update(dt_sed, Q, A, Zbed)
                    h = Z-Zbed
                    A = h*width
                    dt_sed = 0
                    sed_cntr = 0
                else:
                    sed_cntr += 1
                    dt_sed += dt

            A[number_nodes-1] = linear_extrapolation(A[number_nodes-2],A[number_nodes-3])
            Z[number_nodes-1] = A[number_nodes-1]/width + Zbed[number_nodes-1]
            Q[number_nodes-1] = linear_extrapolation(Q[number_nodes-2],Q[number_nodes-3])


            ''' Step 2: Update the plots '''
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

                max_u = max(Q/A)
                if max_u < 0.1:
                    max_u = 0.1
                ax3.set_ylim([0,max_u + max_u*0.25])
                u_line.set_ydata(Q/A)

                max_qbed = max(sed_model.get_Qb_star())
                ax4.set_ylim([0,max_qbed+0.25*max_qbed])
                qbed_line.set_ydata(sed_model.get_Qbed())
                qbstar_line.set_ydata(sed_model.get_Qb_star())


                max_eta = max(sed_model.get_Eta())
                ax5.set_ylim([0,max_eta + 0.25*max_eta])
                eta_line.set_ydata(sed_model.get_Eta())
                fig.canvas.draw()


            #print 'Linear extrapolation at outlet: Q1: %s, Q2: %s and got %s' % (Q[number_nodes-2],Q[number_nodes-3],Q[number_nodes-1])
    except :
        import traceback
        print traceback.format_exc()


    ax2.legend()
    ax4.legend()
    plt.show()

    np.savetxt('Qb.csv', sed_model.get_Qbed(), delimiter=',')
    np.savetxt('Qb_star.csv', sed_model.get_Qb_star(), delimiter=',')
    np.savetxt('Q.csv', Q, delimiter=',')
    np.savetxt('A.csv', A, delimiter=',')
    np.savetxt('Z.csv', Z, delimiter=',')
    np.savetxt('Zbed.csv', Zbed, delimiter=',')

    print 'OK'




if __name__ == "__main__":
    main_EULER()

