#!/usr/bin/env python

'''
From
Ying, X., Khan, A.A. & Wang, S.S.Y., (2004): Upwind Conservative Scheme for the Saint Venant Equations. Journal of Hydraulic Engineering, 130(10), pp.977 988.

'''


import numpy as np
import matplotlib.pyplot as plt
import math
from copy import copy, deepcopy

#--------------------------------------------------------------------------
# Global variables and constants
#-------------------------------------------------------------------------

G = 9.81

def get_upwind_flux(Ai,Aip1,Qi,Qip1):
    F = np.zeros(2)
    k=0
    '''
    This is the flux at i + 1/2
    EQ 5
    '''
    if Qi >= 0:
        k = 0
        F[0] = Qi
        F[1] = (Qi**2.)/Ai
    else:
        k = 1
        F[0] = Qip1
        F[1] = (Qip1**2.)/Aip1

    return F, k



def get_delta_zdown(Z, x, i, k):
    if k == 0.5:
        z1 = 0.5*(Z[i] + Z[i+1])
        x1 = 0.5*(x[i] + x[i+1])

        z2 = 0.5*(Z[i] + Z[i-1])
        x2 = 0.5*(x[i] + x[i-1])
    else:
        z1 = Z[i+1-k]
        x1 = x[i+1-k]

        z2 = Z[i-k]
        x2 = x[i-k]

    return (z1 - z2)/(x1-x2)

def get_delta_zup(Z, x, i, k):
    if k == 0.5:
        z1 = 0.5*(Z[i] + Z[i+1])
        x1 = 0.5*(x[i] + x[i+1])

        z2 = 0.5*(Z[i] + Z[i-1])
        x2 = 0.5*(x[i] + x[i-1])
    else:
        z1 = Z[i+k]
        x1 = x[i+k]

        z2 = Z[i-1+k]
        x2 = x[i-1+k]

    return (z1 - z2)/(x1-x2)

def get_w2(U,x,i,k,dt):
    ''' w1 '''
    if k == 0.5:
        u1 = abs(0.5*(U[i] + U[i+1]))
        x1 = 0.5*(x[i] + x[i+1])

        u2 = abs(0.5*(U[i] + U[i-1]))
        x2 = 0.5*(x[i] + x[i-1])
    else:
        u1 = abs(U[i+k])
        x1 = x[i+k]

        u2 = abs(U[i-1+k])
        x2 = x[i-1+k]

    w2 = 0.5*dt*(u1+u2)/(x1-x2)

    return w2

def get_w1(U,x,i,k,dt):
    ''' w1 '''
    if k == 0.5:
        u1 = abs(0.5*(U[i] + U[i+1]))
        x1 = 0.5*(x[i] + x[i+1])

        u2 = abs(0.5*(U[i] + U[i-1]))
        x2 = 0.5*(x[i] + x[i-1])
    else:
        u1 = abs(U[i+1-k])
        x1 = x[i+1-k]

        u2 = abs(U[i-k])
        x2 = x[i-k]

    w1 = 1. - 0.5*dt*(u1+u2)/(x1-x2)

    return w1

def get_flux(a,q):
    F = np.zeros(2)

    F[0] = q
    F[1] = (q**2)/a

    return F


def get_hll(al, ar, Ql, Qr, width):
    F = np.zeros(2)

    ''' Calculate basic variables '''
    ul = Ql/al
    ur = Qr/ar
    hl = al/width
    hr = ar/width
    al = math.sqrt(G*hl)
    ar = math.sqrt(G*hr)

    ''' Calculate the left and right fluxes '''
    Fr = get_flux(ar,Qr)
    Fl = get_flux(al,Ql)

    hstar = 0.5*(hl + hr) - 0.25*(ur-ul)*(hl+hr)/(al+ar)

    ''' Calculate the eignvectors '''

    lambdal = 1.
    if hstar > hl:
        lambdal = math.sqrt(0.5*( (hstar + hl)*hstar/hl**2. ))

    lambdar = 1.
    if hstar > hr:
        lambdar = math.sqrt(0.5*( (hstar + hr)*hstar/hr**2. ))


    ''' Calculate the wave speeds'''
    sl = ul - al*lambdal
    sr = ur + ar*lambdar

    if sl >= 0:
        F = Fl
    elif sl <= 0 and sr >= 0:
        F[0] = (sr*Fl[0] - sl*Fr[0] + sr*sl*(ar-al) ) / (sr-sl)
        F[1] = (sr*Fl[1] - sl*Fr[1] + sr*sl*(Qr-Ql) ) / (sr-sl)
    else:
        F = Fr

    k = 0.5
    if Ql > 0 and Qr > 0: # if moving towards the right, go upstream
        k = 0
    elif Ql < 0 and Qr < 0: # if moving towads the left, go downstream
        k=1


    return F,k



def main_EULER():

    #--------------------------------------------------------------------------
    # Model run parameters - These are set by the user
    #--------------------------------------------------------------------------

    # Number of nodes
    number_nodes =401

    # Domain Length
    L = 2.
    centre_cell = int(math.floor(number_nodes/2))
    dx = L/(number_nodes-1)
    # Initial epth settings
    depth_l = 0.1
    depth_r = 0.001
    S0 = 0.
    n = 0.0
    # Courant Number
    Cn=0.1
    width = 1.

    method= 'hll'

    # max time
    t0 = math.sqrt(depth_l/G)
    t_max = 3.*t0

    ''' Initialize the domain '''
    x = np.zeros(number_nodes)
    for i in range(len(x)):
        x[i] = i*dx

    A = np.zeros(number_nodes)
    Q = np.zeros(number_nodes)

    ''' Set the water elevations '''
    A[0:centre_cell+1] = depth_l * width
    A[centre_cell:] = depth_r * width


    A_init = deepcopy(A)
    Q_init = deepcopy(Q)
    Z = A/width
    Z_init = A/width




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

        A_tmp = deepcopy(A)
        Q_tmp = deepcopy(Q)
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

        Z = A/width

        for i in range(4,number_nodes-3):
            ''' Calculate the slope based on the update surface '''
            w1 = get_w1(U,x,i,km[i],dt)
            w2 = get_w2(U,x,i,kp[i],dt)

            dzdx_down =get_delta_zdown(Z, x, i, km[i])
            dzdx_up = get_delta_zup(Z, x, i, kp[i])

            S = -G*A[i]*(w1*dzdx_down + w2*dzdx_up) + Sf


            Q[i] = Q_tmp[i] - dt*dQdx[i] + dt*S

    plt.subplot(2,1,1)
    plt.plot(x,Z)
    plt.subplot(2,1,2)
    plt.plot(x,Q)
    plt.show()

    print 'OK'
if __name__ == "__main__":
    main_EULER()



    #------------------------------------------------------------------- def main():
#------------------------------------------------------------------------------
    # #--------------------------------------------------------------------------
    #------------------------ # Model run parameters - These are set by the user
    # #--------------------------------------------------------------------------
#------------------------------------------------------------------------------
    #--------------------------------------------------------- # Number of nodes
    #--------------------------------------------------------- number_nodes =100
    #---------------------------------------------------------------- # max time
    #---------------------------------------------------------------- t_max = 10
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
    #----------------------------------------------------------- # Domain Length
    #------------------------------------------------------------------ L = 1000
    #----------------------------- centre_cell = int(math.floor(number_nodes/2))
    #--------------------------------------------------- dx = L/(number_nodes-1)
    #--------------------------------------------------- # Initial epth settings
    #------------------------------------------------------------- depth_l = 10.
    #-------------------------------------------------------------- depth_r = 1.
    #------------------------------------------------------------------- S0 = 0.
    #------------------------------------------------------------------- n = 0.0
    #---------------------------------------------------------- # Courant Number
    #-------------------------------------------------------------------- Cn=0.2
    #---------------------------------------------------------------- width = 1.
#------------------------------------------------------------------------------
    #--------------------------------------------- ''' Initialize the domain '''
    #------------------------------------------------ x = np.zeros(number_nodes)
    #--------------------------------------------------- for i in range(len(x)):
        #----------------------------------------------------------- x[i] = i*dx
#------------------------------------------------------------------------------
    #------------------------------------------------ A = np.zeros(number_nodes)
    #------------------------------------------------ Q = np.zeros(number_nodes)
#------------------------------------------------------------------------------
    #------------------------------------------ ''' Set the water elevations '''
    #-------------------------------------- A[0:centre_cell+1] = depth_l * width
    #----------------------------------------- A[centre_cell:] = depth_r * width
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
    #------------------------------------------------------- A_int = deepcopy(A)
    #------------------------------------------------------- Q_int = deepcopy(Q)
    #--------------------------------------------------------------- Z = A/width
    #---------------------------------------------------------- Z_init = A/width
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
    #---------------------- ''' ---------------------------------------------'''
    #------------------------------------------- ''' Starting main loop here '''
    #---------------------- ''' ---------------------------------------------'''
    #--------------------------------------------------------------------- t = 0
    #------------------------------------------------------------------ cntr = 0
    #---------------------------------------------------------- while t < t_max:
        #---------------------------------- ''' calculate the courant number '''
        #--------------------- dt = Cn * dx / max(abs(Q/A) + np.sqrt(G*A/width))
#------------------------------------------------------------------------------
        #------------------------------------------------------------- cntr += 1
        #--------------------------------------------------------------- t += dt
        #------------------------------------------------------------ print t,dt
#------------------------------------------------------------------------------
        #------------------------------------- dAdx = np.zeros((number_nodes,2))
        #------------------------------------- dQdx = np.zeros((number_nodes,2))
#------------------------------------------------------------------------------
        #--------------------------------------------------- A_tmp = deepcopy(A)
        #--------------------------------------------------- Q_tmp = deepcopy(Q)
#------------------------------------------------------------------------------
        #------------------------------------------------ for stage in range(2):
            #--------------------------------- for i in range(3,number_nodes-2):
#------------------------------------------------------------------------------
                #------- Flux_m, km  = get_hll(A[i-1], A[i], Q[i-1], Q[i],width)
                #------- Flux_p, kp  = get_hll(A[i], A[i+1], Q[i], Q[i+1],width)
#------------------------------------------------------------------------------
                #----------------- dAdx[i,stage] = (1./dx)*(Flux_p[0]-Flux_m[0])
#------------------------------------------------------------------------------
            #----------------------------------------------------------- U = Q/A
#------------------------------------------------------------------------------
            #--------------------------------- for i in range(3,number_nodes-2):
                #------- ''' Calculate the friction before updating the Area '''
                #----------------------------------- R = (A[i]/(2*Z[i] + width))
                #-------------- Sf = G*(n**2.)*Q[i]*abs(Q[i])/(A[i]*R**(4./3.) )
#------------------------------------------------------------------------------
                #--------------------------------------- ''' Update the area '''
                #------------------------------------------------ if stage == 0:
                    #---------------------------- A[i] = A_tmp[i] - dt*dAdx[i,0]
                #--------------------------------------------------------- else:
                    #-------- A[i] = A_tmp[i] - 0.5 * dt*(dAdx[i,0] + dAdx[i,1])
#------------------------------------------------------------------------------
            #------------------------------------------- ''' Update the depth'''
#------------------------------------------------------------------------------
            #------------------------------------------------------- Z = A/width
#------------------------------------------------------------------------------
            #--------------------------------- for i in range(3,number_nodes-2):
                #------- ''' Calculate the slope based on the update surface '''
                #-------------------------------------- w1 = get_w1(U,x,i,km,dt)
                #-------------------------------------- w2 = get_w2(U,x,i,kp,dt)
#------------------------------------------------------------------------------
                #----------------------- dzdx_down =get_delta_zdown(Z, x, i, km)
                #------------------------- dzdx_up =get_delta_zdown(Z, x, i, kp)
#------------------------------------------------------------------------------
                #------------------ S = -G*A[i]*(w1*dzdx_down + w2*dzdx_up) - Sf
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
                #---------- dQdx[i,stage] = (1./dx)*(Flux_p[1]-Flux_m[1]) - dt*S
#------------------------------------------------------------------------------
                #------------------------------------------------ if stage == 0:
                    #---------------------------- Q[i] = Q_tmp[i] - dt*dQdx[i,0]
                #--------------------------------------------------------- else:
                    #-------- Q[i] = Q_tmp[i] - 0.5 * dt*(dQdx[i,0] + dQdx[i,1])
#------------------------------------------------------------------------------
    #---------------------------------------------------- plt.plot(x,Z,x,Z_init)
    #---------------------------------------------------------------- plt.show()
    #------------------------------------------------------------------- print A
#------------------------------------------------------------------------------
    #---------------------------------------------------------------- print 'OK'