#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import math
from copy import copy, deepcopy

#--------------------------------------------------------------------------
# Global variables and constants
#-------------------------------------------------------------------------

G = 9.81

def get_flux(A,Q,h):
    F = np.zeros(2)

    F[0] = Q
    F[1] = (Q**2.)/A + 0.5*G*(h*A)

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
    Fr = get_flux(ar,Qr,hr)
    Fl = get_flux(al,Ql,hl)

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

def get_hllc_TVD(al, ar, Ql, Qr, width, dt, dx):
    F = np.zeros(2)

    ''' Calculate basic variables '''
    ul = Ql/al
    ur = Qr/ar
    hl = al/width
    hr = ar/width
    al = math.sqrt(G*hl)
    ar = math.sqrt(G*hr)

    ''' Calculate the left and right fluxes '''
    Fr = get_flux(ar,Qr,hr)
    Fl = get_flux(al,Ql,hl)

    ''' Following Yan Gau "A Characteristic Finite Volume Scheme for SWE '''
    ''' EQ 23 '''

    hstar = (1/G)*((0.5*(math.sqrt(G*hl) + math.sqrt(G*hr)) + 0.25*(ul-ur)))**2.;

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

    ''' EQ 25 '''
    sstar = (sl*hr*(ur- sr) - sr*hl*(ul- SL))/(hr*(ur- sr) - hl*(ul-sl))
    
    ustarl = np.zeros(2)
    ustarl[0] = hl*(sl-ul)/(sl-sstar)
    ustarl[1] = ustarl[0]*sstar
    
    ustarr = np.zeros(2)
    ustarr[0] = hr*(sr-ur)/(SR-sstar)
    ustarr[1] = ustarr[0]*sstar
    
    fs = np.zeros(2) 
    fs[0] = (sr*fl[0] - sl*fr[0] + sr*sl*(hr-hl))/(sr-sl) 
    fs[1] = (sr*fl[1] - sl*fr[1] + sr*sl*(Qr-Ql))/(sr-sl) 
        
    cl = sl*(dt/dx) 
    cstar = sstar*(dt/dx) 
    cr = sr*(dt/dx) 
        
    r = 0 
    ak = np.zeros(3) 
    cs = [cl, cstar, cr] 
        
    hs = [hl, ustarl[0],ustarr[0],hr] 

    
    
    
    
    




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
    Cn=0.3
    width = 1.

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



    Flux_m = np.zeros(2)
    Flux_p = np.zeros(2)


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




        for i in range(3,number_nodes-2):
            Flux_m= 0
            km=0
            Flux_p =0
            kp = 0

            Flux_m,km  = get_hll(A[i-1], A[i], Q[i-1], Q[i],width)
            Flux_p,kp  = get_hll(A[i], A[i+1], Q[i], Q[i+1],width)

            dAdx[i] = (1./dx)*(Flux_p[0]-Flux_m[0])
            dQdx[i] = (1./dx)*(Flux_p[1]-Flux_m[1])

            ''' Calculate the friction before updating the Area '''
            R = (A_tmp[i]/(2*Z[i] + width))
            Sf = G*(n**2.)*Q[i]*abs(Q[i])/(A_tmp[i]*R**(4./3.) )

            ''' Update the area '''
            A[i] = A_tmp[i] - dt*dAdx[i]

        ''' Update the depth'''

        Z = A/width

        for i in range(4,number_nodes-3):
            Q[i] = Q_tmp[i] - dt*dQdx[i] + dt*Sf


    plt.subplot(2,1,1)
    plt.plot(x,Z)
    plt.subplot(2,1,2)
    plt.plot(x,Q)
    plt.show()

    print 'OK'
if __name__ == "__main__":
    main_EULER()




