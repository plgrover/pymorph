#!/usr/bin/env python
import numpy as np
import math
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