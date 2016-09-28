#!/usr/bin/env python

import numpy as np
from math import sqrt
G = 9.81

def get_flux(h,q):
    F = np.zeros(2)

    F[0] = q
    F[1] = (q**2.)/h + 0.5*G*(h**2)

    return F




def get_hll(hl, hr,ql,qr):



    F = np.zeros(2)

    ''' Calculate basic variables '''
    ul = ql/hl
    ur = qr/hr
    al = sqrt(G*hl)
    ar = sqrt(G*hr)

    ''' Calculate the left and right fluxes '''
    Fr = get_flux(hr,qr)
    Fl = get_flux(hl,ql)

    hstar = 0.5*(hl + hr) - 0.25*(ur-ul)*(hl+hr)/(al+ar)

    ''' Calculate the eignvectors '''

    lambdal = 1.
    if hstar > hl:
        lambdal = sqrt(0.5*( (hstar + hl)*hstar/hl**2. ))

    lambdar = 1.
    if hstar > hr:
        lambdar = sqrt(0.5*( (hstar + hr)*hstar/hr**2. ))


    ''' Calculate the wave speeds'''
    sl = ul - al*lambdal
    sr = ur + ar*lambdar



    if sl >= 0:
        F = Fl
    elif sl <= 0 and sr >= 0:
        F[0] = (sr*Fl[0] - sl*Fr[0] + sr*sl*(hr-hl) ) / (sr-sl)
        F[1] = (sr*Fl[1] - sl*Fr[1] + sr*sl*(qr-ql) ) / (sr-sl)
    else:
        F = Fr

    return F

if __name__ == "__main__":
    F = np.zeros(2)
    F[0] = 50.9972
    F[1] = 282.7121
    FTest = get_hll(10, 1,0,0)

    print FTest, F


