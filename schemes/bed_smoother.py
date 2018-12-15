#!/usr/bin/env python

import numpy as np
import math
from scipy.optimize import brentq,bisect,newton

curvature_threshold = 0.01


def get_stencil(ulist, start_index, end_index):
    '''
    Method to get the stencil - implements a periodic BC
    '''
    retval = None
    if end_index > len(ulist):
        a = ulist[start_index:]
        b = ulist[:end_index - len(ulist)]
        retval = np.concatenate((a, b), axis=0)
    elif start_index < 0:
        a = ulist[start_index:]
        # print('First {0}'.format(a))
        b = ulist[0:end_index]
        # print('Second: {0}'.format(b))
        retval = np.concatenate((a, b), axis=0)
    else:
        retval = ulist[start_index:end_index]

    return retval

def get_slope(x1,z1,x2,z2):
    slope = math.atan((z2-z1)/(x2-x1))
    return slope

def smooth_bed(zIntial, xc):
    zFinal = zIntial.copy()

    # Calculate the gradient
    dz_dx = np.gradient(zIntial, xc)
    d2z_dx2 = np.gradient(dz_dx, xc)

    for i in range(86):
        if d2z_dx2[i] > curvature_threshold:
            print('Threshold exceeded {0} and {1}'.format(d2z_dx2[i], dz_dx[i]))

            for j in range(3):
                zlocal = get_stencil(zFinal, i+j-1,i+j+3)
                xlocal = get_stencil(xc, i+j-1, i+j+3)
                print(zlocal,i+j)
                theta = get_slope(xlocal[0], zlocal[0], xlocal[1], zlocal[1])

                dz3 = zlocal[2]-zlocal[1]
                dz2 = zlocal[1]-zlocal[0]
                dx = xlocal[2]-xlocal[1]
                dz3_over = dz3 - dz2
                print(dz3_over)

                if dz3_over > 0.:


                    zFinal[i+j+1] = zFinal[i+j+1]-dz3_over
                    zFinal[i+j+2] = zFinal[i+j+2] + dz3_over

                    print('Updating {0} - was {1} now {2}'.format(i+j, zIntial[i+j],zFinal[i+j]))

            dz_dx = np.gradient(zFinal, xc)
            d2z_dx2 = np.gradient(dz_dx, xc)







    return dz_dx, d2z_dx2, zFinal
