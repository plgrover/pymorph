#!/usr/bin/env python

'''
This contains code and functions to assist in setting boundary conditions
'''

def linear_extrapolation(y_1,y_2,x_1=1.0,x_2=2.0,x_0=0.0):
    y = y_1 - (y_2-y_1)*(x_1-x_0)/(x_2-x_1)
    return y


if __name__ == "__main__":
    y0 = linear_extrapolation(6,4)
    print ('Test on linear extrapolation: Expected 8 got %s' % y0)

    y0 = linear_extrapolation(6,8)
    print ('Test on linear extrapolation: Expected 4 got %s' % y0)