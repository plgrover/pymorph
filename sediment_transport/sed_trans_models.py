#!/usr/bin/env python
import numpy as np
import math
from copy import copy, deepcopy
from sed_trans import *
from bc_utilities import *

class null_sed_transport_model(object):
    def __init__(self,rho_sediment, D50, sed_model='mpm', D90=0.0):
        pass

    def set_flow_properties(self, Q, A, Zbed, width, x, dx):
        self.number_nodes = len(x)


    def get_Qbed(self):
        return np.zeros(self.number_nodes)

    def get_Eta(self):
        return np.zeros(self.number_nodes)

    def get_sed_exchange_terms(self):
        ''' The sediment exchange terms are given on the LHS of eq 9.68'''
        return np.zeros(self.number_nodes)

    def get_pm(self):
        return 0

    def update(self, dt):
        pass


class equlibrium_sed_transport_model(null_sed_transport_model):
    def __init__(self,rho_sediment, D50, sed_model='mpm', D90=0.0):
        self.rho_sediment = rho_sediment
        self.D50 = D50
        self.sed_model =sed_model
        if D90 != 0.0:
            self.D90 = D90
        else:
            self.D90 = D50

        self.pm = get_bed_porosity(D50)
        self.Ycr = get_Ycr(D50,rho_sediment)

    def set_flow_properties(self, Q, A, Zbed, width, x, dx):
        self.Q = Q
        self.A = A
        self.Zbed = Zbed
        self.Zbed_init = deepcopy(Zbed)
        self.width = width
        self.x = x
        self.dx = dx
        self.number_nodes = len(x)
        self.qbed = np.zeros(self.number_nodes)
        self.eta = np.zeros(self.number_nodes)


    def get_Qbed(self):
        return self.qbed * self.width

    def get_Eta(self):
        return self.eta

    def get_sed_exchange_terms(self):
        ''' The sediment exchange terms are given on the LHS of eq 9.68'''
        return np.zeros(self.number_nodes)

    def get_pm(self):
        return self.pm

    def update(self, dt):
        '''
        To assist with the following notation see Figure 4.18 on page 146 of Wu 2007
        '''
        Zbed_tmp = deepcopy(self.Zbed)
        for i in range(1,self.number_nodes-1):
            hW = self.A[i-1]/self.width
            hP = self.A[i]/self.width
            hE = self.A[i+1]/self.width

            uW = self.Q[i-1]/self.A[i-1]
            uP = self.Q[i]/self.A[i]
            uE = self.Q[i+1]/self.A[i+1]

            if i==1 or i== self.number_nodes-2:
                Flux_Qbed_in = get_upwind_bedload_flux(hW, hP, uW, uP, self.D50, self.rho_sediment, self.sed_model)*self.width
                Flux_Qbed_out = get_upwind_bedload_flux(hP, hE, uP, uE, self.D50, self.rho_sediment, self.sed_model)*self.width
            else:
                hWW = self.A[i-2]/self.width
                hEE = self.A[i+2]/self.width

                uWW = self.Q[i-2]/self.A[i-1]
                uEE = self.Q[i+2]/self.A[i+1]

                Flux_Qbed_in = get_quick_bedload_flux(hWW, hW, hP, hE, uWW, uW, uP, uE, self.D50, self.rho_sediment, self.sed_model)*self.width
                Flux_Qbed_out = get_quick_bedload_flux(hW, hP, hE, hEE, uW, uP, uE, uEE, self.D50, self.rho_sediment, self.sed_model)*self.width


            ''' Abed = Zbed*width*dx, therefore dAbed = (Zbed_new-Zbed_old)*width*dx'''
            self.Zbed[i] = Zbed_tmp[i] - (dt/(self.dx*self.width))*(1./(1-self.pm))*(Flux_Qbed_out-Flux_Qbed_in)

            ''' Not necessary but calculate the bedload '''
            self.qbed[i] = get_unit_bed_load(hP,uP,self.D50, self.rho_sediment, self.sed_model)*self.width
            self.eta[i] = get_Y(hP,uP,self.D50, self.rho_sediment)/self.Ycr


        self.update_boundary_conditions()

    def update_boundary_conditions(self):
        self.qbed[0] = linear_extrapolation(self.qbed[1],self.qbed[2])
        #self.Zbed[0] = linear_extrapolation(self.qbed[1],self.qbed[2])
        self.qbed[self.number_nodes-1] = linear_extrapolation(self.qbed[self.number_nodes-2],self.qbed[self.number_nodes-3])
        self.Zbed[self.number_nodes-1] = linear_extrapolation(self.Zbed[self.number_nodes-2],self.Zbed[self.number_nodes-3])


class non_equlibrium_sed_transport_model(equlibrium_sed_transport_model):
    def set_flow_properties(self, Q, A, Zbed, width, x, dx):
        self.Q = Q
        self.A = A
        self.Zbed = Zbed
        self.width = width
        self.x = x
        self.dx = dx
        self.number_nodes = len(x)
        self.qbed = np.zeros(self.number_nodes)
        self.eta = np.zeros(self.number_nodes)
        self.Qbu = np.zeros(self.number_nodes)
        self.Qb_star = np.zeros(self.number_nodes)
        self.Zbed_init = deepcopy(Zbed)

        self.L = 6.0*A/width

    def get_sed_exchange_terms(self):
        ''' The sediment exchange terms are given on the LHS of eq 9.68'''
        retval = np.zeros(self.number_nodes)
        for i in range(self.number_nodes):
            retval[i] = (1./self.L[i]) * (self.Qb_star[i] - self.qbed[i])
        return retval

    def get_Qb_star(self):
        return self.Qb_star

    def update(self, dt, Q, A, Zbed):
        Qbu_tmp = deepcopy(self.Qbu)
        dQbdx= np.zeros(self.number_nodes)
        U = self.Q/self.A
        h = self.A/self.width
        Zbed_tmp = deepcopy(self.Zbed)
        qbed_tmp = deepcopy(self.qbed)

        for i in range(1,self.number_nodes-1):

            ''' Calculate the sediment transport fluxes '''
            Flux_Qsb_in = get_upwind_bedload_flux_neq(qbed_tmp[i-1], qbed_tmp[i],self.Q[i])
            Flux_Qsb_out = get_upwind_bedload_flux_neq(qbed_tmp[i], qbed_tmp[i+1], self.Q[i])
            dQbdx[i] = (1./self.dx)*(Flux_Qsb_out-Flux_Qsb_in)

            self.Qb_star[i] = get_unit_bed_load(h[i],U[i],self.D50,self.rho_sediment)
            self.L[i] = 0.6

            self.Qbu[i] = Qbu_tmp[i] - (dt)*dQbdx[i] + (dt/self.L[i])*(self.Qb_star[i] - qbed_tmp[i])

            ''' Calculate ub '''
            ubed = get_Ubed(U[i], h[i], self.D50,self.rho_sediment)
            self.qbed[i] = self.Qbu[i]*ubed

            self.Zbed[i] = Zbed_tmp[i] - 100.0*(dt/self.dx)*(1./(1-self.pm))*(1./self.L[i])*(self.Qb_star[i] - self.qbed[i])
            self.eta[i] = get_Y(h[i],U[i],self.D50, self.rho_sediment)/self.Ycr

        print 'Max delta z %s ' % max(abs(self.Zbed_init - self.Zbed))
        print 'Max Qbu %s ' % max(self.Qbu)

        self.update_boundary_conditions()
        print 'Max delta z %s ' % max(abs(self.Zbed_init - self.Zbed))
        return Zbed

class quasi_equlibrium_sed_transport_model(equlibrium_sed_transport_model):
    def set_flow_properties(self, Q, A, Zbed, width, x, dx):
        self.Q = Q
        self.A = A
        self.Zbed = Zbed
        self.width = width
        self.x = x
        self.dx = dx
        self.number_nodes = len(x)
        self.qbed = np.zeros(self.number_nodes)
        self.eta = np.zeros(self.number_nodes)
        self.Qbu = np.zeros(self.number_nodes)
        self.Qb_star = np.zeros(self.number_nodes)
        self.Zbed_init = deepcopy(Zbed)

        self.L = 6.0*A/width

    def get_sed_exchange_terms(self):
        ''' The sediment exchange terms are given on the LHS of eq 9.68'''
        retval = np.zeros(self.number_nodes)
        #for i in range(self.number_nodes):
        #    retval[i] = (1./self.L[i]) * (self.Qb_star[i] - self.qbed[i])
        return retval

    def get_Qb_star(self):
        return self.Qb_star

    def update(self, dt, Q, A, Zbed):
        qbed_tmp = deepcopy(self.qbed)
        dQbdx= np.zeros(self.number_nodes)
        U = Q/A
        h = A/self.width
        Zbed_tmp = deepcopy(Zbed)
        self.L = 6.0 * np.mean(h)

        for i in range(self.number_nodes):
            self.Qb_star[i] = get_unit_bed_load(h[i],U[i],self.D50,self.rho_sediment)*self.width

        for it in range(200):
            for i in range(1,self.number_nodes-1):
                ''' Using Eq. 5.131 on page 210 to calculate the bedload'''
                if self.Q[i] >= 0.0:
                    self.qbed[i] = qbed_tmp[i-1] + (self.dx/self.L)*(self.Qb_star[i] - qbed_tmp[i-1])
                else:
                    self.qbed[i] = qbed_tmp[i+1] - (self.dx/self.L)*(self.Qb_star[i] - qbed_tmp[i+1])

        for i in range(1,self.number_nodes-1):
            if self.Q[i] >= 0.0:
                Zbed[i] = Zbed_tmp[i] - 100.0*(dt/(self.width*self.dx*(1-self.pm)))*(1./self.L)*(self.Qb_star[i] - self.qbed[i-1])
            else:
                Zbed[i] = Zbed_tmp[i] - (dt/(self.width*self.dx*(1-self.pm)))*(1./self.L)*(self.Qb_star[i] - self.qbed[i+1])
            self.eta[i] = get_Y(h[i],U[i],self.D50, self.rho_sediment)/self.Ycr

        self.update_boundary_conditions()
        self.qbed[0] = 0.10*get_unit_bed_load(h[i],U[i],self.D50,self.rho_sediment)*self.width
        Zbed[0] = linear_extrapolation(Zbed[1],Zbed[2])
        Zbed[self.number_nodes-1] = linear_extrapolation(Zbed[self.number_nodes-2],Zbed[self.number_nodes-3])

        print 'Max delta z %s ' % max(abs(self.Zbed_init - self.Zbed))
        return Zbed

    def update_boundary_conditions(self):

        #self.Zbed[0] = linear_extrapolation(self.qbed[1],self.qbed[2])
        self.qbed[self.number_nodes-1] = linear_extrapolation(self.qbed[self.number_nodes-2],self.qbed[self.number_nodes-3])

