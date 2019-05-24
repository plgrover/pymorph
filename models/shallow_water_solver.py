from clawpack import riemann
#import clawpack.petclaw as pyclaw

from clawpack import pyclaw
import numpy as np
import math
from scipy.sparse.linalg.isolve._iterative import zbicgrevcom
# http://www.clawpack.org/pyclaw/parallel.html


# http://www.clawpack.org/bc.html
# https://groups.google.com/forum/#!searchin/claw-users/dirichlet%7Csort:date/claw-users/dg7SzX6keqM/1nanvItqBwAJ

def inlet_BC(state,dim,t,qbc,auxbc,num_ghost):
    qIn = state.problem_data['lower_bc_data']
    # The num_ghost + 1 I think is wrong but seems to stablize the model?
    qbc[0, :num_ghost] =qbc[0, num_ghost+1]
    qbc[1,:num_ghost] = qIn
    
    
def outlet_BC(state,dim,t,qbc,auxbc,num_ghost):
    sOut = state.problem_data['upper_bc_data']
    n = len(state.grid.x.centers)
    hOut = sOut - state.aux[0, n-1]
    # print(sOut, hOut, qbc[0, -num_ghost:] )
    qbc[0, -num_ghost:] = hOut
    qbc[1, -num_ghost:] = qbc[1, -num_ghost - 1]


def source_mannings(solver,state,dt):
    """
    Take a look at the paper on Hudson et al. 2005 regarding treatment of the source. 
    
    Note that I did run it for calm or quiesent conditions (i.e. u=0 for all t) and everything seemed okay.
           | 0 |
    S(Q) = |   |
           |gh(Slope-Sf)|
    """
    #Eventually the slope should be calculate from the actual bed.
    # Slope = 1.0/792.0
    q = state.q
    #grid = state.grid
    #xc=grid.x.centers 
    # Get the flow depth
    # Now adjust the momentum term
    n = state.problem_data['mannings']
    Slope = state.problem_data['slope']
    
    Sf = (n**2)*q[1,:]*np.abs(q[1,:])/(q[0,:]**(10./3.))
    #q[1,:] = q[1,:] + q[0,:]* state.problem_data['grav'] * (Slope-Sf) *dt
    #print(Slope_bed.mean(), Slope)
    q[1,:] = q[1,:] + q[0,:]* state.problem_data['grav'] * (Slope-Sf) *dt


def source_chezy(solver,state,dt):
    
    kappa = 0.4
    Bs = 8.5
    b = 0.76
    Slope = state.problem_data['slope']
    ks = state.problem_data['ks']
    g = state.problem_data['grav']
    q = state.q
     
    # From eq. 1.14 in Yalin and da Silva
    cf = ((1./kappa)*np.log(0.368*q[0,:]/ks) + Bs)   
    
    # Based on eq. 1.16 and solving for S. (mulitply by h first)
    Sf = q[1,:]*np.abs(q[1,:])/(g*(cf**2.)*q[0,:]**(3.))
    
    q[1,:] = q[1,:] + q[0,:]*g*(Slope - Sf)*dt

class shallow_water_solver():
    
    def __init__(self, kernel_language='Fortran', solver_type='classic'):
        
        # ============================
        # Select the solver
        # ============================
        if kernel_language == 'Fortran':
            self.solver = pyclaw.ClawSolver1D(riemann.shallow_bathymetry_fwave_1D)
            #self.solver.kernel_language = 'Fortran'
        elif kernel_language == 'Python':
            self.solver = pyclaw.ClawSolver1D(riemann.shallow_1D_py.shallow_fwave_1d)
            self.solver.kernel_language = 'Python'
            
        self.state = None
        self.controller = None
        self.domain = None
        
    def get_controller(self):
        return self.controller
    
    def get_state(self):
        return self.state
    
    def get_hf(self):
        return self.controller.frames[self.controller.num_output_times].q[0,:]
        
    def get_qf(self):
        return self.controller.frames[self.controller.num_output_times].q[1,:]
    
    def get_uf(self):
        depth = self.controller.frames[self.controller.num_output_times].q[0,:]
        return self.controller.frames[self.controller.num_output_times].q[1,:]/depth
            
    def set_solver(self, limiter = pyclaw.limiters.tvd.vanleer, source_term=source_mannings, max_steps = 10000):
        # ===============================
        # Configure the solver
        # ===============================
        self.solver.limiters = pyclaw.limiters.tvd.minmod
        self.solver.fwave = True
        self.solver.num_waves = 2
        self.solver.num_eqn = 2        
        self.solver.max_steps = max_steps
        self.solver.source_split = 1
        self.solver.order = 2
        
    def set_state_domain(self,x,z):
        # ============================
        # Setup the domain and state
        # ============================
        x = pyclaw.Dimension(0.0,x.max(),len(x),name='x')
        self.domain = pyclaw.Domain(x)
        self.state = pyclaw.State(self.domain, 2, 1)
        
        xc = self.state.grid.x.centers
        dx = self.state.grid.delta[0]      
        
        # Specify the bathymetry
        self.state.aux[0, :] = z
        
        # Gravitational constant
        self.state.problem_data['grav'] = 9.8
        self.state.problem_data['dry_tolerance'] = 1.e-5
        self.state.problem_data['sea_level'] = 0.0
        
    def set_mannings_source_term(self, mannings=0.022, slope=1/792.):        
        self.solver.step_source = source_mannings
        self.state.problem_data['mannings'] = mannings
        self.state.problem_data['slope'] = slope
        self.state.problem_data['ks'] = 0.0033
        
    def set_chezy_source_term(self, ks=0.0033, slope=1/792.):        
        self.solver.step_source = source_chezy
        self.state.problem_data['ks'] = ks
        self.state.problem_data['slope'] = slope
        self.state.problem_data['mannings'] = 0.024
        self.state.problem_data['slope'] = slope
        
        
        
    def set_inital_conditions(self, surface, intial_flow):
        # Set the starting water depth (h)
        self.state.q[0, :] = surface - self.state.aux[0, :]
        # Set the intial flow         
        self.state.q[1, :] = intial_flow
        
    def set_conditions_from_previous(self, h, q):
        # Set the starting water depth (h)
        self.state.q[0, :] = h
        # Set the intial flow         
        self.state.q[1, :] = q
        
        
    def set_boundary_conditions(self,
                                bc_lower = pyclaw.BC.periodic,
                                bc_upper = pyclaw.BC.periodic,
                                aux_bc_lower = pyclaw.BC.periodic,
                                aux_bc_upper = pyclaw.BC.periodic):
        self.solver.bc_lower[0] = bc_lower
        self.solver.bc_upper[0] = bc_upper
        self.solver.aux_bc_lower[0] = aux_bc_lower
        self.solver.aux_bc_upper[0] = aux_bc_upper
        
        self.state.problem_data['lower_bc_data'] = 0
        self.state.problem_data['upper_bc_data'] = 0
        
        
    def set_Dirichlet_BC(self, sOut, qIn):
        
        self.solver.bc_lower[0] = pyclaw.BC.custom
        self.solver.bc_upper[0] = pyclaw.BC.custom
        
        self.state.problem_data['lower_bc_data'] = qIn
        self.state.problem_data['upper_bc_data'] = sOut

        self.solver.user_bc_lower = inlet_BC
        self.solver.user_bc_upper = outlet_BC

        self.solver.aux_bc_lower[0] = pyclaw.BC.extrap
        self.solver.aux_bc_upper[0] = pyclaw.BC.extrap
    

        
        
        
    def set_controller(self, tfinal, num_output_times=1, 
                       write_aux_init=True, keep_copy=True):
        # ============================
        # Setup the controller
        # ============================
        self.controller = pyclaw.Controller()
        self.controller.keep_copy = keep_copy
        self.controller.tfinal = tfinal
        self.controller.solution = pyclaw.Solution(self.state, self.domain)
        self.controller.solver = self.solver
        self.controller.write_aux_init = write_aux_init
        self.controller.num_output_times = num_output_times
        
        
    def run(self):
        status = None
        #print('num_dim: {0}'.format(self.solver.num_dim))
        
        if self.controller != None and self.state!=None:
            status = self.controller.run()
        else:
            raise ValueError('Model is not parameterized...')
        return status
    






