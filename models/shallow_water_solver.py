from clawpack import riemann
import clawpack.petclaw as pyclaw
import numpy as np
import math
from scipy.sparse.linalg.isolve._iterative import zbicgrevcom
# http://www.clawpack.org/pyclaw/parallel.html

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
 
    cf = ((1./kappa)*np.log(0.368*q[0,:]/ks) + Bs)
    R = b*q[0,:]/(b + 2*q[0,:])
    n = R**(1/6.)/cf
    
    
    Sf = (n**2)*q[1,:]*np.abs(q[1,:])/(q[0,:]**(10./3.))
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
        self.solver.limiters = pyclaw.limiters.tvd.vanleer
        self.solver.fwave = True
        self.solver.num_waves = 2
        self.solver.num_eqn = 2        
        self.solver.max_steps = max_steps
        
    def set_state_domain(self,x,z):
        # ============================
        # Setup the domain and state
        # ============================
        x = pyclaw.Dimension(0.0,x.max(),len(x),name='x')
        self.domain = pyclaw.Domain(x)
        self.state = pyclaw.State(self.domain, 2, 1)
        
        xc = self.state.grid.x.centers
        dx = self.state.grid.delta[0]
        #print('Grid dx = {0}'.format(dx))
        #print('Grid nx = {0}'.format(len(xc)))
        
        
        # Specify the bathymetry
        self.state.aux[0, :] = z
        
        # Gravitational constant
        self.state.problem_data['grav'] = 9.8
        self.state.problem_data['dry_tolerance'] = 1.e-3
        self.state.problem_data['sea_level'] = 0.0
        
    def set_mannings_source_term(self, mannings=0.022, slope=1/792.):        
        self.solver.step_source = source_mannings
        self.state.problem_data['mannings'] = mannings
        self.state.problem_data['slope'] = slope
        
    def set_chezy_source_term(self, ks=0.0033, slope=1/792.):        
        self.solver.step_source = source_chezy
        self.state.problem_data['ks'] = ks
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
        if self.controller != None and self.state!=None:
            status = self.controller.run()
        else:
            raise ValueError('Model is not parameterized...')
        return status
    






