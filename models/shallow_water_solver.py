from clawpack import pyclaw
from clawpack import riemann



import numpy as np
import math
from scipy.sparse.linalg.isolve._iterative import zbicgrevcom




class shallow_solver(): 

    
    
    def __init__(self,domain):
        self.slope = 0.001
        self.mannings = 0.032
        self.domain = domain

    
    def run(self, zb,s):
        claw = pyclaw.Controller()
        claw.keep_copy = True       # Keep solution data in memory for plotting
        claw.output_format = None   # Don't write solution data to file
        claw.num_output_times = 1  # Write 50 output frames
        
        solver = pyclaw.ClawSolver1D(riemann.shallow_1D_py.shallow_fwave_1d)
        solver.limiters = pyclaw.limiters.tvd.vanleer
        solver.kernel_language = "Python"
        
        solver.step_source = source_term_here
        solver.verbosity = 10
        
        solver.fwave = True
        solver.num_waves = 2
        solver.num_eqn = 2
        
        #solver.bc_lower[0] = pyclaw.BC.extrap
        #solver.bc_upper[0] = pyclaw.BC.extrap
        #solver.aux_bc_lower[0] = pyclaw.BC.extrap
        #solver.aux_bc_upper[0] = pyclaw.BC.extrap
        
        solver.bc_lower[0] = pyclaw.BC.periodic
        solver.bc_upper[0] = pyclaw.BC.periodic
        solver.aux_bc_lower[0] = pyclaw.BC.periodic
        solver.aux_bc_upper[0] = pyclaw.BC.periodic
        
        
        state = pyclaw.State(self.domain, 2, 1)
        
         # Gravitational constant
        state.problem_data['grav'] = 9.8
        state.problem_data['sea_level'] = s
        state.problem_data['dry_tolerance'] = 1e-3
        #print(len(zb),len(state.aux[0, :]))
        state.aux[0, :] = zb
        
        # This is a flat surface
        state.q[0, :] = s - state.aux[0, :]
        
        # Set the intial flow to 0.0 m/s
        state.q[1, :] = 0.0
        
        claw.tfinal = 120.0
        claw.solution = pyclaw.Solution(state, self.domain)
        claw.solver = solver
        claw.write_aux_init = True
        
        status = claw.run()
        
        depth = claw.frames[claw.num_output_times].q[0,:]
        velocity = claw.frames[claw.num_output_times].q[1,:]/depth
        surface = depth +  state.aux[0, :]
        
        
        return velocity, surface, depth




def source_term_here(solver,state,dt):
    """
    Take a look at the paper on Hudson et al. 2005 regarding treatment of the source. 
    
    Note that I did run it for calm or quiesent conditions (i.e. u=0 for all t) and everything seemed okay.
           | 0 |
    S(Q) = |   |
           |gh(Slope-Sf)|
    """
    #Eventually the slope should be calculate from the actual bed.
    Slope = 1.0/792.0
    q = state.q
    #grid = state.grid
    #xc=grid.x.centers 
    # Get the flow depth
    # Now adjust the momentum term
    n = 0.025
    Sf = (n**2)*q[1,:]*np.abs(q[1,:])/(q[0,:]**(10./3.))
    q[1,:] = q[1,:] + q[0,:]* state.problem_data['grav'] * (Slope-Sf) *dt

