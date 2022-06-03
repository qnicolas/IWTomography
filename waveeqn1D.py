""" 
This file was built to solve numerically a classical PDE, 1D wave equation. The equation corresponds to :
$c(x)^2\dfrac{\partial}{\partial x} \left( \dfrac{\partial U}{\partial x} \right) = \dfrac{\partial^2 U}{\partial t^2}$
 
where
 - U represent the signal
 - x represent the position
 - t represent the time
 - c represent the velocity of the wave (depends on space parameters)

The numerical scheme is based on finite difference method. This program is also providing several boundary conditions. More particularly the Neumann, Dirichlet and Mur boundary conditions.
Copyright - © SACHA BINDER - 2021 (QN - inspired from)
"""

############## MODULES IMPORTATION ###############
import numpy as np
import matplotlib.pyplot as plt

class WaveEquation:
    def __init__(self,leftbc,rightbc,Lx=1.5,Lt=4,dx=0.01):
        self.left_bound_cond = leftbc
        self.right_bound_cond = leftbc
        assert self.left_bound_cond in [1,2,3]
        assert self.right_bound_cond in [1,2,3]
        
        #Spatial mesh - i indices
        self.L_x = Lx #Range of the domain according to x [m]
        self.dx = dx #Infinitesimal distance
        self.N_x = int(self.L_x/self.dx) #Points number of the spatial mesh
        self.X = np.linspace(0,self.L_x,self.N_x+1) #Spatial array
        
        #Temporal mesh with CFL < 1 - j indices
        self.L_t = Lt #Duration of simulation [s]
        self.dt = 0.01*self.dx  #Infinitesimal time with CFL (Courant–Friedrichs–Lewy condition)
        self.N_t = int(self.L_t/self.dt) #Points number of the temporal mesh
        self.T = np.linspace(0,self.L_t,self.N_t+1) #Temporal array
    
    def I(self,x):
        """
        Single space variable fonction that 
        represent the wave form at t = 0
        """
        return np.exp(-(x)**2/0.01)

    ############## SET-UP THE PROBLEM ###############
    def celer(self,x):
        """
        Single space variable fonction that represent 
        the wave's velocity at a position x
        """
        return (x <=0.7) + 0.5*(x>0.7)

    def prepare_loop(self):
        #SET u0 and u1
        self.c = self.celer(self.X)
        self.q = self.c**2

        self.c_1 = self.c[0]
        self.c_2 = self.c[self.N_x]
        
        self.C2 = (self.dt/self.dx)**2

        self.CFL_1 = self.c_1*(self.dt/self.dx)
        self.CFL_2 = self.c_2*(self.dt/self.dx)
        
        # $\forall i \in {0,...,N_x}$
        self.u_jm1 = np.zeros(self.N_x+1,float)   #Vector array u_i^{j-1}
        self.u_j   = np.zeros(self.N_x+1,float)   #Vector array u_i^j
        self.u_jp1 = np.zeros(self.N_x+1,float)   #Vector array u_i^{j+1}
        
        self.U = np.zeros((self.N_x+1,self.N_t+1),float) #Global solution
        
        #init cond - at t = 0
        self.u_j = self.I(self.X)
        self.U[:,0] = self.u_j.copy()
        
        
        #init cond - at t = 1
        #without boundary cond
        self.u_jp1[1:self.N_x] =  (self.u_j[1:self.N_x] 
                                  + 0.5*self.C2*( 0.5*(self.q[1:self.N_x] + self.q[2:])*(self.u_j[2:] - self.u_j[1:self.N_x])
                                                - 0.5*(self.q[:self.N_x-1] + self.q[1:self.N_x])*(self.u_j[1:self.N_x] - self.u_j[:self.N_x-1]))
                                  )
    
    
        ########   SET LEFT BOUNDARY CONDITION FOR FIRST TIME STEP  ########
        if self.left_bound_cond == 1:
            #Dirichlet bound cond
            self.u_jp1[0] = 0
            
        elif self.left_bound_cond == 2:
            #Neumann bound cond
            #i = 0
            self.u_jp1[0] = self.u_j[0] +0.5*self.C2*( 0.5*(self.q[0] + self.q[0+1])*(self.u_j[0+1] - self.u_j[0]) - 0.5*(self.q[0] + self.q[0+1])*(self.u_j[0] - self.u_j[0+1]))
                            
        elif self.left_bound_cond == 3:
            #Mur bound cond
            #i = 0
            self.u_jp1[0] = self.u_j[1] + (self.CFL_1 -1)/(self.CFL_1 + 1)*( self.u_jp1[1] - self.u_j[0])
    
        ########   SET RIGHT BOUNDARY CONDITION FOR FIRST TIME STEP  ########
        if self.right_bound_cond == 1:
            #Dirichlet bound cond
            self.u_jp1[self.N_x] = 0
            
        elif self.right_bound_cond == 2:
            #Nuemann bound cond
            #i = N_x
            self.u_jp1[self.N_x] =  self.u_j[self.N_x] + 0.5*self.C2*( 0.5*(self.q[self.N_x-1] + self.q[self.N_x])*(self.u_j[self.N_x-1] - self.u_j[self.N_x]) - 0.5*(self.q[self.N_x-1] + self.q[self.N_x])*(self.u_j[self.N_x] - self.u_j[self.N_x-1]))
            
        elif self.right_bound_cond == 3:
            #Mur bound cond
            #i = N_x
            self.u_jp1[self.N_x] = self.u_j[self.N_x-1] + (self.CFL_2 -1)/(self.CFL_2 + 1)*(self.u_jp1[self.N_x-1] - self.u_j[self.N_x])
        
        self.u_jm1 = self.u_j.copy()  #go to the next step
        self.u_j = self.u_jp1.copy()  #go to the next step
        self.U[:,1] = self.u_j.copy()
    
    
    ######################################################################
    ############################ TIME STEPPING ###########################
    ######################################################################
    def time_step(self,j):
        #without boundary cond
        self.u_jp1[1:self.N_x] = (-self.u_jm1[1:self.N_x] 
                                  +2*self.u_j[1:self.N_x] 
                                  + self.C2*( 0.5*(self.q[1:self.N_x] + self.q[2:])*(self.u_j[2:] - self.u_j[1:self.N_x])
                                             -0.5*(self.q[:self.N_x-1] + self.q[1:self.N_x])*(self.u_j[1:self.N_x] - self.u_j[:self.N_x-1]))
                                 )
           
        ########   SET LEFT BOUNDARY CONDITION  ########
        if self.left_bound_cond == 1:
            #Dirichlet bound cond
            self.u_jp1[0] = 0

        elif self.left_bound_cond == 2:
            #Nuemann bound cond
            #i = 0
            self.u_jp1[0] = -self.u_jm1[0] + 2*self.u_j[0] + self.C2*( 0.5*(self.q[0] + self.q[0+1])*(self.u_j[0+1] - self.u_j[0]) - 0.5*(self.q[0] + self.q[0+1])*(self.u_j[0] - self.u_j[0+1])) 
                             
        elif self.left_bound_cond == 3:
            #Mur bound cond
            #i = 0
            self.u_jp1[0] = self.u_j[1] + (self.CFL_1 -1)/(self.CFL_1 + 1)*( self.u_jp1[1] - self.u_j[0])

        ########   SET LEFT BOUNDARY CONDITION  ########
        if self.right_bound_cond == 1:
            #Dirichlet bound cond
            self.u_jp1[self.N_x] = 0
            
        elif self.right_bound_cond == 2:
            #Nuemann bound cond
            #i = N_x
            self.u_jp1[self.N_x] = -self.u_jm1[self.N_x] + 2*self.u_j[self.N_x] + self.C2*( 0.5*(self.q[self.N_x-1] + self.q[self.N_x])*(self.u_j[self.N_x-1] - self.u_j[self.N_x]) - 0.5*(self.q[self.N_x-1] + self.q[self.N_x])*(self.u_j[self.N_x] - self.u_j[self.N_x-1]))
            
        elif self.right_bound_cond == 3:
            #Mur bound cond
            #i = N_x
            self.u_jp1[self.N_x] = self.u_j[self.N_x-1] + (self.CFL_2 -1)/(self.CFL_2 + 1)*(self.u_jp1[self.N_x-1] - self.u_j[self.N_x])
        
        self.u_jm1[:] = self.u_j.copy()   #go to the next step
        self.u_j[:] = self.u_jp1.copy()   #go to the next step
        self.U[:,j] = self.u_j.copy()
        
    def integrate(self):
        self.prepare_loop()
        for j in range(1,self.N_t+1):
            self.time_step(j)
