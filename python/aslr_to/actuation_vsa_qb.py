import numpy as np
import pinocchio
import crocoddyl

#   Actuator model for qbMove VSA 
#   Hope it will result in smoother output trajectories  

class ASRActuation(crocoddyl.ActuationModelAbstract):
    def __init__(self, state):
        crocoddyl.ActuationModelAbstract.__init__(self, state, int(state.nv/2))

    def calc(self, data, x, u):
        r = u[:self.nu]
        d = u[self.nu:]

        nq=self.state.nq
        # nv=self.state.nv
        nq_l = int(nq/2)
        # nv_l = int(nv/2)
        # q_l = x[:nq_l]
        q_m = x[nq_l:nq]
        a = (data.a1+data.a2)/2.0
        k = (data.k2+data.k2)/2.0
        
        data.tau[int(self.state.nv/2):] = 2*k*np.cosh(a*d)*np.sinh(a(q_m-r))
        data.K = 2*a*k*np.cosh(a*d)*np.cosh(a(q_m-r))
    
    def calcDiff(self, data, x, u):
        # r,d are the output variables
        # q are the input variables
        # a1, a2, k1, k2 are the parameters

        nq=self.state.nq
        # nv=self.state.nv
        nq_l = int(nq/2)
        # nv_l = int(nv/2)
        # q_l = x[:nq_l]
        q_m = x[nq_l:nq]
        # v_l = x[-nv:-nv_l]
        # v_m = x[-nv_l:]
        # x_m = np.hstack([q_m,v_m])
        r = u[:self.nu]
        d = u[self.nu:]

        t1 = q_m-r
        a = (data.a1+data.a2)/2.0
        k = (data.k2+data.k2)/2.0
        t2 = d*a
        t3 = t1*a
        t4 = np.cosh(t2)
        t5 = np.sinh(t3)
        t6 = 2*a*a*k*t4*t5
        
        data.dK_du[:,:int(self.state.nv/2)] = -np.diag(t6)
        data.dK_du[:,int(self.state.nv/2):] = np.diag(2*a*a*k*np.cosh(t3)*np.sinh(t2))
        data.dK_dx[:,nq_l:nq] = np.diag(t6)




        t1 = q_m-u[:self.nu]
        a = (data.a1+data.a2)/2.0
        k = (data.k2+data.k2)/2.0

        t11 = d*a
        t13 = t1*a
        t12 = np.cosh(t11)
        t14 = np.sinh(t13)
        t15 = 2*a*a*k*t12*t14
        tau_simb_dot = [-t15,2*a*a*k*np.cosh(t13)*np.sinh(t11),t15]
        
        data.dtau_du[int(self.state.nv/2):] =  
        data.dtau_dx[int(self.state.nv/2):, :] =  np.eye(self.nu)
