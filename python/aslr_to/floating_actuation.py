import numpy as np
import pinocchio
import crocoddyl

class ASRFreeFloatingActuation(crocoddyl.ActuationModelAbstract):
    def __init__(self, state, K, B):
        assert (state.pinocchio.joints[1].shortname() == 'JointModelFreeFlyer')
        crocoddyl.ActuationModelAbstract.__init__(self, state, state.nv_m)
        self.nv_l = state.nv_l
        self.nq_l = state.nq_l
        self.nv_m =state.nv_m
        self.K = K
        self.B = B

    def calc(self, data, x, u):
        nl= self.nq_l+self.nv_l
        q_l = x[:self.nq_l]
        q_m = x[nl:-self.nu]
        tau_couple = np.zeros(self.nv_l)
        tau_couple[-self.state.nv_m:] = np.dot(self.K[-self.nu:,-self.nu:], q_l[-self.state.nv_m:]-q_m)
        data.tau[:self.nv_l] = tau_couple
        data.tau[-self.nu:] = u - tau_couple[-self.state.nv_m:]

    def calcDiff(self, data, x, u):
        data.dtau_dx[:self.nv_l,:self.nv_l] = self.K[-self.nv_l:,-self.nv_l]
        data.dtau_dx[:self.nv_l,2*self.nv_l:-self.nu] = -self.K[-self.nv_l:,-self.nu:]
        data.dtau_dx[self.nv_l:,:self.nv_l] = -self.K[-self.nv_m:,-self.nv_l:]
        data.dtau_dx[self.nv_l: ,2*self.nv_l:-self.nu] = self.K[-self.nv_m:,-self.nu:]
        data.dtau_du[-self.nu:, :] =  np.eye(self.nu)

