import numpy as np
import pinocchio
import crocoddyl

class ASRFreeFloatingActuation(crocoddyl.ActuationModelAbstract):
    def __init__(self, state):
        assert (state.pinocchio.joints[1].shortname() == 'JointModelFreeFlyer')
        crocoddyl.ActuationModelAbstract.__init__(self, state, state.nv_m)

    def calc(self, data, x, u):
        data.tau[self.nu:] = u

    def calcDiff(self, data, x, u):
        data.dtau_du[self.nu:, :] =  np.eye(self.nu)
