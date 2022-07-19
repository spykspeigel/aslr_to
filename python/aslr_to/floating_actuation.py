import numpy as np
import pinocchio
import crocoddyl
class ASRFreeFloatingActuation(crocoddyl.ActuationModelAbstract):

    def __init__(self, state):
        assert (state.pinocchio.joints[1].shortname() == 'JointModelFreeFlyer')
        crocoddyl.ActuationModelAbstract.__init__(self, state, state.nv_m)

    def calc(self, data, x, u):
        data.tau = np.hstack([np.zeros(6), u[:self.state.nv_m]])

    def calcDiff(self, data, x, u):
        data.dtau_du[6:self.state.nv_l, :self.state.nv_m] = np.eye(self.state.nv_m)
