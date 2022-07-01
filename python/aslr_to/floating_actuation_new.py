import numpy as np
import pinocchio
import crocoddyl
class FreeFloatingActuationNew(crocoddyl.ActuationModelAbstract):

    def __init__(self, state,nu):
        assert (state.pinocchio.joints[1].shortname() == 'JointModelFreeFlyer')
        crocoddyl.ActuationModelAbstract.__init__(self, state, nu)

    def calc(self, data, x, u):
        data.tau = np.hstack([np.zeros(6), u[:self.state.nv_m]])

    def calcDiff(self, data, x, u):
        data.dtau_du[6:self.state.nv_l, :self.state.nv_m] = np.eye(self.state.nv_m)
