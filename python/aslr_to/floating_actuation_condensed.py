import numpy as np
import pinocchio
import crocoddyl
class FreeFloatingActuationCondensed(crocoddyl.ActuationModelAbstract):

    def __init__(self, state):
        assert (state.pinocchio.joints[1].shortname() == 'JointModelFreeFlyer')
        crocoddyl.ActuationModelAbstract.__init__(self, state, 2*(state.nv - 6))

    def calc(self, data, x, u):
        nv = self.state.nv -6
        data.tau = np.hstack([np.zeros(6), u[:nv]-u[nv:]])

    def calcDiff(self, data, x, u):
        nv = self.state.nv -6
        data.dtau_du[:, :nv] = np.vstack([np.zeros((6, nv)), np.eye(nv)])
        data.dtau_du[:, nv:] = np.vstack([np.zeros((6, nv)), -np.eye(nv)])