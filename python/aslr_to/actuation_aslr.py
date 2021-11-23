import numpy as np
import pinocchio
import crocoddyl

class ASRActuation(crocoddyl.ActuationModelAbstract):
    def __init__(self, state):
        crocoddyl.ActuationModelAbstract.__init__(self, state, int(state.nv/2))

    def calc(self, data, x, u):
        data.tau[self.nu:] = u

    def calcDiff(self, data, x, u):
        data.dtau_du[self.nu:, :] =  np.eye(self.nu)
