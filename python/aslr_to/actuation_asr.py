import numpy as np
import pinocchio
import crocoddyl

class ASRActuation(crocoddyl.ActuationModelAbstract):
    def __init__(self, state):
        crocoddyl.ActuationModelAbstract.__init__(self, state, int(state.nv/2))

    def calc(self, data, x, u):
        data.tau[int(self.state.nv/2):] = u[:self.nu]

    def calcDiff(self, data, x, u):
        data.dtau_du[int(self.state.nv/2):, :] =  np.eye(self.nu)
