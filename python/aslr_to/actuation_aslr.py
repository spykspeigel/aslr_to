import numpy as np
import pinocchio
import crocoddyl

class ASRActuation(crocoddyl.ActuationModelAbstract):
    def __init__(self, state):
        crocoddyl.ActuationModelAbstract.__init__(self, state, state.nv_l-6)

    def calc(self, data, x, u):
        data.tau[-self.nv_m:] = u

    def calcDiff(self, data, x, u):
        data.dtau_du[-self.nv_m:, :] =  np.eye(self.nu)
