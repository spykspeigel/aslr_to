import numpy as np
import pinocchio
import crocoddyl

class ASRActuationCondensed(crocoddyl.ActuationModelAbstract):
    #u = [u ; \ddot{theta}]
    def __init__(self, state):
        crocoddyl.ActuationModelAbstract.__init__(self, state, 2*state.nv)

    def calc(self, data, x, u):
        nv = self.state.nv
        data.tau = u[:nv] - u[nv:]

    def calcDiff(self, data, x, u):
        nv = self.state.nv
        data.dtau_du[:, :nv] =  np.eye(nv)
        data.dtau_du[:, nv:] =  -np.eye(nv)
