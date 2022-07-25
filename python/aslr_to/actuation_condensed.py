###################################3
####
####Deprecated code
####
#####################################



import numpy as np
import pinocchio
import crocoddyl

class ASRActuationCondensed(crocoddyl.ActuationModelAbstract):
    #u = [u ; \ddot{theta}]
    def __init__(self, state, nu, B):
        crocoddyl.ActuationModelAbstract.__init__(self, state, nu)
        self.B = B

    def calc(self, data, x, u):
        nv = self.state.nv
        data.tau = u[:nv] - np.dot(self.B,u[nv:2*nv])

    def calcDiff(self, data, x, u):
        nv = self.state.nv
        data.dtau_du[:, :nv] =  np.eye(nv)
        data.dtau_du[:, nv:2*nv] =  -self.B
