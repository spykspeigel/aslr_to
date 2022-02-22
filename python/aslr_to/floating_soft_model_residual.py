import numpy as np
import pinocchio
import crocoddyl


class FloatingSoftDynamicsResidualModel(crocoddyl.ResidualModelAbstract):
    def __init__(self, state, nu, K):
        crocoddyl.ResidualModelAbstract.__init__(self, state, state.nv-6, nu, True, True, True)
        self.K = K

    def calc(self, data, x, u):
        nv = self.state.nv-6
        nq = self.state.nq
        q = x[:nq]
        tau = u[:nv]
        theta_dot_dot = u[nv:2*nv]
        # print(np.dot(self.K,q[-nv:]).shape)
        # print(tau)
        data.r = np.dot(self.K,q[-nv:]) + tau -theta_dot_dot

    def calcDiff(self, data, x, u):
        nq = self.state.nq
        nv = self.state.nv -6
        data.Rx[:, 6:self.state.nv] = self.K[-nv:,-nv:]
        data.Ru[:, :nv] = np.eye(nv)
        data.Ru[:, nv:2*nv] = -np.eye(nv)

class FloatingSoftDynamicsResidualData(crocoddyl.ResidualDataAbstract):
    def __init__(self, model, collector):
        crocoddyl.ResidualDataAbstract.__init__(self, model, collector)
        