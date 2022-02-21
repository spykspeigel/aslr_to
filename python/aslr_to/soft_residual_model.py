import numpy as np
import pinocchio
import crocoddyl


class SoftDynamicsResidualModel(crocoddyl.ResidualModelAbstract):
    def __init__(self, state, nu, K):
        crocoddyl.ResidualModelAbstract.__init__(self, state,state.nv, nu, True, True, True)
        self.K = K

    def calc(self, data, x, u):
        nv = int(self.nu/2)
        nq = self.state.nq
        q = x[:nq]

        tau = u[:nv]
        theta_dot_dot = u[nv:2*nv]
        data.r[:] = np.dot(self.K,q) + tau -theta_dot_dot

    def calcDiff(self, data, x, u):
        nq = self.state.nq
        nv = self.state.nv
        data.Rx[:, :nv] = self.K[-nv:,-nv:]
        print(data.Ru.shape)
        data.Ru[:, :nv] = np.eye(nv)
        data.Ru[:, nv:2*nv] = -np.eye(nv)
class SoftDynamicsResidualData(crocoddyl.ResidualDataAbstract):
    def __init__(self, model, collector):
        crocoddyl.ResidualDataAbstract.__init__(self, model, collector)
        nv = model.state.nv
        self.Ru[:, :nv] = np.eye(nv)
        self.Ru[:, nv:2*nv] = -np.eye(nv)
