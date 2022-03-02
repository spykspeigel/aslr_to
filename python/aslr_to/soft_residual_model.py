import numpy as np
import pinocchio
import crocoddyl


class SoftDynamicsResidualModel(crocoddyl.ResidualModelAbstract):
    def __init__(self, state, nu, K, B):
        crocoddyl.ResidualModelAbstract.__init__(self, state,state.nv, nu, True, True, True)
        self.K = K
        self.B = B
    def calc(self, data, x, u):
        nv = int(self.nu/2)
        nq = self.state.nq
        q = x[:nq]

        tau = u[:nv]
        theta_dot_dot = u[nv:2*nv]
        data.r[:] = q[-nv:] + np.dot(np.linalg.inv(self.K), (tau -np.dot(self.B, theta_dot_dot)))

    def calcDiff(self, data, x, u):
        nq = self.state.nq
        nv = self.state.nv

        Kinv = np.linalg.inv(self.K)
        data.Rx[:, :nv] = np.eye(nv)
        data.Ru[:, :nv] = Kinv
        data.Ru[:, nv:2*nv] = -np.dot(Kinv,self.B)

class SoftDynamicsResidualData(crocoddyl.ResidualDataAbstract):
    def __init__(self, model, collector):
        crocoddyl.ResidualDataAbstract.__init__(self, model, collector)
        nv = model.state.nv
        # self.Ru[:, :nv] = np.eye(nv)
        # self.Ru[:, nv:2*nv] = -np.eye(nv)
