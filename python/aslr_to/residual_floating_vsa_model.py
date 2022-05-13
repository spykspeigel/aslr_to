import numpy as np
import pinocchio
import crocoddyl


class FloatingVSADynamicsResidualModel(crocoddyl.ResidualModelAbstract):
    def __init__(self, state, nu):
        crocoddyl.ResidualModelAbstract.__init__(self, state, state.nv-6, nu, True, True, True)

    def calc(self, data, x, u):
        nv = self.state.nv-6
        nq = self.state.nq
        q = x[:nq]
        tau = u[:nv]
        theta_dot_dot = u[nv:2*nv]
        K = np.diag(u[2*nv:])
        data.r[:] = q[-nv:] + np.dot(np.linalg.inv(K), (tau - theta_dot_dot))

    def calcDiff(self, data, x, u):
        nq = self.state.nq
        nv = self.state.nv -6
        tau = u[:nv]
        theta_dot_dot = u[nv:2*nv]
        K = np.diag(u[2*nv:])
        Kinv = np.linalg.inv(K)
        data.Rx[:, 6:self.state.nv] = np.eye(nv)
        data.Ru[:, :nv] = Kinv
        data.Ru[:, nv:2*nv] = -Kinv
        data.Ru[:, 2*nv:] = -np.dot(np.linalg.inv(K*K),np.diag((tau - theta_dot_dot)))

class FloatingVSADynamicsResidualData(crocoddyl.ResidualDataAbstract):
    def __init__(self, model, collector):
        crocoddyl.ResidualDataAbstract.__init__(self, model, collector)
