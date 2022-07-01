import numpy as np
import pinocchio
import crocoddyl


class ResidualModelTauCouple(crocoddyl.ResidualModelAbstract):
    def __init__(self, state, nu, K):
        crocoddyl.ResidualModelAbstract.__init__(self, state, state.nv_m, nu, True, True, True)
        self.K = K

    def calc(self, data, x, u):
        nq_l = self.state.nq_l
        nv_l = self.state.nv_l
        nv_m = self.state.nv_m
        nl = nq_l + nv_l
        q_l = x[:nq_l]
        theta = x[nl:-nv_m]
        tau = u[nv_m:]
        data.r[:] = tau - np.dot(self.K[-nv_m:,-nv_m:], ( -theta + q_l[7:]))

    def calcDiff(self, data, x, u):
        nq_l = self.state.nq_l
        nv_l = self.state.nv_l
        nv_m = self.state.nv_m
        nl = nq_l + nv_l

        # K = self.K[-nv_m:,-nv_m:]

        data.Rx[:, 6:nv_l] = -self.K[-nv_m:,-nv_m:]
        data.Rx[:, 2*nv_l:-nv_m] = self.K[-nv_m:,-nv_m:]
    
        data.Ru[:, nv_m:] = np.eye(nv_m)

class ResidualDataTauCouple(crocoddyl.ResidualDataAbstract):
    def __init__(self, model, collector):
        crocoddyl.ResidualDataAbstract.__init__(self, model, collector)
        