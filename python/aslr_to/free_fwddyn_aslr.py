import numpy as np
import pinocchio
import crocoddyl


class DifferentialFreeASRFwdDynamicsModel(crocoddyl.DifferentialActionModelAbstract):
    def __init__(self, state, actuationModel, costModel):
        crocoddyl.DifferentialActionModelAbstract.__init__(self, state, actuationModel.nu, costModel.nr)
        self.actuation = actuationModel
        self.costs = costModel
        self.enable_force = True

    def calc(self, data, x, u=None):
        if u is None:
            u = np.zeros(self.nu)
        nq=self.state.nq
        nv=self.state.nv
        q_l = x[:int(nq/2)]
        q_m = x[int(nq/2):nq]
        v_l = x[-nv:-int(nv/2)]
        v_m = x[-int(nv/2):]
        x_m = np.hstack([q_m,v_m])

        self.actuation.calc(data.actuation, x_m, u)
        tau = data.actuation.tau

        data.tau_couple = np.dot(data.K, q_l-q_m)

        # Computing the fwd dynamics manually
        pinocchio.computeAllTerms(self.state.pinocchio, data.pinocchio, q_l, v_l)
        data.M = data.pinocchio.M
        data.Minv = np.linalg.inv(data.M)
        data.Binv = np.linalg.inv(data.B)
        data.xout[:int(nv/2)] = np.dot(data.Minv, (tau[:int(nv/2)] - data.pinocchio.nle - data.tau_couple))

        # Computing the motor side dynamics
        data.xout[int(nv/2):] = np.dot(data.Binv, tau[int(nv/2):] + data.tau_couple)

        # Computing the cost value and residuals
        pinocchio.forwardKinematics(self.state.pinocchio, data.pinocchio, q_l, v_l)
        pinocchio.updateFramePlacements(self.state.pinocchio, data.pinocchio)
        self.costs.calc(data.costs, x, u)
        data.cost = data.costs.cost

        # Computing the cost value and residuals
        self.costs.calc(data.costs, x, u)
        data.cost = data.costs.cost

    def calcDiff(self, data, x, u=None):
        if u is None:
            u = self.unone
        nq, nv = self.state.nq, self.state.nv
        q_l = x[:int(nq/2)]
        q_m = x[int(nq/2):nq]
        v_l = x[-nv:-int(nv/2)]
        v_m = x[-int(nv/2):]

        x_m = np.hstack([q_m,v_m])
        # Computing the actuation derivatives
        self.actuation.calcDiff(data.actuation, x_m, u)
        tau = data.actuation.tau
        
        # Computing the dynamics derivatives
        pinocchio.computeRNEADerivatives(self.state.pinocchio, data.pinocchio, q_l, v_l, data.xout[:int(nv/2)])
        ddq_dq = np.dot(data.Minv, ( - data.pinocchio.dtau_dq - data.K))
        ddq_dv = np.dot(data.Minv, ( - data.pinocchio.dtau_dv))
        data.Fx[:int(nv/2) , :int(nv/2)] = ddq_dq
        data.Fx[:int(nv/2), int(nv/2):nv] = np.dot(data.Minv,data.K)
        data.Fx[:int(nv/2), nv:-int(nv/2)] = ddq_dv
        data.Fx[int(nv/2):, :int(nv/2)] = np.dot(data.Binv,data.K)
        data.Fx[int(nv/2):, int(nv/2):nv] = -np.dot(data.Binv,data.K)
        
        data.Fu[int(nv/2):, :] = data.Binv

        # Computing the cost derivatives
        self.costs.calcDiff(data.costs, x, u)


    def createData(self):
        data = DifferentialFreeASRFwdDynamicsData(self)
        return data

class DifferentialFreeASRFwdDynamicsData(crocoddyl.DifferentialActionDataAbstract):
    def __init__(self, model):
        crocoddyl.DifferentialActionDataAbstract.__init__(self, model)
        self.pinocchio = pinocchio.Model.createData(model.state.pinocchio)
        self.multibody = crocoddyl.DataCollectorMultibody(self.pinocchio)
        self.actuation = model.actuation.createData()
        self.costs = model.costs.createData(self.multibody)
        self.costs.shareMemory(self)
        self.Minv = None
        self.Binv = None
        #sea terms
        self.K = 1e-1*np.eye(int(model.state.nv/2))
        self.B = 1e-4*np.eye(int(model.state.nv/2))
