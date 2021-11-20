import numpy as np
import pinocchio
import crocoddyl

# this files only for testing purpose and not included 
class DifferentialFreeFwdDynamicsModel_1(crocoddyl.DifferentialActionModelAbstract):
    def __init__(self, state, actuationModel, costModel):
        crocoddyl.DifferentialActionModelAbstract.__init__(self, state, actuationModel.nu, costModel.nr)
        self.actuation = actuationModel
        self.costs = costModel

    def calc(self, data, x, u=None):
        if u is None:
            u = self.unone
        q, v = x[:self.state.nq], x[-self.state.nv:]
        self.actuation.calc(data.actuation, x, u)
        tau = data.actuation.tau

        tau_couple = np.dot(data.K,q)
        pinocchio.computeAllTerms(self.state.pinocchio, data.pinocchio, q, v)
        data.M = data.pinocchio.M
        data.Minv = np.linalg.inv(data.M)

        data.xout[:] = np.dot(data.Minv, (tau - data.pinocchio.nle-tau_couple))
        # Computing the cost value and residuals
        pinocchio.forwardKinematics(self.state.pinocchio, data.pinocchio, q, v)
        pinocchio.updateFramePlacements(self.state.pinocchio, data.pinocchio)
        self.costs.calc(data.costs, x, u)
        data.cost = data.costs.cost

    def calcDiff(self, data, x, u=None):
        if u is None:
            u = self.unone
        nq, nv = self.state.nq, self.state.nv
        q, v = x[:nq], x[-nv:]
        # Computing the actuation derivatives
        self.actuation.calcDiff(data.actuation, x, u)
        tau = data.actuation.tau
        # Computing the dynamics derivatives
        tau_couple = np.dot(data.K,q)
        pinocchio.computeRNEADerivatives(self.state.pinocchio, data.pinocchio, q, v, data.xout)
        ddq_dq = np.dot(data.Minv, (data.actuation.dtau_dx[:, :nv] - data.pinocchio.dtau_dq -data.K))
        ddq_dv = np.dot(data.Minv, (data.actuation.dtau_dx[:, nv:] - data.pinocchio.dtau_dv))
        data.Fx[:, :] = np.hstack([ddq_dq, ddq_dv])
        data.Fu[:, :] = np.dot(data.Minv, data.actuation.dtau_du)
        # Computing the cost derivatives
        self.costs.calcDiff(data.costs, x, u)

    def createData(self):
        data = DifferentialFreeFwdDynamicsData_1(self)
        return data

class DifferentialFreeFwdDynamicsData_1(crocoddyl.DifferentialActionDataAbstract):
    def __init__(self, model):
        crocoddyl.DifferentialActionDataAbstract.__init__(self, model)
        self.pinocchio = pinocchio.Model.createData(model.state.pinocchio)
        self.multibody = crocoddyl.DataCollectorMultibody(self.pinocchio)
        self.actuation = model.actuation.createData()
        self.costs = model.costs.createData(self.multibody)
        self.costs.shareMemory(self)
        self.Minv = None
        self.K = np.ones([model.state.nv,model.state.nv])
        # self.K = np.diag([1,2,4,5,6,7,4])
        self.B = np.random.random([model.state.nv,model.state.nv])