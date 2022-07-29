import numpy as np
import pinocchio
import crocoddyl


class DAM2(crocoddyl.DifferentialActionModelAbstract):
    def __init__(self, state, actuationModel, costModel, K=None, D=None):
        crocoddyl.DifferentialActionModelAbstract.__init__(self, state, actuationModel.nu, costModel.nr)
        self.actuation = actuationModel
        self.costs = costModel
        self.enable_force = True
        if K is None:
            self.K = 1e-1*np.eye(state.nv)
        else:
            self.K = K
        if D is None:
            self.D = 1e-3*np.eye(state.nv)
        else:
            self.D = D
    def calc(self, data, x, u=None):
        if u is None:
            u = np.zeros(self.nu)
        nq=self.state.nq
        nv=self.state.nv

        q = x[:nq]
        v = x[-nv:]


        self.actuation.calc(data.actuation, x, u)
        tau = data.actuation.tau


        # Computing the fwd dynamics manually
        pinocchio.computeAllTerms(self.state.pinocchio, data.pinocchio, q, v)
        data.M = data.pinocchio.M
        data.Minv = np.linalg.inv(data.M)

        data.xout[:] = np.dot(data.Minv, (tau - data.pinocchio.nle - np.dot(self.K,q)- np.dot(self.D,v)))


        # Computing the cost value and residuals
        pinocchio.forwardKinematics(self.state.pinocchio, data.pinocchio, q, v)
        pinocchio.updateFramePlacements(self.state.pinocchio, data.pinocchio)
        self.costs.calc(data.costs, x, u)
        data.cost = data.costs.cost

        # Computing the cost value and residuals
        self.costs.calc(data.costs, x, u)
        data.cost = data.costs.cost

    def calcDiff(self, data, x, u=None):
        if u is None:
            u = self.unone
        nq=self.state.nq
        nv=self.state.nv

        q = x[:nq]
        v = x[-nv:]


        # Computing the actuation derivatives
        self.actuation.calcDiff(data.actuation, x, u)


        # Computing the dynamics derivatives
        pinocchio.computeRNEADerivatives(self.state.pinocchio, data.pinocchio, q, v, data.xout[:])

        ddq_dq = np.dot(data.Minv, ( - data.pinocchio.dtau_dq - self.K))
        ddq_dv = np.dot(data.Minv, ( - data.pinocchio.dtau_dv- self.D))

        data.Fx[: , :nv] = ddq_dq 
        data.Fx[:,-nv:] = ddq_dv

        data.Fu[:] = np.dot(data.Minv,data.actuation.dtau_du[:])

        # Computing the cost derivatives
        self.costs.calcDiff(data.costs, x, u)

    def quasiStatic(self, data, x, maxiter, tol):

        nq=self.state.nq
        nv=self.state.nv

        q = x[:nq]
        v = x[-nv:]


        # Check the velocity input is zero
        try:
            x.tail(nv).isZero()
        except:
            print("The velocity input should be zero for quasi-static to work.")
        data.tmp_xstatic[:nq] = q
        data.tmp_xstatic[-nv:] *= 0
        data.tmp_ustatic[:] *= 0.

        pinocchio.rnea(self.state.pinocchio, data.multibody.pinocchio, q_l, data.tmp_xstatic[-nv:], data.tmp_xstatic[-nv:])
        self.actuation.calc(data.actuation, data.tmp_xstatic, data.tmp_ustatic)
        self.actuation.calcDiff(data.actuation, data.tmp_xstatic, data.tmp_ustatic)
        data.tmp_ustatic = np.dot(np.linalg.pinv(data.actuation.dtau_du).T, data.multibody.pinocchio.tau)
        return data.tmp_ustatic 


    def createData(self):
        data = DAD2(self)
        return data

class DAD2(crocoddyl.DifferentialActionDataAbstract):
    def __init__(self, model):
        crocoddyl.DifferentialActionDataAbstract.__init__(self, model)
        self.pinocchio = pinocchio.Model.createData(model.state.pinocchio)
        self.multibody = crocoddyl.DataCollectorMultibody(self.pinocchio)
        self.actuation = model.actuation.createData()
        self.costs = model.costs.createData(self.multibody)
        self.costs.shareMemory(self)
        self.Minv = None
        self.Binv = None
        self.tmp_xstatic = np.zeros(model.state.nx)
        self.tmp_ustatic = np.zeros(model.nu)

