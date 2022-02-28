import numpy as np
import pinocchio
import crocoddyl
import aslr_to
class DifferentialContactFwdDynModelRigid(crocoddyl.DifferentialActionModelAbstract):
    def __init__(self, state, actuation, contacts, costs, constraints=None):
        crocoddyl.DifferentialActionModelAbstract.__init__(self, state, actuation.nu, costs.nr)
        self.actuation = actuation  
        self.costs = costs
        self.contacts = contacts

    def calc(self, data, x, u):
        if len(x) != self.state.nx:
            raise Exception("Invalid argument: u has wrong dimension (it should be " + str(self.state.nx))
        if len(u) != self.nu:
            raise Exception("Invalid argument: u has wrong dimension (it should be " + str(self.nu)+"it is "+ str(len(u)))

        nq, nv, nc = self.state.nq, self.state.nv, self.contacts.nc
        q = x[:nq]
        v = x[nq:]
        
        # pinocchio.updateFramePlacements(self.state.pinocchio, data.multibody.pinocchio)
        pinocchio.computeAllTerms(self.state.pinocchio, data.multibody.pinocchio, q, v)
        pinocchio.computeCentroidalMomentum(self.state.pinocchio, data.multibody.pinocchio, q, v)
        self.actuation.calc(data.multibody.actuation, x, u)
        self.contacts.calc(data.multibody.contacts, x)
        tau = data.multibody.actuation.tau

        theta_dot_dot = u[nv:2*nv]

        data.multibody.motor.theta[:] = np.dot(np.diag(data.multibody.motor.K), q[-nv:]) + tau -theta_dot_dot
        data.multibody.motor.theta_ddot = theta_dot_dot

        JMinvJt_damping_=0
        pinocchio.forwardDynamics(self.state.pinocchio, data.multibody.pinocchio, tau, data.multibody.contacts.Jc[:nc,:],
                        data.multibody.contacts.a0, JMinvJt_damping_)
        data.xout[:] = data.multibody.pinocchio.ddq
        self.contacts.updateAcceleration(data.multibody.contacts, data.xout)
        self.contacts.updateForce(data.multibody.contacts, data.multibody.pinocchio.lambda_c)
        self.costs.calc(data.costs, x, u)
        data.cost = data.costs.cost

    def calcDiff(self, data, x, u):
        nq, nv, nc = self.state.nq, self.state.nv, self.contacts.nc

        q = x[:nq]
        v = x[nq:]
        nu = self.actuation.nu

        pinocchio.computeRNEADerivatives(self.state.pinocchio, data.multibody.pinocchio, q, v, data.xout,
                                         data.multibody.contacts.fext)
        data.Kinv[:, :] = pinocchio.getKKTContactDynamicMatrixInverse(self.state.pinocchio, data.multibody.pinocchio, data.multibody.contacts.Jc[:nc,:])

        self.actuation.calcDiff(data.multibody.actuation, x, u)
        self.contacts.calcDiff(data.multibody.contacts, x)

        #Extracting the TopLeft corner block diagonal matrix
        a_partial_dtau = data.Kinv[:nv,:nv]
        a_partial_da = data.Kinv[:nv,-nc:]
        f_partial_dtau = data.Kinv[nv:,:nv]
        f_partial_da = data.Kinv[nv:,-nc:]

        data.Fx[:, :nv] = -np.dot(a_partial_dtau, data.multibody.pinocchio.dtau_dq)
        data.Fx[:, nv:] = -np.dot(a_partial_dtau, data.multibody.pinocchio.dtau_dv)
        data.Fx[:, :] -= np.dot(a_partial_da, data.multibody.contacts.da0_dx[:nc,:])
        data.Fu[:, :] = np.dot(a_partial_dtau, data.multibody.actuation.dtau_du)

        #computing the jacobian of contact forces (required with contact dependent costs)
        data.df_dx[:nc, :nv] = np.dot(f_partial_dtau, data.multibody.pinocchio.dtau_dq)
        data.df_dx[:nc, nv:] = np.dot(f_partial_dtau, data.multibody.pinocchio.dtau_dv)
        data.df_dx[:nc, :] += np.dot(f_partial_da, data.multibody.contacts.da0_dx[:nc,:])
        data.df_dx[:nc, :] -= np.dot(f_partial_dtau, data.multibody.actuation.dtau_dx[:, :])
        data.df_du[:nc, :] = -np.dot(f_partial_dtau, data.multibody.actuation.dtau_du[:, :])
        self.contacts.updateAccelerationDiff(data.multibody.contacts, data.Fx[:nv, :])
        self.contacts.updateForceDiff(data.multibody.contacts, data.df_dx, data.df_du)

        self.costs.calcDiff(data.costs, x, u)

    def createData(self):
        data = DifferentialContactFwdDynDataRigid(self)
        return data

class DifferentialContactFwdDynDataRigid(crocoddyl.DifferentialActionDataAbstract):
    def __init__(self, model):
        crocoddyl.DifferentialActionDataAbstract.__init__(self, model)
        self.pinocchio = pinocchio.Model.createData(model.state.pinocchio)
        motor = aslr_to.SoftModelData(model.state.nv-6)
        self.multibody = aslr_to.DataCollectorSoftActMultibodyInContact(self.pinocchio, model.actuation.createData(), motor, model.contacts.createData(self.pinocchio))
        nx, ndx, nv, nu, nc = model.state.nx, model.state.ndx, model.state.nv, model.nu, model.contacts.nc
        self.Kinv = np.zeros((nv + nc, nv + nc))
        self.df_dx = np.zeros([nc, ndx])
        self.df_du = np.zeros([nc, nu])
        self.tmp_xstatic = np.zeros(nx)
        self.tmp_ustatic = np.zeros(nu)
        self.tmp_Jstatic = np.zeros([nv, nu + nc])
        self.costs = model.costs.createData(self.multibody)
        self.costs.shareMemory(self)