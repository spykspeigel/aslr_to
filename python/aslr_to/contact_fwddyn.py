###################################3
####
####Deprecated code
####
#####################################



import numpy as np
import pinocchio
import crocoddyl

class DifferentialContactASLRFwdDynModel(crocoddyl.DifferentialActionModelAbstract):
    def __init__(self, state, actuation, contacts, costs, constraints=None):
        nu =  actuation.nu 
        crocoddyl.DifferentialActionModelAbstract.__init__(self, state, nu, costs.nr)

        self.actuation = actuation
        self.costs = costs
        self.contacts = contacts

    def calc(self, data, x, u):
        if len(x) != self.state.nx:
            raise Exception("Invalid argument: u has wrong dimension (it should be " + self.state.nx)
        if len(u) != self.nu:
            raise Exception("Invalid argument: u has wrong dimension (it should be " + self.nu)
        nq, nv, nu = self.state.nq, self.state.nv, self.actuation.nu
        nc = self.contacts.nc
        q, v = x[:nq], x[nq:]
        tau_couple = np.dot(data.K,q)

        pinocchio.computeAllTerms(self.state.pinocchio, data.multibody.pinocchio, q, v)
        pinocchio.computeCentroidalDynamics(self.state.pinocchio, data.multibody.pinocchio,q ,v)

        self.actuation.calc(data.multibody.actuation, x, u)
        self.contacts.calc(data.multibody.contacts, x)

        JMinvJt_damping_=0
        pinocchio.forwardDynamics(self.state.pinocchio, data.multibody.pinocchio, data.multibody.actuation.tau - tau_couple, data.multibody.contacts.Jc[:nc,:],
                        data.multibody.contacts.a0[:nc], JMinvJt_damping_)

        data.xout[:] = data.multibody.pinocchio.ddq

        self.contacts.updateAcceleration(data.multibody.contacts, data.multibody.pinocchio.ddq)
        self.contacts.updateForce(data.multibody.contacts, data.multibody.pinocchio.lambda_c)

        self.costs.calc(data.costs, x, u)
        data.cost = data.costs.cost


    def calcDiff(self, data, x, u):
        if len(x) != self.state.nx:
            raise Exception("Invalid argument: u has wrong dimension (it should be " + self.state.nx)
        if len(u) != self.nu:
            raise Exception("Invalid argument: u has wrong dimension (it should be " + self.nu)
        nq, nv, nu = self.state.nq, self.state.nv, self.actuation.nu
        q, v = x[:nq], x[nq:]
        nc=self.contacts.nc
        pinocchio.computeRNEADerivatives(self.state.pinocchio, data.multibody.pinocchio, q, v, data.xout,
                                         data.multibody.contacts.fext)
        data.Kinv = pinocchio.getKKTContactDynamicMatrixInverse(self.state.pinocchio, data.multibody.pinocchio, data.multibody.contacts.Jc[:nc,:],
                                                    )
        self.actuation.calcDiff(data.multibody.actuation, x, u)
        self.contacts.calcDiff(data.multibody.contacts, x)

        a_partial_dtau = data.Kinv[:nv,:nv]
        a_partial_da = data.Kinv[:nv,-nc:]
        f_partial_dtau = data.Kinv[-nc:,:nv]
        f_partial_da = data.Kinv[-nc:,-nc:]
        data.Fx[:,:nv] = -np.dot(a_partial_dtau,data.multibody.pinocchio.dtau_dq + data.K[:,-nv:])
        data.Fx[:,nv:] = -np.dot(a_partial_dtau, data.multibody.pinocchio.dtau_dv)
        data.Fx -= np.dot(a_partial_da,data.multibody.contacts.da0_dx[:nc,:])
        data.Fx += np.dot(a_partial_dtau,data.multibody.actuation.dtau_dx)
        data.Fu = np.dot(a_partial_dtau, data.multibody.actuation.dtau_du)

        self.costs.calcDiff(data.costs, x, u)

    def createData(self):
        data = DifferentialContactASLRFwdDynData(self)
        return data

class DifferentialContactASLRFwdDynData(crocoddyl.DifferentialActionDataAbstract):
    def __init__(self, model):
        crocoddyl.DifferentialActionDataAbstract.__init__(self, model)
        pmodel = pinocchio.Model.createData(model.state.pinocchio)
        actuation = model.actuation.createData()
        contacts = model.contacts.createData(pmodel)
        self.multibody = crocoddyl.DataCollectorActMultibodyInContact(pmodel, actuation, contacts)
        self.costs = model.costs.createData(self.multibody)
        self.costs.shareMemory(self)
        self.Minv = None
        self.Kinv = None
        self.K = np.zeros([model.state.nv,model.state.nq])
        nu = model.actuation.nu
        self.K[-nu:,-nu:]= np.eye(nu)
        self.B = np.random.random([model.state.nv,model.state.nv])