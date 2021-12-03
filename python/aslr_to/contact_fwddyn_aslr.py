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
            raise Exception("Invalid argument: u has wrong dimension (it should be " + str(self.state.nx))
        if len(u) != self.nu:
            raise Exception("Invalid argument: u has wrong dimension (it should be " + str(self.nu)+"it is "+ str(len(u)))

        nu = self.actuation.nu
        nc = self.contacts.nc
        nq_l = self.state.nq_l
        nv_l = self.state.nv_l
        nq = self.state.nq
        nv = nv_l +self.state.nv_m

        q_l = x[:nq_l]
        q_m = x[nq_l:nq]
        v_l = x[nq:-self.state.nv_m]
        v_m = x[-self.state.nv_m:]

        x_l = np.hstack([q_l,v_l])
        data.tau_couple = np.dot(data.K, q_l-np.hstack([np.zeros(7),q_m]))

        pinocchio.computeAllTerms(self.state.pinocchio, data.multibody.pinocchio, q_l, v_l)
        pinocchio.computeCentroidalDynamics(self.state.pinocchio, data.multibody.pinocchio, q_l, v_l)

        self.actuation.calc(data.multibody.actuation, x, u)

        self.contacts.calc(data.multibody.contacts, x_l)
        data.Binv = np.linalg.inv(data.B)
        tau = data.multibody.actuation.tau
        JMinvJt_damping_=0
        pinocchio.forwardDynamics(self.state.pinocchio, data.multibody.pinocchio, - data.tau_couple, data.multibody.contacts.Jc[:nc,:self.state.nv_l],
                        data.multibody.contacts.a0[:nc], JMinvJt_damping_)

        data.xout[:nv_l] = data.multibody.pinocchio.ddq
        data.xout[nv_l:] =  np.dot(data.Binv, tau[-self.state.nv_m:] + data.tau_couple[-self.state.nv_m:])
        self.contacts.updateAcceleration(data.multibody.contacts, data.xout)

        self.contacts.updateForce(data.multibody.contacts, data.multibody.pinocchio.lambda_c)
        self.costs.calc(data.costs, x[:self.state.nx], u)
        data.cost = data.costs.cost

    def calcDiff(self, data, x, u):

        nu = self.actuation.nu
        nc = self.contacts.nc
        nq_l = self.state.nq_l
        nv_l = self.state.nv_l
        nq = self.state.nq
        nv = self.state.nv

        q_l = x[:nq_l]
        q_m = x[nq_l:nq]
        v_l = x[nq:-self.state.nv_m]
        v_m = x[-self.state.nv_m:]

        x_l = np.hstack([q_l,v_l])

        pinocchio.computeRNEADerivatives(self.state.pinocchio, data.multibody.pinocchio, q_l, v_l, data.xout[:nv_l],
                                         data.multibody.contacts.fext)
        data.Kinv = pinocchio.getKKTContactDynamicMatrixInverse(self.state.pinocchio, data.multibody.pinocchio, data.multibody.contacts.Jc[:nc,:])
        self.actuation.calcDiff(data.multibody.actuation, x, u)
        self.contacts.calcDiff(data.multibody.contacts, x_l)

        a_partial_dtau = data.Kinv[:nv_l,:nv_l]
        a_partial_da = data.Kinv[:nv_l,-nc:]
        f_partial_dtau = data.Kinv[-nc:,:nv_l]
        f_partial_da = data.Kinv[-nc:,-nc:]

        data.Fx[:nv_l,:nv_l] = -np.dot(a_partial_dtau,data.multibody.pinocchio.dtau_dq + data.K[:,-nv_l:])
        data.Fx[:nv_l,nv:-self.state.nv_m] = -np.dot(a_partial_dtau, data.multibody.pinocchio.dtau_dv)

        data.Fx[:nv_l,nv_l:nv] = np.dot(a_partial_dtau,data.K[:,-self.state.nv_m:])

        data.Fx[:nv_l,:nv_l] -=   np.dot(a_partial_da, data.multibody.contacts.da0_dx[:nc,:nv_l])
        data.Fx[:nv_l,2*nv_l:-self.state.nv_m] -= np.dot(a_partial_da, data.multibody.contacts.da0_dx[:nc,2*nv_l:-self.state.nv_m])

        #data.Fx += np.dot(a_partial_dtau,data.multibody.actuation.dtau_dx)
        data.Fu[nv_l:, ] = np.dot(data.Binv, data.multibody.actuation.dtau_du[nv_l:, :])
        # print(data.Fu)
        self.costs.calcDiff(data.costs, x[:self.state.nx], u)

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
        #print(self.multibody.contacts.Jc.shape)
        self.costs = model.costs.createData(self.multibody)
        self.costs.shareMemory(self)
        self.Minv = None
        self.Kinv = None
        self.K = np.zeros([model.state.pinocchio.nv,model.state.pinocchio.nq])
        nu = model.actuation.nu
        self.K[-nu:,-nu:]= np.eye(nu)
        self.B = np.eye(model.state.nv_m)
