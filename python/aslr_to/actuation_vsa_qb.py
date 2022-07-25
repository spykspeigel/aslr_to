###################################3
####
####Deprecated code
####
#####################################



import numpy as np
import pinocchio
import crocoddyl

#   Actuator model for qbMove VSA 

class QbActuationModel(crocoddyl.ActuationModelAbstract):
    def __init__(self, state):
        crocoddyl.ActuationModelAbstract.__init__(self, state, state.nv)
        self.nv = self.state.nv

    def calc(self, data, x, u):
        r = u[:int(self.nu/2)]
        d = u[int(self.nu/2):]
        nq=self.state.nq
        nv=self.state.nv
        nq_l = int(nq/2)
        nv_l = int(nv/2)
        q_l = x[:nv_l]
        a = 8.9992
        k = 0.0019

        data.tau[nv_l:] = 2*k*np.cosh(a*d)*np.sinh(a*(x[:nv_l]-r))
        # data.K = 2*a*k*np.cosh(a*d)*np.cosh(a*(q_l-r))
        data.K = 2*a*k*np.cosh(a*d)
    def calcDiff(self, data, x, u):
        # r,d are the output variables
        # q are the input variables
        # a1, a2, k1, k2 are the parameters

        nq=self.state.nq
        nv=self.state.nv
        nq_l = int(nq/2)
        nv_l = int(nv/2)
        q_l = x[:nq_l]
        
        r = u[:int(self.nu/2)]
        d = u[int(self.nu/2):]

        a = 8.9992
        k = 0.0019        
        t2 = d*a
        t3 = (q_l-r)*a
        t4 = np.cosh(t2)
        t5 = np.sinh(t3)
        t6 = 2*a*a*k*t4*t5
        
        # data.dK_du[:,:int(self.state.nv/2)] = -np.diag(t6)
        # data.dK_du[:,int(self.state.nv/2):] = np.diag(2*a*a*k*np.cosh(t3)*np.sinh(t2))
        # data.dK_dx[:,:nq_l] = np.diag(t6)

        data.dK_du[:,int(self.state.nv/2):] = 2*a*a*k*np.diag(np.sinh(d*a))

        # data.dtau_du[nv_l:,:int(self.state.nv/2)] = -np.diag(2*a*k*np.cosh(t3)*np.cosh(t2))
        # data.dtau_du[nv_l:,int(self.state.nv/2):] = np.diag(2*a*k*np.sinh(t2)*np.sinh(t3))
        # data.dtau_dx[nv_l:,:nq_l] = np.diag(2*a*k*np.cosh(t3)*np.cosh(t2))

        data.dtau_du[nv_l:,:int(self.state.nv/2)] = -2*a*k*np.diag(np.cosh(t3)*np.cosh(t2))
        data.dtau_du[nv_l:,int(self.state.nv/2):] = 2*a*k*np.diag(np.sinh(t2)*np.sinh(t3))
        # print(np.cosh(a*d))
        # print(np.cosh(a*(q_l-r)))
        # print(np.diag(np.cosh(a*d)*np.cosh(a*(q_l-r))))
        data.dtau_dx[nv_l:,:nv_l] = 2*a*k*np.diag(np.cosh(t3)*np.cosh(t2))

    def createData(self):
        data = QbActuationData(self)
        return data

class QbActuationData(crocoddyl.ActuationDataAbstract):
    def __init__(self, model):
        crocoddyl.ActuationDataAbstract.__init__(self, model)

        self.K = np.zeros(model.nu)
        self.dK_du = np.zeros([int(model.nu/2), model.nu])
        self.dK_dx = np.zeros([int(model.nu/2), 2*model.nv])
