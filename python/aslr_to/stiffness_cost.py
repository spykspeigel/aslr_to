import numpy as np
import crocoddyl

#linear cost for variable stiffness 

class CostModelStiffness(crocoddyl.CostModelAbstract):
    def __init__(self, state, nu, lamda, Kref=None):
        crocoddyl.CostModelAbstract.__init__(self, state, int(nu/2) , nu)
        self.Kref = Kref if Kref is not None else state.zero()
        self.nu_ = nu
        self.lamda = lamda
    
    def calc(self, data, x, u):
        K = u[int(self.nu_/2):]
        data.residual.r[:] = self.lamda*(K-self.Kref)
        data.cost = np.sum(data.residual.r)

    def calcDiff(self, data, x, u):
        data.residual.Ru[:, int(self.nu_/2):] = self.lamda*np.eye(int(self.nu_/2))
        data.Lu[int(self.nu_/2):] = self.lamda*np.ones(int(self.nu_/2))

    def createData(self, collector):
        return CostDataStiffness(self,collector)

class CostDataStiffness(crocoddyl.CostDataAbstract):
    def __init__(self, model, collector):
        crocoddyl.CostDataAbstract.__init__(self, model, collector)