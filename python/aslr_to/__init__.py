from .statemultibody_aslr import (StateMultibodyASR)
from .contact_fwddyn import (DifferentialContactASLRFwdDynModel, DifferentialContactASLRFwdDynData)
from .free_fwddyn_aslr import (DifferentialFreeASRFwdDynamicsModel,DifferentialFreeASRFwdDynamicsData)
from .free_fwddyn_vsa import (DifferentialFreeFwdDynamicsModelVSA, DifferentialFreeFwdDynamicsDataVSA)
from .residual_frame_placement import (ResidualModelFramePlacementASR,ResidualDataFramePlacementASR)
from .integrated_action import (IntegratedActionModelEulerASR, IntegratedActionDataEulerASR)
from .stiffness_cost import (CostModelStiffness, CostDataStiffness)
from .vsa_asr_actuation import VSAASRActuation
from .actuation_aslr import ASRActuation
from .actuation_condensed import ASRActuationCondensed
from .soft_residual_model import (SoftDynamicsResidualModel,SoftDynamicsResidualData)
from .solver import DDPASLR
import numpy as np
import crocoddyl
import pinocchio
import time
import warnings
import matplotlib.pyplot as plt

def u_squared(log):
    u1_sqaured = 0
    u2_sqaured = 0
    u3_sqaured = 0
    u4_sqaured = 0
    for i in range(len(log.us)):
        u1_sqaured += log.us[i][0]**2
        u2_sqaured += log.us[i][1]**2
        try:
            u3_sqaured += log.us[i][2]**2
            u4_sqaured += log.us[i][3]**2
        except:
            return [u1_sqaured, u2_sqaured]        
    return [u1_sqaured, u2_sqaured, u3_sqaured, u4_sqaured]

def plotOCSolution(xs=None, us=None, figIndex=1, show=True, figTitle=""):
    import matplotlib.pyplot as plt
    import numpy as np
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42

    # Getting the state and control trajectories
    if xs is not None:
        xsPlotIdx = 111
        nx = int(xs[0].shape[0]/2)
        X = [0.] * nx
        for i in range(nx):
            X[i] = [np.asscalar(x[i]) for x in xs]
    if us is not None:
        usPlotIdx = 111
        nu = us[0].shape[0]
        U = [0.] * nu
        for i in range(nu):
            U[i] = [np.asscalar(u[i]) if u.shape[0] != 0 else 0 for u in us]
    if xs is not None and us is not None:
        xsPlotIdx = 211
        usPlotIdx = 212
    plt.figure(figIndex)

    # Plotting the state trajectories
    if xs is not None:
        plt.subplot(xsPlotIdx)
        [plt.plot(X[i], label="x" + str(i)) for i in range(nx)]
        plt.legend()
        plt.title(figTitle, fontsize=14)

    # Plotting the control commands
    if us is not None:
        plt.subplot(usPlotIdx)
        [plt.plot(U[i], label="u" + str(i)) for i in range(nu)]
        plt.legend()
        plt.xlabel("knots")
    if show:
        plt.show()

class CostModelDoublePendulum(crocoddyl.CostModelAbstract):
    def __init__(self, state, activation, nu):
        activation = activation if activation is not None else crocoddyl.ActivationModelQuad(state.ndx)
        crocoddyl.CostModelAbstract.__init__(self, state, activation, nu=nu)

    def calc(self, data, x, u):
        c1, c2 = np.cos(x[0]), np.cos(x[1])
        s1, s2 = np.sin(x[0]), np.sin(x[1])
        data.residual.r[:] = np.array([s1, s2, 1 - c1, 1 - c2, x[4], x[5]])
        self.activation.calc(data.activation, data.residual.r)
        data.cost = data.activation.a_value

    def calcDiff(self, data, x, u):
        c1, c2 = np.cos(x[0]), np.cos(x[1])
        s1, s2 = np.sin(x[0]), np.sin(x[1])

        self.activation.calcDiff(data.activation, data.residual.r)

        data.residual.Rx[:2, :2] = np.diag([c1, c2])
        data.residual.Rx[2:4, :2] = np.diag([s1, s2])
        data.residual.Rx[4:6, 4:6] = np.diag([1, 1])
        data.Lx[:] = np.dot(data.residual.Rx.T, data.activation.Ar)

        data.Rxx[:2, :2] = np.diag([c1**2 - s1**2, c2**2 - s2**2])
        data.Rxx[2:4, :2] = np.diag([s1**2 + (1 - c1) * c1, s2**2 + (1 - c2) * c2])
        data.Rxx[4:6, 4:6] = np.diag([1, 1])
        data.Lxx[:, :] = np.diag(np.dot(data.Rxx.T, np.diag(data.activation.Arr)))

    def createData(self, collector):
        data = CostDataDoublePendulum(self, collector)
        return data


class CostDataDoublePendulum(crocoddyl.CostDataAbstract):
    def __init__(self, model, collector):
        crocoddyl.CostDataAbstract.__init__(self, model, collector)
        self.Rxx = np.zeros((6, 8))


class ActuationModelDoublePendulum(crocoddyl.ActuationModelAbstract):
    def __init__(self, state, actLink):
        crocoddyl.ActuationModelAbstract.__init__(self, state, 2)
        self.nv = state.nv
        self.actLink = actLink

    def calc(self, data, x, u):
        data.tau[:] = np.dot(data.S, u)

    def calcDiff(self, data, x, u):
        data.dtau_du[:] = data.S

    def createData(self):
        data = ActuationDataDoublePendulum(self)
        return data


class ActuationDataDoublePendulum(crocoddyl.ActuationDataAbstract):
    def __init__(self, model):
        crocoddyl.ActuationDataAbstract.__init__(self, model)
        if model.nu == 1:
            self.S = np.zeros([model.nv,1])
        else:
            self.S = np.zeros((model.nv, model.nu))
    
        if model.actLink == 1:
            self.S[-model.nu,-model.nu]=1
        else:
            self.S[0,0]=1

