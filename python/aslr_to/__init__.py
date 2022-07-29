from .statemultibody_aslr import (StateMultibodyASR)
from .contact_fwddyn import (DifferentialContactASLRFwdDynModel, DifferentialContactASLRFwdDynData)
from .free_fwddyn_asr import (DifferentialFreeASRFwdDynamicsModel,DifferentialFreeASRFwdDynamicsData)
from .free_fwddyn_vsa import (DifferentialFreeFwdDynamicsModelVSA, DifferentialFreeFwdDynamicsDataVSA)
from .free_fwddyn_fishing_2 import (DAM2, DAD2)
from .free_fwddyn_vsa_qb import (DifferentialFreeFwdDynamicsModelQb, DifferentialFreeFwdDynamicsDataQb)
from .residual_frame_placement import (ResidualModelFramePlacementASR,ResidualDataFramePlacementASR)
from .integrated_action import (IntegratedActionModelEulerASR, IntegratedActionDataEulerASR)
from .stiffness_cost import (CostModelStiffness, CostDataStiffness)
from .actuation_vsa import VSAASRActuation
from .actuation_aslr import ASRActuation
from .actuation_condensed import ASRActuationCondensed
from .actuation_fishing import ASRFishing
from .actuation_vsa_qb import (QbActuationModel,QbActuationData)
from .soft_residual_model import (SoftDynamicsResidualModel,SoftDynamicsResidualData)
from .vsa_dynamics_residual import (VSADynamicsResidualModel, VSADynamicsResidualData)
from .residual_acrobot import (ResidualModelDoublePendulum,ResidualDataDoublePendulum)
from .solver import DDPASLR
import numpy as np
import crocoddyl
import pinocchio
import time
import warnings
import matplotlib.pyplot as plt

def plot_theta(log, K):
    for i in [-1]:
        for k in range(len(log.residual[i][0])):
            theta = []
            for j in range(len(log.residual[i])):
                theta.append(log.residual[i][j][k])
            plt.xlabel('Nodes')
            plt.ylabel( "theta ")
            plt.title('')
            plt.plot(theta,label="theta_"+str(k))
            plt.legend()
        plt.show()

def plot_stiffness(us):
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42

    nu = int(us[0].shape[0])
    U = [0.] * nu

    for i in range(int(nu/2),nu):
        U[i] = [np.asscalar(u[i]) if u.shape[0] != 0 else 0 for u in us]


    [plt.plot(U[i], label="K" + str(i-2)) for i in range(int(nu/2),nu)]
    plt.legend()
    
    plt.xlabel("knots")
    plt.ylabel( "Stiffness [Nm/rad]")
    plt.title('', fontsize=14)
    plt.show()

def plotKKTerror(fs, figIndex=1):
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42
    plt.figure(figIndex, figsize=(6.4, 8))

    # Plotting the feasibility
    plt.ylabel("KKT error")
    plt.plot(fs, label = "KKT error")
    # plt.title(figTitle, fontsize=14)
    plt.xlabel("iteration")
    plt.yscale('log')
    # plt.legend(["dynamic", "equality", "total"])
    plt.show()

def u_squared(log):
    u_sqaured = np.zeros(len(log.us[0]))
    for i in range(len(log.us)):
        for j in range(len(log.us[0])):
            u_sqaured[j] += log.us[i][j]**2
    return u_sqaured

def plotrigidOCSolution(xs=None, us=None, figIndex=1, show=True, figTitle=""):
    import matplotlib.pyplot as plt
    import numpy as np
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42

    # Getting the state and control trajectories
    if xs is not None:
        xsPlotIdx = 111
        nx = int(xs[0].shape[0]/2)
        print(nx)
        X = [0.] * nx
        for i in range(nx):
            X[i] = [np.asscalar(x[i]) for x in xs]
    if us is not None:
        usPlotIdx = 111
        nu = int(us[0].shape[0])
        U = [0.] * nu
        for i in range(nu):
            U[i] = [np.asscalar(u[i]) if u.shape[0] != 0 else 0 for u in us]
    if xs is not None and us is not None:
        xsPlotIdx = 211
        usPlotIdx = 212
    plt.figure(figIndex)

    # # Plotting the state trajectories
    if xs is not None:
        plt.subplot(xsPlotIdx)
        [plt.plot(X[i], label="q" + str(i)) for i in range(nx)]
        plt.ylabel("Joint Positions [rad]")
        plt.xlabel("Knots")
        plt.legend()
        plt.title(figTitle, fontsize=14)

    # Plotting the control commands
    if us is not None:
        plt.subplot(usPlotIdx)
        [plt.plot(U[i], label="u" + str(i)) for i in range(nu)]
        plt.ylabel("Input [Nm]")
        plt.legend()
        plt.xlabel("knots")

    if show:
        plt.show()

def plotSEAOCSolution(xs=None, us=None, figIndex=1, show=True, figTitle=""):
    import matplotlib.pyplot as plt
    import numpy as np
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42

    # Getting the state and control trajectories
    if xs is not None:
        xsPlotIdx = 111
        nx = int(xs[0].shape[0]/4)
        print(nx)
        X = [0.] * nx
        for i in range(nx):
            X[i] = [np.asscalar(x[i]) for x in xs]
    if us is not None:
        usPlotIdx = 111
        nu = int(us[0].shape[0])
        U = [0.] * nu
        for i in range(nu):
            U[i] = [np.asscalar(u[i]) if u.shape[0] != 0 else 0 for u in us]
    if xs is not None and us is not None:
        xsPlotIdx = 211
        usPlotIdx = 212
    plt.figure(figIndex)

    # # Plotting the state trajectories
    if xs is not None:
        plt.subplot(xsPlotIdx)
        [plt.plot(X[i], label="q" + str(i)) for i in range(nx)]
        plt.ylabel("Joint Positions [rad]")
        plt.xlabel("Knots")
        plt.legend()
        plt.title(figTitle, fontsize=14)

    # Plotting the control commands
    if us is not None:
        plt.subplot(usPlotIdx)
        [plt.plot(U[i], label="u" + str(i)) for i in range(nu)]
        plt.ylabel("Input [Nm]")
        plt.legend()
        plt.xlabel("knots")

    if show:
        plt.show()

def plotOCSolution(xs=None, us=None, stiffness= False, figIndex=1, show=True, figTitle=""):
    import matplotlib.pyplot as plt
    import numpy as np
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42

    # Getting the state and control trajectories
    if xs is not None:
        xsPlotIdx = 111
        nx = int(xs[0].shape[0]/4)
        print(nx)
        X = [0.] * nx
        for i in range(nx):
            X[i] = [np.asscalar(x[i]) for x in xs]
    if us is not None:
        usPlotIdx = 111
        nu = int(us[0].shape[0]/2)
        U = [0.] * nu
        for i in range(nu):
            U[i] = [np.asscalar(u[i]) if u.shape[0] != 0 else 0 for u in us]
    if xs is not None and us is not None:
        xsPlotIdx = 211
        usPlotIdx = 212
    plt.figure(figIndex)
    if stiffness:
        xsPlotIdx = 311
        usPlotIdx = 312
        stiffPlotIdx = 313
    # # Plotting the state trajectories
    if xs is not None:
        plt.subplot(xsPlotIdx)
        [plt.plot(X[i], label="q" + str(i)) for i in range(nx)]
        plt.ylabel("Joint Positions [rad]")
        plt.xlabel("Knots")
        plt.legend()
        plt.title(figTitle, fontsize=14)

    # Plotting the control commands
    if us is not None:
        plt.subplot(usPlotIdx)
        [plt.plot(U[i], label="u" + str(i)) for i in range(nu)]
        plt.ylabel("Input [Nm]")
        plt.legend()
        plt.xlabel("knots")

    if (stiffness):
        nu = int(us[0].shape[0])
        U = [0.] * nu
        for i in range(int(nu/2),nu):
            U[i] = [np.asscalar(u[i]) if u.shape[0] != 0 else 0 for u in us]


        plt.subplot(stiffPlotIdx)
        [plt.plot(U[i], label="K" + str(i-2)) for i in range(int(nu/2),nu)]
        plt.legend()
        
        plt.xlabel("knots")
        plt.ylabel( "Stiffness [Nm/rad]")
        plt.title('', fontsize=14)

    if show:
        plt.show()

class CostModelDoublePendulum(crocoddyl.CostModelAbstract):
    def __init__(self, state, activation, nu):
        activation = activation if activation is not None else crocoddyl.ActivationModelQuad(state.ndx)
        crocoddyl.CostModelAbstract.__init__(self, state, activation, nu=nu)

    def calc(self, data, x, u):
        c1, c2 = np.cos(x[0]), np.cos(x[1])
        s1, s2 = np.sin(x[0]), np.sin(x[1])
        data.residual.r[:] = np.array([s1, s2, 1 + c1, 1 + c2, x[4], x[5]])
        self.activation.calc(data.activation, data.residual.r)
        data.cost = data.activation.a_value

    def calcDiff(self, data, x, u):
        c1, c2 = np.cos(x[0]), np.cos(x[1])
        s1, s2 = np.sin(x[0]), np.sin(x[1])

        self.activation.calcDiff(data.activation, data.residual.r)

        data.residual.Rx[:2, :2] = np.diag([c1, c2])
        data.residual.Rx[2:4, :2] = -np.diag([s1, s2])
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
    def __init__(self, state, actLink, nu):
        crocoddyl.ActuationModelAbstract.__init__(self, state, nu)
        self.nv = state.nv
        self.actLink = actLink

    def calc(self, data, x, u):
        data.tau[:] = np.dot(data.S, u[:int(self.nu)])

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
            self.S[-1,-1] = 1
        else:
            self.S[int(model.nv/2),0] = 1
            # self.S[int(model.nv/2):8,:1] = np.eye(1)
            # print(self.S)
