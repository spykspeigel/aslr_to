from .statemultibody_aslr import (StateMultibodyASR)
from .contact_fwddyn import (DifferentialContactASLRFwdDynModel, DifferentialContactASLRFwdDynData)
from .free_fwddyn_aslr import (DifferentialFreeASRFwdDynamicsModel,DifferentialFreeASRFwdDynamicsData)
from .residual_frame_placement import (ResidualModelFramePlacementASR,ResidualDataFramePlacementASR)
from .integrated_action import (IntegratedActionModelEulerASR, IntegratedActionDataEulerASR)
from .actuation_aslr import ASRActuation
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
    for i in range(len(log.us)):
        u1_sqaured += log.us[i][0]**2
        u2_sqaured += log.us[i][1]**2
    return [u1_sqaured, u2_sqaured]

def plotOCSolution(xs=None, us=None, figIndex=1, show=True, figTitle=""):
    import matplotlib.pyplot as plt
    import numpy as np
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42

    # Getting the state and control trajectories
    if xs is not None:
        xsPlotIdx = 111
        nx = int(xs[0].shape[0]/4)
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