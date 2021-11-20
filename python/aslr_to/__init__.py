from .statemultibody_aslr import (StateMultibodyASLR)
from .contact_fwddyn import (DifferentialContactASLRFwdDynModel, DifferentialContactASLRFwdDynData)
from .free_fwddyn_aslr import (DifferentialFreeASLRFwdDynamicsModel,DifferentialFreeASLRFwdDynamicsData)
from .actuation_aslr import ASLRActuation
import numpy as np
import crocoddyl
import pinocchio
import time
import warnings


# def plot_residual(log, ord):
#     for i in [-1]:
#         residual_norm = []
#         for j in range(len(log.residual[i])):
#             residual_norm.append(np.linalg.norm(log.residual[i][j], ord))
#         plt.xlabel('Nodes')
#         plt.ylabel(str(ord) + "-norm residual each node")
#         plt.title('')
#         plt.legend()
#         plt.plot(residual_norm)
#         plt.show()
