from .statemultiasr import StateMultiASR
from .contact_fwddyn_aslr import (DifferentialContactASLRFwdDynModel, DifferentialContactASLRFwdDynData)
from .residual_frame_placement import (ResidualModelFramePlacementASR,ResidualDataFramePlacementASR)
from .floating_actuation import ASRFreeFloatingActuation
from .contact_fwddyn_rigid import (DifferentialContactFwdDynModelRigid,DifferentialContactFwdDynDataRigid)
from .floating_actuation_condensed import FreeFloatingActuationCondensed
from .residual_floating_soft_model import (FloatingSoftDynamicsResidualModel, FloatingSoftDynamicsResidualData)
from .residual_floating_vsa_model import (FloatingVSADynamicsResidualModel,FloatingVSADynamicsResidualData)
from .softmodel_data import (SoftModelData,DataCollectorSoftModel)
from .datacollector_softmodel_multibody import (DataCollectorSoftActMultibody,DataCollectorSoftActMultibodyInContact)
from .solver import DDPASLR
import numpy as np
import crocoddyl
import pinocchio
import time
import warnings
import matplotlib.pyplot as plt

def plot_theta(log):
    for i in [-1]:
        for k in range(12):
            theta = []
            for j in range(len(log.residual[i])):
                theta.append(log.residual[i][j][k])
            plt.xlabel('Nodes')
            plt.ylabel( "theta ")
            plt.title('')
            plt.plot(theta,label="theta_"+str(k))
            plt.legend()
        plt.show()

def u_squared(log):
    u_sqaured = np.zeros(len(log.us[0]))
    for i in range(len(log.us)):
        for j in range(len(log.us[i])):
            u_sqaured[j] += log.us[i][j]**2
    return u_sqaured
