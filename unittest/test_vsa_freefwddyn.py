import os
import sys

#This unittest is for easier reference

import crocoddyl
import pinocchio
import numpy as np
import aslr_to
import example_robot_data
from test_utils_ex import NUMDIFF_MODIFIER, assertNumDiff
two_dof = example_robot_data.load('asr_twodof')
robot_model = two_dof.model
state = aslr_to.StateMultibodyASR(robot_model)
actuation = aslr_to.ASRActuation(state)
nu = actuation.nu
costs = crocoddyl.CostModelSum(state, nu)

model = aslr_to.DifferentialFreeASRFwdDynamicsModel(state, actuation, costs)
x =   model.state.rand()
u = np.random.rand(model.nu)
data =  model.createData()
MODEL_ND = crocoddyl.DifferentialActionModelNumDiff(model)
MODEL_ND.disturbance *= 10
DATA_ND = MODEL_ND.createData()
model.calc(data, x, u)
model.calcDiff(data, x, u)
MODEL_ND.calc(DATA_ND, x, u)
MODEL_ND.calcDiff(DATA_ND, x, u)
assertNumDiff(data.Fu, DATA_ND.Fu, NUMDIFF_MODIFIER *
                MODEL_ND.disturbance)  # threshold was 2.7e-2, is now 2.11e-4 (see assertNumDiff.__doc__)

assertNumDiff(data.Fx, DATA_ND.Fx, NUMDIFF_MODIFIER *
                MODEL_ND.disturbance)  # threshold was 2.7e-2, is now 2.11e-4 (see assertNumDiff.__doc__)

assertNumDiff(data.Lx, DATA_ND.Lx, NUMDIFF_MODIFIER *
                MODEL_ND.disturbance)  # threshold was 2.7e-2, is now 2.11e-4 (see assertNumDiff.__doc__)
assertNumDiff(data.Lu, DATA_ND.Lu, NUMDIFF_MODIFIER *
                MODEL_ND.disturbance)  # threshold was 2.7e-2, is now 2.11e-4 (see assertNumDiff.__doc__)
