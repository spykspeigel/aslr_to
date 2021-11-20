import os
import sys

import crocoddyl
import pinocchio
import numpy as np
import aslr_to
import example_robot_data
from test_utils_ex import NUMDIFF_MODIFIER, assertNumDiff
robot_model = example_robot_data.load('talos_arm').model
state = aslr_to.StateMultibodyASLR(robot_model)
actuation = aslr_to.ASLRActuation(state)
nu = actuation.nu 
costs = crocoddyl.CostModelSum(state, nu)
model = aslr_to.DifferentialFreeASLRFwdDynamicsModel(state, actuation, costs)
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

assertNumDiff(data.Fx[7:,:], DATA_ND.Fx[7:,:], NUMDIFF_MODIFIER *
                MODEL_ND.disturbance)  # threshold was 2.7e-2, is now 2.11e-4 (see assertNumDiff.__doc__)
