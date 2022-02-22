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

state = crocoddyl.StateMultibody(robot_model)
actuation = aslr_to.ASRActuationCondensed(state, 6)
nu = actuation.nu

costs = crocoddyl.CostModelSum(state, nu)

feas_residual = aslr_to.VSADynamicsResidualModel(state, nu)

feasCost = crocoddyl.CostModelResidual(state,feas_residual)
costs.addCost("feascost",feasCost,nu)
# xResidual = crocoddyl.ResidualModelControl(state, nu)
# xRegCost = crocoddyl.CostModelResidual(state, xResidual)
# costs.addCost("xReg", xRegCost, 1e-2)

model = crocoddyl.DifferentialActionModelFreeFwdDynamics(state, actuation, costs)
x =   model.state.rand()
u = np.random.rand(model.nu)
DATA =  model.createData()
MODEL_ND = crocoddyl.DifferentialActionModelNumDiff(model)
MODEL_ND.disturbance *= 10
DATA_ND = MODEL_ND.createData()
model.calc(DATA, x, u)
model.calcDiff(DATA, x, u)
MODEL_ND.calc(DATA_ND, x, u)
MODEL_ND.calcDiff(DATA_ND, x, u)
assertNumDiff(DATA.Fu, DATA_ND.Fu, NUMDIFF_MODIFIER *
                MODEL_ND.disturbance)  # threshold was 2.7e-2, is now 2.11e-4 (see assertNumDiff.__doc__)

assertNumDiff(DATA.Fx, DATA_ND.Fx, NUMDIFF_MODIFIER *
                MODEL_ND.disturbance)  # threshold was 2.7e-2, is now 2.11e-4 (see assertNumDiff.__doc__)

print(DATA.Lu)
print(DATA_ND.Lu)
assertNumDiff(DATA.Lx, DATA_ND.Lx, NUMDIFF_MODIFIER *
                1e-6)  # threshold was 2.7e-2, is now 2.11e-4 (see assertNumDiff.__doc__)
assertNumDiff(DATA.Lu, DATA_ND.Lu, NUMDIFF_MODIFIER *
                1e-6)  # threshold was 2.7e-2, is now 2.11e-4 (see assertNumDiff.__doc__)