import os
import sys

#This unittest is for easier reference

import crocoddyl
import pinocchio
import numpy as np
import aslr_to
import example_robot_data
from test_utils_ex import NUMDIFF_MODIFIER, assertNumDiff
robot_model = example_robot_data.load('fishing_rod').model
state = crocoddyl.StateMultibody(robot_model)
actuation = aslr_to.ASRFishing(state)
nu = 1
costs = crocoddyl.CostModelSum(state, nu)

# framePlacementResidual = aslr_to.ResidualModelFramePlacementASR(state, robot_model.getFrameId("gripper_left_joint"),
#                                                                pinocchio.SE3(np.eye(3), np.array([.0, .0, .4])), nu)
# goalTrackingCost = crocoddyl.CostModelResidual(state, framePlacementResidual)
# costs.addCost("gripperPose",goalTrackingCost,nu)
# xResidual = crocoddyl.ResidualModelControl(state, nu)
# xRegCost = crocoddyl.CostModelResidual(state, xResidual)
# costs.addCost("xReg", xRegCost, 1e-2)

model = aslr_to.DAM2(state, actuation, costs)
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
assertNumDiff(data.Fx[:,:21], DATA_ND.Fx[:,:21], NUMDIFF_MODIFIER *
                MODEL_ND.disturbance)  # threshold was 2.7e-2, is now 2.11e-4 (see assertNumDiff.__doc__)
assertNumDiff(data.Fu, DATA_ND.Fu, NUMDIFF_MODIFIER *
                MODEL_ND.disturbance)  # threshold was 2.7e-2, is now 2.11e-4 (see assertNumDiff.__doc__)

assertNumDiff(data.Lx, DATA_ND.Lx, NUMDIFF_MODIFIER *
                1e-6)  # threshold was 2.7e-2, is now 2.11e-4 (see assertNumDiff.__doc__)
assertNumDiff(data.Lu, DATA_ND.Lu, NUMDIFF_MODIFIER *
                1e-6)  # threshold was 2.7e-2, is now 2.11e-4 (see assertNumDiff.__doc__)
