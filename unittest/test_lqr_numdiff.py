import os
import sys

#This unittest is for easier reference

import crocoddyl
import pinocchio
import numpy as np
import aslr_to
import example_robot_data
from test_utils_ex import NUMDIFF_MODIFIER, assertNumDiff

model = aslr_to.DifferentialLQRModel(10,10)
x =   model.state.rand()
u = np.random.rand(model.nu)
data =  model.createData()
MODEL_ND = aslr_to.NumDiffASRFwdDynamicsModel(model,1e-4)
# MODEL_ND.disturbance *= 10
DATA_ND = MODEL_ND.createData()
model.calc(data, x, u)
model.calcDiff(data, x, u)
MODEL_ND.calc(DATA_ND, x, u)
MODEL_ND.calcDiff(DATA_ND, x, u)
# assertNumDiff(data.Fu, DATA_ND.Fu, NUMDIFF_MODIFIER *
#                 MODEL_ND.disturbance)  # threshold was 2.7e-2, is now 2.11e-4 (see assertNumDiff.__doc__)

# assertNumDiff(data.Fx, DATA_ND.Fx, NUMDIFF_MODIFIER *
#                 MODEL_ND.disturbance)  # threshold was 2.7e-2, is now 2.11e-4 (see assertNumDiff.__doc__)

# assertNumDiff(data.Lx, DATA_ND.Lx, NUMDIFF_MODIFIER *
#                 MODEL_ND.disturbance)  # threshold was 2.7e-2, is now 2.11e-4 (see assertNumDiff.__doc__)
# assertNumDiff(data.Lu, DATA_ND.Lu, NUMDIFF_MODIFIER *
#                 MODEL_ND.disturbance)  # threshold was 2.7e-2, is now 2.11e-4 (see assertNumDiff.__doc__)

# print(DATA_ND.Lxx)
# print(data.Luu)
# assertNumDiff(data.Lxx, DATA_ND.Lxx, NUMDIFF_MODIFIER *
#                 MODEL_ND.disturbance)  # threshold was 2.7e-2, is now 2.11e-4 (see assertNumDiff.__doc__)


dt = 1e-2
runningModel = aslr_to.IntegratedActionModelEulerASR(MODEL_ND, dt)


T = 1
q0 = np.zeros(10)
x0 = np.concatenate([q0,q0])

problem = crocoddyl.ShootingProblem(x0, [runningModel] * T, runningModel)

# Creating the DDP solver for this OC problem, defining a logger
solver = aslr_to.DDPASLR(problem)

xs = [x0] * (solver.problem.T + 1)
us = [q0] * (solver.problem.T)
solver.th_stop = 1e-7
solver.setCallbacks([crocoddyl.CallbackLogger(), crocoddyl.CallbackVerbose() ])

# Solving it with the DDP algorithm
solver.solve(xs, us, 10)