import os
import sys

import numpy as np
import time
import crocoddyl
import example_robot_data
import pinocchio
import aslr_to
import inv_dyn
from aslr_to import u_squared
from quadruped_problem import SimpleQuadrupedalGaitProblem

WITHDISPLAY = "display" in sys.argv or "CROCODDYL_DISPLAY" in os.environ
WITHPLOT = "plot" in sys.argv or "CROCODDYL_PLOT" in os.environ

# Loading the anymal model
anymal = example_robot_data.load("anymal")

# Defining the initial state of the robot
q0 = anymal.model.referenceConfigurations["standing"].copy()
v0 = pinocchio.utils.zero(anymal.model.nv)
x0 = np.concatenate([q0, v0, np.zeros(24)])

# Setting up the 3d walking problem
lfFoot, rfFoot, lhFoot, rhFoot = "LF_FOOT", "RF_FOOT", "LH_FOOT", "RH_FOOT"
# S = np.zeros([12,12])
# S[:12,:12] = np.diag(np.ones(12))
gait = SimpleQuadrupedalGaitProblem(anymal.model, lfFoot, rfFoot, lhFoot, rhFoot)

# Defining the walking gait parameters
walking_gait = {'stepLength': 0.25, 'stepHeight': 0.25, 'timeStep': 1e-3, 'stepKnots': 200, 'supportKnots': 10}

# Setting up the control-limited DDP solver
solver = crocoddyl.SolverFDDP(
    gait.createWalkingProblem(x0, walking_gait['stepLength'], walking_gait['stepHeight'], walking_gait['timeStep'],
                              walking_gait['stepKnots'], walking_gait['supportKnots']))

solver.problem.nthreads = 1
cameraTF = [2.0, 2.68, 0.84, 0.2, 0.62, 0.72, 0.22]

solver.th_stop = 1e-5
if WITHDISPLAY and WITHPLOT:
    display = aslr_to.GepettoDisplay(anymal, 4, 4, cameraTF, frameNames=[lfFoot, rfFoot, lhFoot, rhFoot])
    solver.setCallbacks([
        crocoddyl.CallbackLogger(),
        crocoddyl.CallbackVerbose(),
        crocoddyl.CallbackDisplay(display),
    ])
elif WITHDISPLAY:
    display = aslr_to.GepettoDisplay(anymal, 4, 4, cameraTF, frameNames=[lfFoot, rfFoot, lhFoot, rhFoot])
    solver.setCallbacks([crocoddyl.CallbackVerbose(), crocoddyl.CallbackDisplay(display)])
elif WITHPLOT:
    solver.setCallbacks([crocoddyl.CallbackLogger(), crocoddyl.CallbackVerbose()])
else:
    solver.setCallbacks([crocoddyl.CallbackVerbose()])
xs = [x0] * (solver.problem.T + 1)
us = solver.problem.quasiStatic([x0] * solver.problem.T)
# us = [np.zeros(12)] * solver.problem.T

solver.solve(xs, us, 100)
# log = solver.getCallbacks()[-1]
# aslr_to.plot_theta(log)
# log1 = solver.getCallbacks()[0]
# rd=solver.problem.runningDatas.tolist()
# print(rd[0].differential.multibody.pinocchio.oMf[anymal.model.getFrameId(
#     lfFoot)].translation.T)
# print(solver.problem.terminalData.differential.multibody.pinocchio.oMf[anymal.model.getFrameId(
#     lfFoot)].translation.T)
if WITHDISPLAY:
    display = aslr_to.GepettoDisplay(anymal, frameNames=[lfFoot, rfFoot, lhFoot, rhFoot])
    while True:
        display.displayFromSolver(solver)
        time.sleep(2.0)
# print(np.sum(u_squared(log1)[:12]))

if WITHPLOT:
    log = solver.getCallbacks()[0]
    aslr_to.plotSEAOCSolution(log.xs ,log.us,figIndex=1, show=True)
