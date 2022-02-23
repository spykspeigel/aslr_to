import os
import sys

import numpy as np
import time
import crocoddyl
import example_robot_data
import pinocchio
import aslr_to
import inv_dyn
from quadruped_condensed_problem import SimpleQuadrupedalGaitProblem

WITHDISPLAY = "display" in sys.argv or "CROCODDYL_DISPLAY" in os.environ
WITHPLOT = "plot" in sys.argv or "CROCODDYL_PLOT" in os.environ

# Loading the anymal model
anymal = example_robot_data.load("anymal")

# Defining the initial state of the robot
q0 = anymal.model.referenceConfigurations["standing"].copy()
v0 = pinocchio.utils.zero(anymal.model.nv)
x0 = np.concatenate([q0, v0])
K = 50*np.eye(12)
# Setting up the 3d walking problem
lfFoot, rfFoot, lhFoot, rhFoot = "LF_FOOT", "RF_FOOT", "LH_FOOT", "RH_FOOT"
gait = SimpleQuadrupedalGaitProblem(anymal.model, lfFoot, rfFoot, lhFoot, rhFoot,K)

timeStep = 1e-2
numKnots = 100
comGoTo = 0.35
solver = crocoddyl.SolverFDDP(gait.createCoMProblem(x0, comGoTo, timeStep, numKnots))
solver.problem.nthreads = 1
cameraTF = [2.0, 2.68, 0.84, 0.2, 0.62, 0.72, 0.22]
if WITHDISPLAY and WITHPLOT:
    display = crocoddyl.GepettoDisplay(anymal, 4, 4, cameraTF, frameNames=[lfFoot, rfFoot, lhFoot, rhFoot])
    solver.setCallbacks([
        crocoddyl.CallbackLogger(),
        crocoddyl.CallbackVerbose(),
        crocoddyl.CallbackDisplay(display),
        inv_dyn.CallbackResidualLogger('feascost')
    ])
elif WITHDISPLAY:
    display = crocoddyl.GepettoDisplay(anymal, 4, 4, cameraTF, frameNames=[lfFoot, rfFoot, lhFoot, rhFoot])
    solver.setCallbacks([crocoddyl.CallbackVerbose(), crocoddyl.CallbackDisplay(display),  inv_dyn.CallbackResidualLogger('feascost')])
elif WITHPLOT:
    solver.setCallbacks([crocoddyl.CallbackLogger(), crocoddyl.CallbackVerbose()])
else:
    solver.setCallbacks([crocoddyl.CallbackVerbose(), inv_dyn.CallbackResidualLogger('feascost')])
solver.th_stop = 1e-5

xs = [x0] * (solver.problem.T + 1)
# us = solver.problem.quasiStatic([x0] * solver.problem.T)
us = [np.zeros(12)] * solver.problem.T
solver.solve(xs, us, 100)
log = solver.getCallbacks()[-1]
aslr_to.plot_theta(log,K[0,0])
# rd=solver.problem.runningDatas.tolist()
# print(rd[0].differential.multibody.pinocchio.oMf[anymal.model.getFrameId(
#     lfFoot)].translation.T)
# print(solver.problem.terminalData.differential.multibody.pinocchio.oMf[anymal.model.getFrameId(
#     lfFoot)].translation.T)
if WITHDISPLAY:
    display = crocoddyl.GepettoDisplay(anymal, frameNames=[lfFoot, rfFoot, lhFoot, rhFoot])
    while True:
        display.displayFromSolver(solver)
        time.sleep(2.0)
