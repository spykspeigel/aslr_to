import os
import sys

import numpy as np
import time
import crocoddyl
import example_robot_data
import pinocchio
import aslr_to
import inv_dyn
from soft_leg_problem import SimpleMonopedProblem


WITHDISPLAY = "display" in sys.argv or "CROCODDYL_DISPLAY" in os.environ
WITHPLOT = "plot" in sys.argv or "CROCODDYL_PLOT" in os.environ

# Loading the anymal model
anymal = example_robot_data.load("softleg")

# Defining the initial state of the robot
# q0 = anymal.model.referenceConfigurations["standing"].copy()
q0 = anymal.q0
v0 = pinocchio.utils.zero(anymal.model.nv)
x0 = np.concatenate([q0, v0, np.zeros(4)])

# Setting up the 3d walking problem
rhFoot =  "FOOT"

gait = SimpleMonopedProblem(anymal.model,  rhFoot)

timeStep = 1e-3
numKnots =500
comGoTo = 0.25

solver = crocoddyl.SolverFDDP(gait.createCoMGoalProblem(x0, comGoTo, timeStep, numKnots))
solver.problem.nthreads = 1
cameraTF = [2.0, 2.68, 0.84, 0.2, 0.62, 0.72, 0.22]
if WITHDISPLAY and WITHPLOT:
    display = crocoddyl.GepettoDisplay(anymal, 1, 1, cameraTF, frameNames=[rhFoot])
    solver.setCallbacks([
        crocoddyl.CallbackLogger(),
        crocoddyl.CallbackVerbose(),
        crocoddyl.CallbackDisplay(display),
    ])
elif WITHDISPLAY:
    display = crocoddyl.GepettoDisplay(anymal, 1, 1, cameraTF, frameNames=[rhFoot])
    solver.setCallbacks([crocoddyl.CallbackVerbose(), crocoddyl.CallbackDisplay(display)])
elif WITHPLOT:
    solver.setCallbacks([crocoddyl.CallbackLogger(), crocoddyl.CallbackVerbose()])
else:
    solver.setCallbacks([crocoddyl.CallbackVerbose()])
solver.th_stop = 1e-5

xs = [x0] * (solver.problem.T + 1)
# us = solver.problem.quasiStatic([x0] * solver.problem.T)
# rd = solver.problem.runningDatas.tolist()
# td = solver.problem.terminalData.differential.tmp_xstatic
# for i in range(len(rd)):
    # xs[i] = rd[i].differential.tmp_xstatic
# xs[-1] = rd[-1].differential.tmp_xstatic
# xs[-1] = solver.problem.terminalData.differential.tmp_xstatic
us = [np.zeros(2)] * solver.problem.T
solver.solve(xs, us, 100)

# print(rd[0].differential.multibody.pinocchio.oMf[anymal.model.getFrameId(
#     lfFoot)].translation.T)
print(solver.problem.terminalData.differential.multibody.pinocchio.oMf[anymal.model.getFrameId(
    rhFoot)].translation.T)
if WITHDISPLAY:
    display = inv_dyn.GepettoDisplayCustom(anymal, frameNames=[ rhFoot])
    while True:
        display.displayFromSolver(solver)
        time.sleep(2.0)

if WITHPLOT:
    log = solver.getCallbacks()[0]
    crocoddyl.plotOCSolution(log.xs ,log.us,figIndex=1, show=True)