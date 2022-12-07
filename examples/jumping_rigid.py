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
gait = SimpleQuadrupedalGaitProblem(anymal.model, lfFoot, rfFoot, lhFoot, rhFoot)

# Defining the walking gait parameters
jumping_gait = {
        'jumpHeight': 0.15,
        'jumpLength': [0.0, 0.3, 0.],
        'timeStep': 1e-3,
        'groundKnots': 100,
        'flyingKnots': 200
}
# trotting_gait = {
#         'stepLength': 0.15,
#         'stepHeight': 0.1,
#         'timeStep': 1e-2,
#         'stepKnots': 25,
#         'supportKnots': 2
#     }
# Setting up the control-limited DDP solver
solver = crocoddyl.SolverFDDP(
    gait.createJumpingProblem(x0, jumping_gait['jumpHeight'], jumping_gait['jumpLength'], jumping_gait['timeStep'],
                                jumping_gait['groundKnots'], jumping_gait['flyingKnots']))

# solver = crocoddyl.SolverFDDP(
#     gait.createTrottingProblem(x0, trotting_gait['stepLength'], trotting_gait['stepHeight'], trotting_gait['timeStep'],
#                               trotting_gait['stepKnots'], trotting_gait['supportKnots']))

solver.problem.nthreads = 1
cameraTF = [2.0, 2.68, 0.84, 0.2, 0.62, 0.72, 0.22]
if WITHDISPLAY and WITHPLOT:
    display = aslr_to.GepettoDisplay(anymal, 4, 4, cameraTF, frameNames=[lfFoot, rfFoot, lhFoot, rhFoot])
    solver.setCallbacks([
        crocoddyl.CallbackLogger(),
        crocoddyl.CallbackVerbose(),
        crocoddyl.CallbackDisplay(display)
    ])
elif WITHDISPLAY:
    display = aslr_to.GepettoDisplay(anymal, 4, 4, cameraTF, frameNames=[lfFoot, rfFoot, lhFoot, rhFoot])
    solver.setCallbacks([crocoddyl.CallbackVerbose(), aslr_to.CallbackDisplay(display)])
elif WITHPLOT:
    solver.setCallbacks([crocoddyl.CallbackLogger(), crocoddyl.CallbackVerbose()])
else:
    solver.setCallbacks([crocoddyl.CallbackLogger(),crocoddyl.CallbackVerbose()])
solver.th_stop = 1e-5

xs = [x0] * (solver.problem.T + 1)
us = solver.problem.quasiStatic([x0] * solver.problem.T)
# us = [np.zeros(12)] * solver.problem.T

solver.solve(xs, us, 100)
log = solver.getCallbacks()[-1]

log1 = solver.getCallbacks()[0]
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
print(np.sum(u_squared(log1)))

if WITHPLOT:
    log = solver.getCallbacks()[0]
    aslr_to.plotSEAOCSolution(log.xs ,log.us,figIndex=1, show=True)