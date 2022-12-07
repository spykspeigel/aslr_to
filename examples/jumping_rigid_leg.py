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
from rigid_leg_problem import RigidMonopedProblem

WITHDISPLAY = "display" in sys.argv or "CROCODDYL_DISPLAY" in os.environ
WITHPLOT = "plot" in sys.argv or "CROCODDYL_PLOT" in os.environ

# Loading the softleg model
softleg = example_robot_data.load("softleg")

# Defining the initial state of the robot
q0 = softleg.q0
# OPTION 1 Select the angle of the first joint wrt vertical
angle = np.pi/4
# q0[0] = 1 * np.cos(angle)
# q0[1] = -np.pi + angle
# q0[2] = -.1 * angle

# OPTION 2 Initial configuration distributing the joints in a semicircle with foot in O (scalable if n_joints > 2)
q0[0] = 0.26    
q0[1] = -np.pi/3
q0[2] = -np.pi/2

v0 = pinocchio.utils.zero(softleg.model.nv)
x0 = np.concatenate([q0, v0])

# Setting up the 3d walking problem
rhFoot =  "softleg_1_contact_link"
gait = RigidMonopedProblem(softleg.model,  rhFoot)

# Defining the walking gait parameters
jumping_gait = {
        'jumpHeight': 1,
        'jumpLength': [0.0, 0, .0],
        'timeStep': 1e-3,
        'groundKnots': 50,
        'flyingKnots': 400
}

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
    display = crocoddyl.GepettoDisplay(softleg, 1, 1, cameraTF, frameNames=[rhFoot])
    solver.setCallbacks([
        crocoddyl.CallbackLogger(),
        crocoddyl.CallbackVerbose(),
        crocoddyl.CallbackDisplay(display),
    ])
elif WITHDISPLAY:
    display = crocoddyl.GepettoDisplay(softleg, 1, 1, cameraTF, frameNames=[rhFoot])
    solver.setCallbacks([crocoddyl.CallbackVerbose(), crocoddyl.CallbackDisplay(display)])
elif WITHPLOT:
    solver.setCallbacks([crocoddyl.CallbackLogger(), crocoddyl.CallbackVerbose()])
else:
    solver.setCallbacks([crocoddyl.CallbackVerbose()])
solver.th_stop = 1e-5

xs = [x0] * (solver.problem.T + 1)
us = solver.problem.quasiStatic([x0] * solver.problem.T)
# us = [np.zeros(12)] * solver.problem.T

solver.solve(xs, us, 100)
log = solver.getCallbacks()[-1]
if WITHPLOT:
    log = solver.getCallbacks()[0]
    aslr_to.plotSEAOCSolution(log.xs ,log.us,figIndex=1, show=True)

log1 = solver.getCallbacks()[0]
# rd=solver.problem.runningDatas.tolist()
# print(rd[0].differential.multibody.pinocchio.oMf[anymal.model.getFrameId(
#     lfFoot)].translation.T)
# print(solver.problem.terminalData.differential.multibody.pinocchio.oMf[anymal.model.getFrameId(
#     lfFoot)].translation.T)
if WITHDISPLAY:
    display = inv_dyn.GepettoDisplayCustom(softleg, frameNames=[ rhFoot])
    while True:
        display.displayFromSolver(solver)
        time.sleep(2.0)
