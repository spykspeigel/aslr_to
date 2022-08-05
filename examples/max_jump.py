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
from soft_leg_problem import MonopedProblem

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
q0[0] = 0.315    
q0[1] = -np.pi/3
q0[2] = -np.pi/3

v0 = pinocchio.utils.zero(softleg.model.nv)
x0 = np.concatenate([q0, v0, np.zeros(4)])

# Setting up the 3d walking problem
rhFoot =  "softleg_1_contact_link"
gait = MonopedProblem(softleg.model,  rhFoot)

# Setting up the control-limited DDP solver
jumpHeight = .1
solver = crocoddyl.SolverFDDP(
     gait.max_jump(x0, jumpHeight, 1e-3, 100, 400)) #target, timestep, groundknots, flyingknots

xs = [x0] * (solver.problem.T + 1)
us = solver.problem.quasiStatic([x0] * solver.problem.T)
# us = [np.zeros(12)] * solver.problem.T
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

solver.solve(xs, us, 100)
log = solver.getCallbacks()[-1]
if WITHPLOT:
    print("hey")
    log = solver.getCallbacks()[0]
    aslr_to.plotSEAOCSolution(log.xs ,log.us,figIndex=1, show=True)

# log1 = solver.getCallbacks()[0]
rd=solver.problem.runningDatas.tolist()
print(rd[0].differential.multibody.pinocchio.oMf[softleg.model.getFrameId(
    rhFoot)].translation.T)
print(rd[200].differential.multibody.pinocchio.oMf[softleg.model.getFrameId(
    rhFoot)].translation.T)
if WITHDISPLAY:
    display = inv_dyn.GepettoDisplayCustom(softleg, frameNames=[ rhFoot])
    while True:
        display.displayFromSolver(solver)
        time.sleep(2.0)