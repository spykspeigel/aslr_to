
from __future__ import print_function

import os
import sys

import crocoddyl
import pinocchio
import numpy as np
import example_robot_data
import aslr_to
import time
WITHDISPLAY = 'display' in sys.argv or 'CROCODDYL_DISPLAY' in os.environ
WITHPLOT = 'plot' in sys.argv or 'CROCODDYL_PLOT' in os.environ

#WITHPLOT =True
panda = example_robot_data.load('panda')
robot_model = panda.model
state = crocoddyl.StateMultibody(robot_model)
actuation = aslr_to.ASRActuation(state)

nu = actuation.nu

runningCostModel = crocoddyl.CostModelSum(state,nu)
terminalCostModel = crocoddyl.CostModelSum(state,nu)

xActivation = crocoddyl.ActivationModelWeightedQuad(np.array([1e0] * 6 + [1e-3] * 6 + [1e0] * robot_model.nv + [1e-3] * robot_model.nv))
xResidual = crocoddyl.ResidualModelState(state, state.zero(), nu)
xRegCost = crocoddyl.CostModelResidual(state, xActivation, xResidual)
uResidual = crocoddyl.ResidualModelControl(state, nu)
uRegCost = crocoddyl.CostModelResidual(state, uResidual)
target = np.array([.01, .1, .1])
framePlacementResidual = aslr_to.ResidualModelFramePlacementASR(state, robot_model.getFrameId("panda_joint7"),
                                                               pinocchio.SE3(np.eye(3), target), nu)
goalTrackingCost = crocoddyl.CostModelResidual(state, framePlacementResidual)
#xRegCost = crocoddyl.CostModelResidual(state, xResidual)

# Then let's added the running and terminal cost functions
runningCostModel.addCost("gripperPose", goalTrackingCost, 1e0)
runningCostModel.addCost("xReg", xRegCost, 1e-2)
runningCostModel.addCost("uReg", uRegCost, 1e-1)
terminalCostModel.addCost("gripperPose", goalTrackingCost, 4e4)


K = 1e1*np.eye(int(state.nv/2))
B = 1e-2*np.eye(int(state.nv/2))

dt = 1e-3
runningModel = aslr_to.IntegratedActionModelEulerASR(
    aslr_to.DifferentialFreeASRFwdDynamicsModel(state, actuation, runningCostModel,K,B), dt)
#runningModel.differentia# log = solver.getCallbacks()[0]
# u1 , u2 = aslr_to.u_squared(log)
# print("printing usquared")
# print(aslr_to.u_squared(log))l.armature = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.])
terminalModel = aslr_to.IntegratedActionModelEulerASR(
    aslr_to.DifferentialFreeASRFwdDynamicsModel(state, actuation, terminalCostModel,K,B), 0)
#terminalModel.differential.armature = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.])

T = 250

q0 = np.array([0.173046, 1., -0.52366, 0., 0., 0.1])
x0 = np.concatenate([q0, q0,pinocchio.utils.zero(state.nv)])

problem = crocoddyl.ShootingProblem(x0, [runningModel] * T, terminalModel)

# Creating the DDP solver for this OC problem, defining a logger
solver = crocoddyl.SolverFDDP(problem)
cameraTF = [2., 2.68, 0.54, 0.2, 0.62, 0.72, 0.22]
# if WITHDISPLAY and WITHPLOT:
#     display = crocoddyl.GepettoDisplay(talos_arm, 4, 4, cameraTF)
#     solver.setCallbacks([crocoddyl.CallbackLogger(), crocoddyl.CallbackVerbose(), crocoddyl.CallbackDisplay(display)])
# elif WITHDISPLAY:
#     display = crocoddyl.GepettoDisplay(talos_arm, 4, 4, cameraTF)
#     solver.setCallbacks([crocoddyl.CallbackVerbose(), crocoddyl.CallbackDisplay(display)])
# elif WITHPLOT:
#     solver.setCallbacks([
#         crocoddyl.CallbackLogger(),
#         crocoddyl.CallbackVerbose(),
#     ])
# else:
#     solver.setCallbacks([crocoddyl.CallbackVerbose()])

if WITHDISPLAY:
    display = crocoddyl.GepettoDisplay(kinova)
    display.robot.viewer.gui.addSphere('world/point', .05, [1., 0., 0., 1.])  # radius = .1, RGBA=1001
    display.robot.viewer.gui.applyConfiguration('world/point',
                                                target.tolist() + [0., 0., 0., 1.])  # xyz+quaternion
    display.robot.viewer.gui.refresh()

solver.setCallbacks([crocoddyl.CallbackLogger(), crocoddyl.CallbackVerbose() ])

# xs = [x0] * (solver.problem.T + 1)
# us = solver.problem.quasiStatic([x0] * solver.problem.T)
solver.th_stop = 1e-7
# Solving it with the DDP algorithm
solver.solve([],[],300)

log = solver.getCallbacks()[0]
# u1 , u2 = aslr_to.u_squared(log)
# print("printing usquared")
print(aslr_to.u_squared(log))
log = solver.getCallbacks()[0]
# aslr_to.plotKKTerror(log.fs)

print('Finally reached = ', solver.problem.terminalData.differential.multibody.pinocchio.oMf[robot_model.getFrameId(
    "gripper_left_joint")].translation.T)

# Plotting the solution and the DDP convergence
if WITHPLOT:
    log = solver.getCallbacks()[0]
    crocoddyl.plotOCSolution(log.xs, log.us,figIndex=1, show=True)
    #crocoddyl.plotConvergence(log.costs, log.u_regs, log.x_regs, log.grads, log.stops, log.steps, figIndex=2)

# Visualizing the solution in gepetto-viewer
if WITHDISPLAY:
    while True:
        display.displayFromSolver(solver)
        time.sleep(2.0)