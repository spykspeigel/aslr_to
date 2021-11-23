from __future__ import print_function

import os
import sys

import crocoddyl
import pinocchio
import numpy as np
import example_robot_data
import aslr_to

WITHDISPLAY = 'display' in sys.argv or 'CROCODDYL_DISPLAY' in os.environ
WITHPLOT = 'plot' in sys.argv or 'CROCODDYL_PLOT' in os.environ


talos_arm = example_robot_data.load('talos_arm')
robot_model = talos_arm.model
state = aslr_to.StateMultibodyASLR(robot_model)
actuation = aslr_to.ASLRActuation(state)

nu = actuation.nu

runningCostModel = crocoddyl.CostModelSum(state,nu)
terminalCostModel = crocoddyl.CostModelSum(state,nu)

framePlacementResidual = aslr_to.ResidualModelFramePlacementASLR(state, robot_model.getFrameId("gripper_left_joint"),
                                                               pinocchio.SE3(np.eye(3), np.array([.0, .0, .4])), nu)
uResidual = crocoddyl.ResidualModelControl(state, nu)
xResidual = crocoddyl.ResidualModelControl(state, nu)
goalTrackingCost = crocoddyl.CostModelResidual(state, framePlacementResidual)
xRegCost = crocoddyl.CostModelResidual(state, xResidual)
uRegCost = crocoddyl.CostModelResidual(state, uResidual)

# Then let's added the running and terminal cost functions
runningCostModel.addCost("gripperPose", goalTrackingCost, 1e2)
runningCostModel.addCost("xReg", xRegCost, 1e-3)
runningCostModel.addCost("uReg", uRegCost, 1e-4)
terminalCostModel.addCost("gripperPose", goalTrackingCost, 1e6)


dt = 1e-3
runningModel = aslr_to.IntegratedActionModelEulerASLR(
    aslr_to.DifferentialFreeASLRFwdDynamicsModel(state, actuation, runningCostModel), dt)
runningModel.differential.armature = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.])
terminalModel = aslr_to.IntegratedActionModelEulerASLR(
    aslr_to.DifferentialFreeASLRFwdDynamicsModel(state, actuation, terminalCostModel), 0.)
terminalModel.differential.armature = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.])

T = 250
q0 = np.array([0.173046, 1., -0.52366, 0., 0., 0.1, -0.005])
x0 = np.concatenate([q0, q0,pinocchio.utils.zero(state.nv)])
print(x0.shape)
problem = crocoddyl.ShootingProblem(x0, [runningModel] * T, terminalModel)

# Creating the DDP solver for this OC problem, defining a logger
solver = aslr_to.DDPASLR(problem)
cameraTF = [2., 2.68, 0.54, 0.2, 0.62, 0.72, 0.22]
if WITHDISPLAY and WITHPLOT:
    display = crocoddyl.GepettoDisplay(talos_arm, 4, 4, cameraTF)
    solver.setCallbacks([crocoddyl.CallbackLogger(), crocoddyl.CallbackVerbose(), crocoddyl.CallbackDisplay(display)])
elif WITHDISPLAY:
    display = crocoddyl.GepettoDisplay(talos_arm, 4, 4, cameraTF)
    solver.setCallbacks([crocoddyl.CallbackVerbose(), crocoddyl.CallbackDisplay(display)])
elif WITHPLOT:
    solver.setCallbacks([
        crocoddyl.CallbackLogger(),
        crocoddyl.CallbackVerbose(),
    ])
else:
    solver.setCallbacks([crocoddyl.CallbackVerbose()])

xs = [x0] * (solver.problem.T + 1)
us = solver.problem.quasiStatic([x0] * solver.problem.T)
# Solving it with the DDP algorithm
solver.solve()

print('Finally reached = ', solver.problem.terminalData.differential.multibody.pinocchio.oMf[robot_model.getFrameId(
    "gripper_left_joint")].translation.T)

# Plotting the solution and the DDP convergence
if WITHPLOT:
    log = solver.getCallbacks()[0]
    crocoddyl.plotOCSolution(log.xs, log.us, figIndex=1, show=False)
    crocoddyl.plotConvergence(log.costs, log.u_regs, log.x_regs, log.grads, log.stops, log.steps, figIndex=2)

# Visualizing the solution in gepetto-viewer
if WITHDISPLAY:
    display = crocoddyl.GepettoDisplay(talos_arm, 4, 4, cameraTF)
    display.displayFromSolver(solver)
