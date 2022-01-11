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
two_dof = example_robot_data.load('asr_twodof')
robot_model = two_dof.model
robot_model.gravity.linear = np.array([0,-9.81,0])
state = aslr_to.StateMultibodyASR(robot_model)
actuation = aslr_to.ASRActuation(state)
nu = actuation.nu +1

runningCostModel = crocoddyl.CostModelSum(state,nu)
terminalCostModel = crocoddyl.CostModelSum(state,nu)

xActivation = crocoddyl.ActivationModelWeightedQuad(np.array([1e0] *2 + [0] *2 + [1e0] * robot_model.nv + [0]* robot_model.nv))
xResidual = crocoddyl.ResidualModelState(state, state.zero(), nu)
xRegCost = crocoddyl.CostModelResidual(state, xActivation, xResidual)
# uResidual = crocoddyl.ResidualModelControl(state, nu)
uActivation = crocoddyl.ActivationModelWeightedQuad(np.array([1e-1] + [1e0] * 2 ))
uResidual = crocoddyl.ResidualModelControl(state, nu)
uRegCost = crocoddyl.CostModelResidual(state, uActivation,uResidual)
# uRegCost = crocoddyl.CostModelResidual(state, uResidual)

framePlacementResidual = aslr_to.ResidualModelFramePlacementASR(state, robot_model.getFrameId("EE"),
                                                               pinocchio.SE3(np.eye(3), np.array([.01, .2, .18])), nu)
goalTrackingCost = crocoddyl.CostModelResidual(state, framePlacementResidual)
#xRegCost = crocoddyl.CostModelResidual(state, xResidual)

# Then let's added the running and terminal cost functions
runningCostModel.addCost("gripperPose", goalTrackingCost, 1e0)
runningCostModel.addCost("xReg", xRegCost, 1e-2)
runningCostModel.addCost("uReg", uRegCost, 1e0)
terminalCostModel.addCost("gripperPose", goalTrackingCost, 5e3)


B = .02*np.eye(int(state.nv/2))

dt = 1e-2
runningModel = aslr_to.IntegratedActionModelEulerASR(
    aslr_to.DifferentialFreeFwdDynamicsModelVSA(state, actuation, runningCostModel,B), dt)
#runningModel.differential.armature = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.])
terminalModel = aslr_to.IntegratedActionModelEulerASR(
    aslr_to.DifferentialFreeFwdDynamicsModelVSA(state, actuation, terminalCostModel,B), 0)
#terminalModel.differential.armature = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.])
l_lim_0=0
runningModel.u_lb = np.array([0, -50, -50])
runningModel.u_ub = np.array([100, 50, 50])
T = 100

q0 = np.array([.1,.0])
x0 = np.concatenate([q0,np.zeros(2),pinocchio.utils.zero(state.nv)])

problem = crocoddyl.ShootingProblem(x0, [runningModel] * T, terminalModel)



# Creating the DDP solver for this OC problem, defining a logger
solver = crocoddyl.SolverBoxDDP(problem)
cameraTF = [2., 2.68, 0.54, 0.2, 0.62, 0.72, 0.22]

if WITHDISPLAY:
    display = crocoddyl.GepettoDisplay(two_dof)

solver.setCallbacks([crocoddyl.CallbackLogger(), crocoddyl.CallbackVerbose() ])

xs = [x0] * (solver.problem.T + 1)
#us = solver.problem.quasiStatic([x0] * solver.problem.T)
solver.th_stop = 1e-7
# Solving it with the DDP algorithm
solver.solve([], [], 500)

print('Finally reached = ', solver.problem.terminalData.differential.multibody.pinocchio.oMf[robot_model.getFrameId(
    "EE")].translation.T)

log = solver.getCallbacks()[0]
u1 , u2 = aslr_to.u_squared(log)
print("printing usquared")
print(u1)
print("______")
print(u2)
# Plotting the solution and the DDP convergence
if WITHPLOT:
    log = solver.getCallbacks()[0]
    aslr_to.plotOCSolution(log.xs ,log.us,figIndex=1, show=True)
    #crocoddyl.plotConvergence(log.costs, log.u_regs, log.x_regs, log.grads, log.stops, log.steps, figIndex=2)

# Visualizing the solution in gepetto-viewer
if WITHDISPLAY:
    while True:
        display.displayFromSolver(solver)
        time.sleep(2.0)
