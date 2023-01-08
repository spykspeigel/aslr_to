from __future__ import print_function

import os
import sys

import crocoddyl
import pinocchio
import numpy as np
import example_robot_data
import aslr_to
import time
from scipy.io import savemat
WITHDISPLAY = 'display' in sys.argv or 'CROCODDYL_DISPLAY' in os.environ
WITHPLOT = 'plot' in sys.argv or 'CROCODDYL_PLOT' in os.environ

#WITHPLOT =True
talos_arm = example_robot_data.load('talos_arm')
robot_model = talos_arm.model
state = aslr_to.StateMultibodyASR(robot_model)
# actuation = aslr_to.ActuationModelDoublePendulum(state,actLink=0,nu=7)
actuation = aslr_to.ASRActuation(state)
nu = actuation.nu

runningCostModel = crocoddyl.CostModelSum(state,nu)
terminalCostModel = crocoddyl.CostModelSum(state,nu)

xActivation = crocoddyl.ActivationModelWeightedQuad(np.array([1e0] * 7 + [1e0] * 7 + [1e0] * robot_model.nv + [1e0] * robot_model.nv))
xResidual = crocoddyl.ResidualModelState(state, state.zero(), nu)
xRegCost = crocoddyl.CostModelResidual(state, xActivation, xResidual)
uResidual = crocoddyl.ResidualModelControl(state, nu)
uRegCost = crocoddyl.CostModelResidual(state, uResidual)
target = np.array([.0, .0, .4])
framePlacementResidual = aslr_to.ResidualModelFramePlacementASR(state, robot_model.getFrameId("gripper_left_joint"),
                                                               pinocchio.SE3(np.eye(3), target), nu)
goalTrackingCost = crocoddyl.CostModelResidual(state, framePlacementResidual)
# xRegCost = crocoddyl.CostModelResidual(state, xResidual)

# Then let's added the running and terminal cost functions
runningCostModel.addCost("gripperPose", goalTrackingCost, 1e-1)
runningCostModel.addCost("xReg", xRegCost, 1e-1)
runningCostModel.addCost("uReg", uRegCost, 1e-2)
terminalCostModel.addCost("gripperPose", goalTrackingCost, 4e4)

K = 50*np.eye(int(state.nv/2))
B = 1e-3*np.eye(int(state.nv/2))

dt = 1e-2
runningModel = aslr_to.IntegratedActionModelEulerASR(
    aslr_to.DifferentialFreeASRFwdDynamicsModel(state, actuation, runningCostModel,K,B), dt)

terminalModel = aslr_to.IntegratedActionModelEulerASR(
    aslr_to.DifferentialFreeASRFwdDynamicsModel(state, actuation, terminalCostModel,K,B), 0)
#terminalModel.differential.armature = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.])

T = 150

q0 = np.array([0.173046, 1., -0.52366, 0., 0., 0.1, -0.005])
x0 = np.concatenate([q0, q0,pinocchio.utils.zero(state.nv)])

problem = crocoddyl.ShootingProblem(x0, [runningModel] * T, terminalModel)

# Creating the DDP solver for this OC problem, defining a logger
solver = crocoddyl.SolverFDDP(problem)
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

if WITHDISPLAY:
    display = crocoddyl.GepettoDisplay(talos_arm)
    display.robot.viewer.gui.addSphere('world/point', .05, [1., 0., 0., 1.])  # radius = .1, RGBA=1001
    display.robot.viewer.gui.applyConfiguration('world/point',
                                                target.tolist() + [0., 0., 0., 1.])  # xyz+quaternion
    display.robot.viewer.gui.refresh()

solver.setCallbacks([crocoddyl.CallbackLogger(), crocoddyl.CallbackVerbose() ])

xs = [x0] * (solver.problem.T + 1)
us = solver.problem.quasiStatic([x0] * solver.problem.T)
solver.th_stop = 1e-7
# Solving it with the DDP algorithm
solver.solve([],[],300)

log = solver.getCallbacks()[0]
# u1 , u2 = aslr_to.u_squared(log)
# print("printing usquared")
print(np.sum(aslr_to.u_squared(log)))
# log = solver.getCallbacks()[0]
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


u1=np.array([])
u2=np.array([])
u3=np.array([])
u4=np.array([])
u5=np.array([])
u6=np.array([])
u7=np.array([])

x1=np.array([])
x2=np.array([])
x3=np.array([])
x4=np.array([])
x5=np.array([])
x6=np.array([])
x7=np.array([])

for i in range(len(log.us)):
    u1 = np.append(u1,log.us[i][0])
    u2 = np.append(u2,log.us[i][1])
    u3 = np.append(u3,log.us[i][2])
    u4 = np.append(u4,log.us[i][3])
    u5 = np.append(u5,log.us[i][4])
    u6 = np.append(u6,log.us[i][5])
    u7 = np.append(u7,log.us[i][6])

for i in range(len(log.xs)):
    x1 = np.append(x1,log.xs[i][0])
    x2 = np.append(x2,log.xs[i][1])
    x3 = np.append(x3,log.xs[i][2])
    x4 = np.append(x4,log.xs[i][3])
    x5 = np.append(x5,log.xs[i][4])
    x6 = np.append(x6,log.xs[i][5])
    x7 = np.append(x7,log.xs[i][6])

t=np.arange(0,T*dt,dt)

savemat("optimised_trajectory_vsa.mat", {"q1": x1,"q2":x2,"q3": x3,"q4":x4,"q5": x5,"q6":x6,"q7": x7,"t":t})
savemat("controls_vsa.mat", {"u1": u1,"u2":u2,"u3": u3,"u4":u4,"u5": u5,"u6":u6,"u7":u7,"t":t})
