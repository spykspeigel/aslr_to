from __future__ import print_function

import os
import sys

import crocoddyl
import pinocchio
import numpy as np
import example_robot_data
import aslr_to
import time
import  matplotlib.pyplot as plt
from scipy.io import savemat
WITHDISPLAY = 'display' in sys.argv or 'CROCODDYL_DISPLAY' in os.environ
WITHPLOT = 'plot' in sys.argv or 'CROCODDYL_PLOT' in os.environ

#WITHPLOT =True
two_dof = example_robot_data.load('asr_twodof')
robot_model = two_dof.model
robot_model.gravity.linear = np.array([9.81,0,0])
state = crocoddyl.StateMultibody(robot_model)
actuation = crocoddyl.ActuationModelFull(state)
nu = actuation.nu

runningCostModel = crocoddyl.CostModelSum(state,nu)
terminalCostModel = crocoddyl.CostModelSum(state,nu)

xResidual = crocoddyl.ResidualModelState(state, nu)
uResidual = crocoddyl.ResidualModelControl(state, nu)
uRegCost = crocoddyl.CostModelResidual(state, uResidual)

framePlacementResidual = crocoddyl.ResidualModelFramePlacement(state, robot_model.getFrameId("EE"),
                                                               pinocchio.SE3(np.eye(3), np.array([.01, .2, .18])), nu)
goalTrackingCost = crocoddyl.CostModelResidual(state, framePlacementResidual)
xRegCost = crocoddyl.CostModelResidual(state, xResidual)

# Then let's added the running and terminal cost functions
runningCostModel.addCost("gripperPose", goalTrackingCost, 1e-1)
# runningCostModel.addCost("xReg", xRegCost, 1e-2)
# runningCostModel.addCost("uReg", uRegCost, 1e-1)
terminalCostModel.addCost("gripperPose", goalTrackingCost, 4e4)
runningModel= crocoddyl.DifferentialActionModelFreeFwdDynamics(state, actuation, runningCostModel)
model_nd_r = crocoddyl.DifferentialActionModelNumDiff(runningModel)
terminalModel = crocoddyl.DifferentialActionModelFreeFwdDynamics(state, actuation, terminalCostModel)
model_nd_t = crocoddyl.DifferentialActionModelNumDiff(terminalModel)
dt = 1e-2
runningModel = crocoddyl.IntegratedActionModelEuler(
    # runningModel, dt)
    model_nd_r,dt)
terminalModel = crocoddyl.IntegratedActionModelEuler(
    # terminalModel, 0)
    model_nd_t,0)
T = 200

q0 = np.array([.0,0])
x0 = np.concatenate([q0,pinocchio.utils.zero(state.nv)])

problem = crocoddyl.ShootingProblem(x0, [runningModel] * T, terminalModel)

# Creating the DDP solver for this OC problem, defining a logger
solver = crocoddyl.SolverDDP(problem)
cameraTF = [2., 2.68, 0.54, 0.2, 0.62, 0.72, 0.22]


if WITHDISPLAY:
    display = crocoddyl.GepettoDisplay(two_dof)

solver.setCallbacks([crocoddyl.CallbackLogger(), crocoddyl.CallbackVerbose() ])

xs = [x0] * (solver.problem.T + 1)
us = solver.problem.quasiStatic([x0] * solver.problem.T)
solver.th_stop = 1e-5
# Solving it with the DDP algorithm
solver.solve(xs,us,10000)

# print('Finally reached = ', solver.problem.runningDatas.tolist()[0].differential.multibody.pinocchio.oMf[robot_model.getFrameId(
#     "EE")].translation.T)

# print('Finally reached = ', solver.problem.terminalData.differential.multibody.pinocchio.oMf[robot_model.getFrameId(
#     "EE")].translation.T)

# log = solver.getCallbacks()[0]
# print( np.sum(aslr_to.u_squared(log)))
# # print("printing usquared")
# # print(u1)
# # print("______")
# # print(u2)
# # Plotting the solution and the DDP convergence
# if WITHPLOT:
#     log = solver.getCallbacks()[0]
#     aslr_to.plotOCSolution(log.xs, log.us,figIndex=1, show=True)
    #crocoddyl.plotConvergence(log.costs, log.u_regs, log.x_regs, log.grads, log.stops, log.steps, figIndex=2)

# Visualizing the solution in gepetto-viewer
if WITHDISPLAY:
    while True:
        display.displayFromSolver(solver)
        time.sleep(2.0)

# u1=np.array([])
# u2=np.array([])

# q1=np.array([])
# q2=np.array([])
# v1=np.array([])
# v2=np.array([])

# for i in range(len(log.us)):
#     u1 = np.append(u1,log.us[i][0])
#     u2 = np.append(u2,log.us[i][1])

# for i in range(len(log.xs)):
#     q1 = np.append(q1,log.xs[i][0])
#     q2 = np.append(q2,log.xs[i][1])

#     v1 = np.append(v1,log.xs[i][2])
#     v2 = np.append(v2,log.xs[i][3])


# t=np.arange(0,T*dt,dt)

# savemat("optimised_trajectory.mat", {"q1": q1,"q2":q2,"v1": v1,"v2":v2,"t":t})
# savemat("controls.mat", {"u1": u1,"u2":u2,"t":t})

# K=solver.K.tolist()
# K_temp = []
# for i in range(len(K)):
#     K_temp.append(np.linalg.norm(K[i]))


# K_temp = []
# for i in range(len(K)):
#     K_temp.append(np.linalg.norm(K[i]))

# plt.plot(K_temp)
# plt.show()
# savemat("fb.mat", {"K":K,"t":t})
