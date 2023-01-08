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
import matplotlib.pyplot as plt

WITHDISPLAY = 'display' in sys.argv or 'CROCODDYL_DISPLAY' in os.environ
WITHPLOT = 'plot' in sys.argv or 'CROCODDYL_PLOT' in os.environ

#WITHPLOT =True
two_dof = example_robot_data.load('asr_twodof')
robot_model = two_dof.model
robot_model.gravity.linear = np.array([9.81,0,0])
state = aslr_to.StateMultibodyASR(robot_model)
actuation = aslr_to.ASRActuation(state)
nu = actuation.nu

runningCostModel = crocoddyl.CostModelSum(state,nu)
terminalCostModel = crocoddyl.CostModelSum(state,nu)

xActivation = crocoddyl.ActivationModelWeightedQuad(np.array([1e0] *2 + [1e0] *2 + [1e0] * robot_model.nv + [1e0]* robot_model.nv))
xResidual = crocoddyl.ResidualModelState(state, state.zero(), nu)
xRegCost = crocoddyl.CostModelResidual(state, xActivation, xResidual)
uResidual = crocoddyl.ResidualModelControl(state, nu)
uRegCost = crocoddyl.CostModelResidual(state, uResidual)

target = np.array([.01, .2, .18])
target = np.array([.1, .2, .18])
# target = np.array([.1, .1, .18])
framePlacementResidual = aslr_to.ResidualModelFramePlacementASR(state, robot_model.getFrameId("EE"),
                                                               pinocchio.SE3(np.eye(3), target), nu)

# framePlacementResidual = aslr_to.ResidualModelFramePlacementASR(state, robot_model.getFrameId("EE"),
#                                                                pinocchio.SE3(np.eye(3), np.array([.01, .2, .18])), nu)                                                        

goalTrackingCost = crocoddyl.CostModelResidual(state, framePlacementResidual)
#xRegCost = crocoddyl.CostModelResidual(state, xResidual)

# Then let's added the running and terminal cost functions
runningCostModel.addCost("gripperPose", goalTrackingCost, 2e-1)
runningCostModel.addCost("xReg", xRegCost, 1e-2)
runningCostModel.addCost("uReg", uRegCost, 1e-1)
terminalCostModel.addCost("gripperPose", goalTrackingCost, 4e4)
# terminalCostModel.addCost("xReg", xRegCost, 1e0)

K = 5*np.eye(int(state.nv/2))
B = .001*np.eye(int(state.nv/2))

dt = 1e-3
runningModel = aslr_to.IntegratedActionModelEulerASR(
    aslr_to.DifferentialFreeASRFwdDynamicsModel(state, actuation, runningCostModel,K,B), dt)
#runningModel.differential.armature = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.])
terminalModel = aslr_to.IntegratedActionModelEulerASR(
    aslr_to.DifferentialFreeASRFwdDynamicsModel(state, actuation, terminalCostModel,K,B), 0)
#terminalModel.differential.armature = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.])

T = 200
q0 = np.array([.0,0])
x0 = np.concatenate([q0,q0,pinocchio.utils.zero(state.nv)])

problem = crocoddyl.ShootingProblem(x0, [runningModel] * T, terminalModel)

# Creating the DDP solver for this OC problem, defining a logger
# solver = aslr_to.DDPASLR(problem)
# solver = crocoddyl.SolverFDDP(problem)
solver = aslr_to.SolverINTRO(problem)
cameraTF = [2., 2.68, 0.54, 0.2, 0.62, 0.72, 0.22]

if WITHDISPLAY:
    display = crocoddyl.GepettoDisplay(two_dof)

solver.setCallbacks([crocoddyl.CallbackLogger(), crocoddyl.CallbackVerbose() ])

xs = [x0] * (solver.problem.T + 1)
us = solver.problem.quasiStatic([x0] * solver.problem.T)
solver.th_stop = 1e-7
# Solving it with the DDP algorithm
solver.solve([], [], 400)
print('Initial position = ', solver.problem.runningDatas.tolist()[0].differential.multibody.pinocchio.oMf[robot_model.getFrameId(
    "EE")].translation.T)


print('Finally reached = ', solver.problem.terminalData.differential.multibody.pinocchio.oMf[robot_model.getFrameId(
    "EE")].translation.T)


# log = solver.getCallbacks()[0]
# u1 , u2 = aslr_to.u_squared(log)
# print("printing usquared")
# print(np.sum(aslr_to.u_squared(log)))

# a = 8.9992
# k = 0.0019
# temp_u1=[]
# temp_q1=[]
# for i in range(len(log.us.tolist())):
#     temp_u1.append(log.us[i][0])
#     temp_q1.append(log.xs[i][0])
# theta_eq = temp_q1 - np.arcsinh(a * np.array(temp_u1)/K[0,0])/a
# theta_pr = np.arccosh(K[0,0]/(2*a*k))/a

# # Plotting the solution and the DDP convergence
if WITHPLOT:
    log = solver.getCallbacks()[0]
    aslr_to.plotSEAOCSolution(log.xs,log.us,figIndex=1, show=True)
    # crocoddyl.plotOCSolution(log.xs,log.us,figIndex=1, show=True)
    #crocoddyl.plotConvergence(log.costs, log.u_regs, log.x_regs, log.grads, log.stops, log.steps, figIndex=2)

# # Visualizing the solution in gepetto-viewer
if WITHDISPLAY:
    while True:
        display.displayFromSolver(solver)
        time.sleep(2.0)



# u1=np.array([])
# u2=np.array([])

# q1=np.array([])
# q2=np.array([])
# qm1=np.array([])
# qm2=np.array([])
# v1=np.array([])
# v2=np.array([])
# vm1=np.array([])
# vm2=np.array([])

# for i in range(len(log.us)):
#     u1 = np.append(u1,log.us[i][0])
#     u2 = np.append(u2,log.us[i][1])

# for i in range(len(log.xs)):
#     q1 = np.append(q1,log.xs[i][0])
#     q2 = np.append(q2,log.xs[i][1])
#     qm1 = np.append(qm1,log.xs[i][2])
#     qm2 = np.append(qm2,log.xs[i][3])
#     v1 = np.append(v1,log.xs[i][4])
#     v2 = np.append(v2,log.xs[i][5])
#     vm1 = np.append(vm1,log.xs[i][6])
#     vm2 = np.append(vm2,log.xs[i][7])


# t=np.arange(0,T*dt,dt)

# savemat("optimised_trajectory.mat", {"q1": q1,"q2":q2,"qm1": qm1,"qm2":qm2,"v1": v1,"v2":v2,"vm1": vm1,"vm2":vm2,"t":t})
# savemat("controls.mat", {"u1": u1,"u2":u2,"t":t})

# K=solver.K.tolist()
# K_temp = []
# for i in range(len(K)):
#     K_temp.append(np.linalg.norm(K[i]))

# K[-1] = int(K_temp[-2]/K_temp[-3])*K[-2]
# K[-2] = K[-3]
# K[-1] = K[-2]
# K_temp = []
# for i in range(len(K)):
#     K_temp.append(np.linalg.norm(K[i]))

# plt.plot(K_temp)
# plt.show()
# savemat("fb.mat", {"K":K,"t":t})
