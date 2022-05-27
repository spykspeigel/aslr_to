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
runningCostModel.addCost("gripperPose", goalTrackingCost, 1e0)
runningCostModel.addCost("xReg", xRegCost, 1e0)
runningCostModel.addCost("uReg", uRegCost, 1e-1)
terminalCostModel.addCost("gripperPose", goalTrackingCost, 4e4)

dt = 1e-3
runningModel = crocoddyl.IntegratedActionModelEuler(
    crocoddyl.DifferentialActionModelFreeFwdDynamics(state, actuation, runningCostModel), dt)
terminalModel = crocoddyl.IntegratedActionModelEuler(
    crocoddyl.DifferentialActionModelFreeFwdDynamics(state, actuation, terminalCostModel), 0)

T = 250

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
solver.th_stop = 1e-7
# Solving it with the DDP algorithm
solver.solve()


print('Finally reached = ', solver.problem.runningDatas.tolist()[0].differential.multibody.pinocchio.oMf[robot_model.getFrameId(
    "EE")].translation.T)

print('Finally reached = ', solver.problem.terminalData.differential.multibody.pinocchio.oMf[robot_model.getFrameId(
    "EE")].translation.T)

log = solver.getCallbacks()[0]
aslr_to.plotrigidOCSolution(log.xs,figTitle="Rigid actuated")


x1=np.array([])
x2=np.array([])

for i in range(len(log.xs)):
    x1 = np.append(x1,log.xs[i][0])
    x2 = np.append(x2,log.xs[i][1])

t=np.arange(0,T*dt,dt)

savemat("ra.mat", {"q1": x1,"q2":x2,"t":t})



#######################################
### SOFT MODEL
##########################################
state = aslr_to.StateMultibodyASR(robot_model)
actuation = aslr_to.ASRActuation(state)
nu = actuation.nu

runningCostModel = crocoddyl.CostModelSum(state,nu)
terminalCostModel = crocoddyl.CostModelSum(state,nu)

xActivation = crocoddyl.ActivationModelWeightedQuad(np.array([1e0] *2 + [0] *2 + [1e0] * robot_model.nv + [0]* robot_model.nv))
xResidual = crocoddyl.ResidualModelState(state, state.zero(), nu)
xRegCost = crocoddyl.CostModelResidual(state, xActivation, xResidual)
uResidual = crocoddyl.ResidualModelControl(state, nu)
uRegCost = crocoddyl.CostModelResidual(state, uResidual)

framePlacementResidual = aslr_to.ResidualModelFramePlacementASR(state, robot_model.getFrameId("EE"),
                                                               pinocchio.SE3(np.eye(3), np.array([.01, .2, .18])), nu)

# framePlacementResidual = aslr_to.ResidualModelFramePlacementASR(state, robot_model.getFrameId("EE"),
#                                                                pinocchio.SE3(np.eye(3), np.array([.01, .2, .18])), nu)                                                        

goalTrackingCost = crocoddyl.CostModelResidual(state, framePlacementResidual)
#xRegCost = crocoddyl.CostModelResidual(state, xResidual)

# Then let's added the running and terminal cost functions
runningCostModel.addCost("gripperPose", goalTrackingCost, 1e0)
runningCostModel.addCost("xReg", xRegCost, 1e0)
runningCostModel.addCost("uReg", uRegCost, 1e-1)
terminalCostModel.addCost("gripperPose", goalTrackingCost, 4e4)


K =1*np.eye(int(state.nv/2))
B = .001*np.eye(int(state.nv/2))

dt = 1e-3
runningModel = aslr_to.IntegratedActionModelEulerASR(
    aslr_to.DifferentialFreeASRFwdDynamicsModel(state, actuation, runningCostModel,K,B), dt)

terminalModel = aslr_to.IntegratedActionModelEulerASR(
    aslr_to.DifferentialFreeASRFwdDynamicsModel(state, actuation, terminalCostModel,K,B), 0)

T = 250

q0 = np.array([.0,0])
x0 = np.concatenate([q0,q0,pinocchio.utils.zero(state.nv)])

problem = crocoddyl.ShootingProblem(x0, [runningModel] * T, terminalModel)

xs_sa = problem.rollout(log.us)
print('Finally reached = ', problem.terminalData.differential.multibody.pinocchio.oMf[robot_model.getFrameId(
    "EE")].translation.T)

aslr_to.plotSEAOCSolution(xs_sa,figIndex=1, show=True, figTitle="SEA with stiffness 10 Nm/rad")

x1=np.array([])
x2=np.array([])

for i in range(len(log.xs)):
    x1 = np.append(x1,xs_sa[i][0])
    x2 = np.append(x2,xs_sa[i][1])

t=np.arange(0,T*dt,dt)

savemat("sa_3.mat", {"q1": x1,"q2":x2,"t":t})
