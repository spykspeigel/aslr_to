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
four_dof = example_robot_data.load('four_dof')
robot_model = four_dof.model
state = aslr_to.StateMultibodyASR(robot_model)
# actuation = aslr_to.ASRActuation(state)
actuation = aslr_to.ActuationModelDoublePendulum(state,actLink=0,nu=4)
nu = actuation.nu

runningCostModel = crocoddyl.CostModelSum(state,nu)
terminalCostModel = crocoddyl.CostModelSum(state,nu)

xActivation = crocoddyl.ActivationModelWeightedQuad(np.array([1e0] * 4 + [1e0] * 4 + [1e0] * robot_model.nv + [1e0] * robot_model.nv))
xResidual = crocoddyl.ResidualModelState(state, state.zero(), nu)
xRegCost = crocoddyl.CostModelResidual(state, xActivation, xResidual)

xtermActivation = crocoddyl.ActivationModelWeightedQuad(np.array([0] * 4 + [0] * 4 + [1e0] * robot_model.nv + [1e0] * robot_model.nv))
xtermResidual = crocoddyl.ResidualModelState(state, state.zero(), nu)
xtermCost = crocoddyl.CostModelResidual(state, xActivation, xResidual)


uResidual = crocoddyl.ResidualModelControl(state, nu)
uRegCost = crocoddyl.CostModelResidual(state, uResidual)
target = np.array([.2, .3, .18])
framePlacementResidual = aslr_to.ResidualModelFramePlacementASR(state, robot_model.getFrameId("EE"),
                                                               pinocchio.SE3(np.eye(3), target), nu)

goalTrackingCost = crocoddyl.CostModelResidual(state, framePlacementResidual)
#xRegCost = crocoddyl.CostModelResidual(state, xResidual)
# Then let's added the running and terminal cost functions
runningCostModel.addCost("gripperPose", goalTrackingCost, 1e0)
runningCostModel.addCost("xReg", xRegCost, 1e-3)
runningCostModel.addCost("uReg", uRegCost, 1e-1)
terminalCostModel.addCost("gripperPose", goalTrackingCost, 1e5)
terminalCostModel.addCost("termVel", xtermCost, 1e0)


K = 5*np.eye(int(state.nv/2))
B = 1e-3*np.eye(int(state.nv/2))

dt = 1e-2
runningModel_a = aslr_to.DifferentialFreeASRFwdDynamicsModel(state, actuation, runningCostModel,K,B)
#runningModel.differential.armature = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.])
runningModel = aslr_to.IntegratedActionModelEulerASR(aslr_to.NumDiffASRFwdDynamicsModel(runningModel_a, 1e-4),dt)
terminalModel_a = aslr_to.DifferentialFreeASRFwdDynamicsModel(state, actuation, terminalCostModel,K,B)
#terminalModel.differential.armature = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.])
terminalModel = aslr_to.IntegratedActionModelEulerASR(aslr_to.NumDiffASRFwdDynamicsModel(terminalModel_a, 1e-4),0)

T = 100

q0 = np.array([0,0,0,0])
x0 = np.concatenate([q0, q0,pinocchio.utils.zero(state.nv)])

problem = crocoddyl.ShootingProblem(x0, [runningModel] * T, terminalModel)

# Creating the DDP solver for this OC problem, defining a logger
solver = crocoddyl.SolverFDDP(problem)
cameraTF = [2., 2.68, 0.54, 0.2, 0.62, 0.72, 0.22]
if WITHDISPLAY and WITHPLOT:
    display = crocoddyl.GepettoDisplay(four_dof, 4, 4, cameraTF)
    solver.setCallbacks([crocoddyl.CallbackLogger(), crocoddyl.CallbackVerbose(), crocoddyl.CallbackDisplay(display)])
elif WITHDISPLAY:
    display = crocoddyl.GepettoDisplay(four_dof, 4, 4, cameraTF)
    solver.setCallbacks([crocoddyl.CallbackVerbose(), crocoddyl.CallbackDisplay(display)])
elif WITHPLOT:
    solver.setCallbacks([
        crocoddyl.CallbackLogger(),
        crocoddyl.CallbackVerbose(),
    ])
else:
    solver.setCallbacks([crocoddyl.CallbackVerbose()])

if WITHDISPLAY:
    display = crocoddyl.GepettoDisplay(four_dof)
    display.robot.viewer.gui.addSphere('world/point', .05, [1., 0., 0., 1.])  # radius = .1, RGBA=1001
    display.robot.viewer.gui.applyConfiguration('world/point',
                                                target.tolist() + [0., 0., 0., 1.])  # xyz+quaternion
    display.robot.viewer.gui.refresh()

solver.setCallbacks([crocoddyl.CallbackLogger(), crocoddyl.CallbackVerbose() ])

xs = [x0] * (solver.problem.T + 1)
# us = solver.problem.quasiStatic([x0] * solver.problem.T)
# us = [np.zeros(4)]*(solver.problem.T)
solver.th_stop = 5e-5
# Solving it with the DDP algorithm
solver.solve([],[],500)

log = solver.getCallbacks()[0]
# u1 , u2 = aslr_to.u_squared(log)
print('Initial position = ', solver.problem.runningDatas.tolist()[0].differential.multibody.pinocchio.oMf[robot_model.getFrameId(
    "EE")].translation.T)


print('Finally reached = ', solver.problem.terminalData.differential.multibody.pinocchio.oMf[robot_model.getFrameId(
    "EE")].translation.T)

if WITHDISPLAY:
    while True:
        display.displayFromSolver(solver)
        time.sleep(2.0)

if WITHPLOT:
    log = solver.getCallbacks()[0]
    aslr_to.plotSEAOCSolution(log.xs,log.us,figIndex=1, show=True)

u1=np.array([])
u2=np.array([])

q1=np.array([])
q2=np.array([])
q3=np.array([])
q4=np.array([])
qm1=np.array([])
qm2=np.array([])
qm3=np.array([])
qm4=np.array([])

v1=np.array([])
v2=np.array([])
v3=np.array([])
v4=np.array([])
vm1=np.array([])
vm2=np.array([])
vm3=np.array([])
vm4=np.array([])

for i in range(len(log.us)):
    u1 = np.append(u1,log.us[i][0])
    u2 = np.append(u2,log.us[i][2])

for i in range(len(log.xs)):
    q1 = np.append(q1,log.xs[i][0])
    q2 = np.append(q2,log.xs[i][1])
    q3 = np.append(q3,log.xs[i][2])
    q4 = np.append(q4,log.xs[i][3])

    qm1 = np.append(qm1,log.xs[i][4])
    qm2 = np.append(qm2,log.xs[i][5])
    qm3 = np.append(qm3,log.xs[i][6])
    qm4 = np.append(qm4,log.xs[i][7])
    v1 = np.append(v1,log.xs[i][8])
    v2 = np.append(v2,log.xs[i][9])
    v3 = np.append(v3,log.xs[i][10])
    v4 = np.append(v4,log.xs[i][11])
    vm1 = np.append(vm1,log.xs[i][12])
    vm2 = np.append(vm2,log.xs[i][13])
    vm3 = np.append(vm3,log.xs[i][14])
    vm4 = np.append(vm4,log.xs[i][15])


t=np.arange(0,T*dt,dt)

savemat("optimised_trajectory.mat", {"q1": q1,"q2":q2, "q3": q3,"q4":q4,"qm1": qm1,"qm2":qm2,"qm3": qm3,"qm4":qm4,"v1": v1,"v2":v2,"v3": v3,"v4":v4,"vm1": vm1,"vm2":vm2,"vm3": vm3,"vm4":vm4,"t":t})
savemat("controls.mat", {"u1": u1,"u2":u2,"t":t})

K=solver.K.tolist()
K_temp = []
for i in range(len(K)):
    K_temp.append(np.linalg.norm(K[i]))

K[-1] = int(K_temp[-2]/K_temp[-3])*K[-2]
K[-2] = K[-3]
K[-1] = K[-2]
K_temp = []
for i in range(len(K)):
    K_temp.append(np.linalg.norm(K[i]))

plt.plot(K_temp)
plt.show()
savemat("fb.mat", {"K":K,"t":t})