# already Saroj
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

# Michelino
# from sklearn.preprocessing import normalize
# import h5py
from numpy import linalg as LA


WITHDISPLAY = 'display' in sys.argv or 'CROCODDYL_DISPLAY' in os.environ
WITHPLOT = 'plot' in sys.argv or 'CROCODDYL_PLOT' in os.environ

#WITHPLOT =True
fishing_rod = example_robot_data.load('fishing_rod')
robot_model = fishing_rod.model
# robot_model.gravity.linear = np.array([0, 0, 9.81]) # assuming global frame urdf
state = crocoddyl.StateMultibody(robot_model)
actuation = aslr_to.ASRFishing(state)

nu = actuation.nu

runningCostModel = crocoddyl.CostModelSum(state,nu)
terminalCostModel = crocoddyl.CostModelSum(state,nu)

xResidual = crocoddyl.ResidualModelState(state, state.zero(), nu)
uResidual = crocoddyl.ResidualModelControl(state, nu)
uRegCost = crocoddyl.CostModelResidual(state, uResidual)

# target to be reached 
L0 = 3
alpha_des = np.pi/4
vel_cos = 10
target_pos = np.array([L0*np.cos(alpha_des), 0, L0*np.sin(alpha_des)])
target_vel = np.array([vel_cos*np.cos(alpha_des), 0, 0])

framePlacementResidual = crocoddyl.ResidualModelFramePlacement(state, robot_model.getFrameId("Link_EE"),
                                                               pinocchio.SE3(np.eye(3), target_pos),nu)
framePlacementVelocity = crocoddyl.ResidualModelFrameVelocity(state, robot_model.getFrameId("Link_EE"),
                                                               pinocchio.Motion(target_vel,np.zeros(3)),pinocchio.ReferenceFrame(45).WORLD,nu) 

goalTrackingCost = crocoddyl.CostModelResidual(state, framePlacementResidual)
goalVelCost = crocoddyl.CostModelResidual(state, framePlacementVelocity)
xRegCost = crocoddyl.CostModelResidual(state, xResidual)
# Then let's added the running and terminal cost functions
runningCostModel.addCost("gripperPose", goalTrackingCost, 1e-3)
# runningCostModel.addCost("gripperVel", goalVelCost, 1e-3)
runningCostModel.addCost("xReg", xRegCost, 1e-3)
runningCostModel.addCost("uReg", uRegCost, 1e-3)
terminalCostModel.addCost("gripperPose", goalTrackingCost, 1e5)
# terminalCostModel.addCost("gripperVel", goalVelCost, 1e5)

dt = 1e-4
T = 1000 
D = 1e-2*np.eye(state.nv)
K = 1e2*np.eye(state.nv)

runningModel = crocoddyl.IntegratedActionModelEuler(
  aslr_to.DAM2(state, actuation, runningCostModel, K, D), dt)
terminalModel = crocoddyl.IntegratedActionModelEuler(
  aslr_to.DAM2(state, actuation, terminalCostModel, K, D), 0)

# runningModel = crocoddyl.IntegratedActionModelEuler(
#     aslr_to.DAM2(state, actuation, runningCostModel), dt)
# terminalModel = crocoddyl.IntegratedActionModelEuler(
#     aslr_to.DAM2(state, actuation, terminalCostModel), dt)

q0 = fishing_rod.q0
# q0 = np.ones(21)
x0 = np.concatenate([q0,pinocchio.utils.zero(state.nv)])

problem = crocoddyl.ShootingProblem(x0, [runningModel] * T, terminalModel)

# Creating the DDP solver for this OC problem, defining a logger
solver = crocoddyl.SolverFDDP(problem)
cameraTF = [2., 2.68, 0.54, 0.2, 0.62, 0.72, 0.22]

if WITHDISPLAY:
    display = crocoddyl.GepettoDisplay(fishing_rod)

solver.setCallbacks([crocoddyl.CallbackLogger(), crocoddyl.CallbackVerbose() ])

xs = [x0] * (solver.problem.T + 1)
# us = solver.problem.quasiStatic([x0] * solver.problem.T)
u=np.zeros(1)
us = [u] * (solver.problem.T)
solver.th_stop = 1e-5
# Solving it with the DDP algorithm
solver.solve(xs, us, 100)
print('Initial position = ', solver.problem.runningDatas.tolist()[0].differential.multibody.pinocchio.oMf[robot_model.getFrameId(
    "Link_EE")].translation.T)
print('------------------------\n')
print('Finally reached = ', solver.problem.terminalData.differential.multibody.pinocchio.oMf[robot_model.getFrameId(
    "Link_EE")].translation.T)
print('------------------------\n')
final_pos = solver.problem.terminalData.differential.multibody.pinocchio.oMf[robot_model.getFrameId("Link_EE")].translation.T
err = LA.norm(final_pos - target_pos, 2)
print('Final Position error = ', err)

if err < 0.05:
    print("Error is OK")


if WITHPLOT:
    log = solver.getCallbacks()[0]
    crocoddyl.plotOCSolution(log.xs,log.us,figIndex=1, show=True)

# # Visualizing the solution in gepetto-viewer
if WITHDISPLAY:
    while True:
        display.displayFromSolver(solver)
        time.sleep(2.0)

u1 = np.array([]) # only the first joint is acutated 
data_q = np.zeros([len(log.xs),state.nq])
data_q_dot = np.zeros([len(log.xs),state.nq])

for i in range(len(log.us)):
    u1 = np.append(u1,log.us[i][0])

for i in range(len(log.xs)):
    for j in range(state.nq):
        data_q[i][j] = log.xs[i][j]
    for j1 in range(state.nq):
        data_q_dot[i][j1] = log.xs[i][j1 + state.nq]

t = np.arange(0,T*dt,dt)
savemat("optimised_trajectory_q_" + "1.mat", {"q": data_q,"t":t})
savemat("optimised_trajectory_q_dot_" + "1.mat", {"q_dot": data_q_dot,"t":t})
savemat("control_u1_" + "1.mat", {"u1": u1,"t":t})