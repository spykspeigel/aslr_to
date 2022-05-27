from __future__ import print_function

import os
import sys

import crocoddyl
import pinocchio
import numpy as np
import example_robot_data
import aslr_to
import time
import inv_dyn
WITHDISPLAY = 'display' in sys.argv or 'CROCODDYL_DISPLAY' in os.environ
WITHPLOT = 'plot' in sys.argv or 'CROCODDYL_PLOT' in os.environ

#WITHPLOT = True
two_dof = example_robot_data.load('asr_twodof')
robot_model = two_dof.model
robot_model.gravity.linear = np.array([9.81,0,0])
state = crocoddyl.StateMultibody(robot_model)
K = 10*np.eye(state.nv)
B = .1*np.eye(state.nv)
actuation = aslr_to.ASRActuationCondensed(state,4,B)
nu = actuation.nu

runningCostModel = crocoddyl.CostModelSum(state,nu)
terminalCostModel = crocoddyl.CostModelSum(state,nu)

xResidual = crocoddyl.ResidualModelState(state, nu)
uResidual = crocoddyl.ResidualModelControl(state, nu)
uRegCost = crocoddyl.CostModelResidual(state, uResidual)
target = np.array([.0, .0, .4])
framePlacementResidual = crocoddyl.ResidualModelFramePlacement(state, robot_model.getFrameId("EE"),
                                                               pinocchio.SE3(np.eye(3), np.array([.01, .2, .18])), nu)
goalTrackingCost = crocoddyl.CostModelResidual(state, framePlacementResidual)
xRegCost = crocoddyl.CostModelResidual(state, xResidual)
costs = crocoddyl.CostModelSum(state, nu)


# print(nu)
feas_residual = aslr_to.SoftDynamicsResidualModel(state, nu, K, B)
lb = -3.14*K[0,0]*np.ones(state.nv)
ub = 3.14*K[0,0]*np.ones(state.nv)
activation = crocoddyl.ActivationModelQuadraticBarrier(crocoddyl.ActivationBounds(lb,ub))
feasCost = crocoddyl.CostModelResidual(state,activation ,feas_residual)
costs.addCost("feascost",feasCost,nu)

# Then let's added the running and terminal cost functions
runningCostModel.addCost("gripperPose", goalTrackingCost, 1e0)
runningCostModel.addCost("xReg", xRegCost, 1e-2)
runningCostModel.addCost("uReg", uRegCost, 1e0)
runningCostModel.addCost("feascost", feasCost, 1e4)

terminalCostModel.addCost("gripperPose", goalTrackingCost, 1e4)

dt = 1e-5
runningModel = crocoddyl.IntegratedActionModelEuler(
    crocoddyl.DifferentialActionModelFreeFwdDynamics(state, actuation, runningCostModel), dt)
terminalModel = crocoddyl.IntegratedActionModelEuler(
    crocoddyl.DifferentialActionModelFreeFwdDynamics(state, actuation, terminalCostModel), 0)

T = 10000

q0 = np.array([0.,0])
x0 = np.concatenate([q0,pinocchio.utils.zero(state.nv)])

problem = crocoddyl.ShootingProblem(x0, [runningModel] * T, terminalModel)

# Creating the DDP solver for this OC problem, defining a logger
solver = crocoddyl.SolverDDP(problem)
cameraTF = [2., 2.68, 0.54, 0.2, 0.62, 0.72, 0.22]


if WITHDISPLAY:
    display = crocoddyl.GepettoDisplay(two_dof)

solver.setCallbacks([crocoddyl.CallbackLogger(), crocoddyl.CallbackVerbose(),inv_dyn.CallbackResidualLogger('feascost') ])

xs = [x0] * (solver.problem.T + 1)
# us = solver.problem.quasiStatic([x0] * solver.problem.T)
us = [np.zeros(4)]* (solver.problem.T )
solver.th_stop = 1e-7
# Solving it with the DDP algorithm
solver.solve(xs,us,400)

print('Finally reached = ', solver.problem.terminalData.differential.multibody.pinocchio.oMf[robot_model.getFrameId(
    "EE")].translation.T)

log = solver.getCallbacks()[2]
log1 = solver.getCallbacks()[0]
print(np.sum(aslr_to.u_squared(log1)[:2]))
# print("printing usquared")
# print(u1)
# print("______")
# print(u2)
# inv_dyn.plot_residual(log,1)
aslr_to.plot_theta(log,1)
# Plotting the solution and the DDP convergence
if WITHPLOT:
    log = solver.getCallbacks()[0]
    aslr_to.plotOCSolution(log.xs, log.us,figIndex=1, show=True)
    # crocoddyl.plotConvergence(log.costs, log.u_regs, log.x_regs, log.grads, log.stops, log.steps, figIndex=2)

# Visualizing the solution in gepetto-viewer
if WITHDISPLAY:
    while True:
        display.displayFromSolver(solver)
        time.sleep(2.0)

u=np.array([])
for i in range(len(log1.us)):
    u = np.append(u,log1.us[i][3])

i=-1
th =np.array([])
for j in range(len(log.residual[i])):
    th=np.append(th,log.residual[i][j][1])

a=[]
for t in np.arange(2000,6998):
    a.append(((th[t+1]-2*th[t]+ th[t-1])/(dt**2)))

import matplotlib.pyplot as plt
plt.plot(a,label="numdiff")
plt.plot(u,label="analytical")
plt.legend()
plt.show()