import os
import sys

import crocoddyl
import numpy as np
import example_robot_data
#from pendulum import CostModelDoublePendulum, ActuationModelDoublePendulum
import aslr_to
import time
from scipy.io import savemat

WITHDISPLAY = 'display' in sys.argv or 'CROCODDYL_DISPLAY' in os.environ
WITHPLOT = 'plot' in sys.argv or 'CROCODDYL_PLOT' in os.environ

# Loading the double pendulum model
# pendulum = example_robot_data.load('double_pendulum')
# model = pendulum.model
two_dof = example_robot_data.load('asr_twodof')
model = two_dof.model
model.gravity.linear = np.array([9.81,0,0])

state = aslr_to.StateMultibodyASR(model)

actuation = aslr_to.ActuationModelDoublePendulum(state, actLink=1)

nu = actuation.nu
runningCostModel = crocoddyl.CostModelSum(state, nu)
terminalCostModel = crocoddyl.CostModelSum(state, nu)

xActivation = crocoddyl.ActivationModelWeightedQuad(np.array([1e0] *2 + [1e0] *2 + [1e0] * model.nv + [1e0]* model.nv))
xResidual = crocoddyl.ResidualModelState(state, state.zero(), nu)
xRegCost = crocoddyl.CostModelResidual(state, xActivation, xResidual)
uResidual = crocoddyl.ResidualModelControl(state, nu)
uActivation = crocoddyl.ActivationModelWeightedQuad(np.array([1.] * actuation.nu))

uRegCost = crocoddyl.CostModelResidual(state, uActivation, uResidual)
xPendCost = aslr_to.CostModelDoublePendulum(state, crocoddyl.ActivationModelWeightedQuad(np.array([1.] * 4 + [0.1] * 2)), nu)


dt = 1e-2

runningCostModel.addCost("uReg", uRegCost, 1e-1)
runningCostModel.addCost("xReg", xRegCost, 1e-2)
runningCostModel.addCost("xGoalR", xPendCost, 1e0)

terminalCostModel.addCost("xGoal", xPendCost, 1e4)

K = 3*np.eye(int(state.nv/2))
B = .001*np.eye(int(state.nv/2))

runningModel = aslr_to.IntegratedActionModelEulerASR(
    aslr_to.DifferentialFreeASRFwdDynamicsModel(state, actuation, runningCostModel,K,B), dt)
#runningModel.differential.armature = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.])
terminalModel = aslr_to.IntegratedActionModelEulerASR(
    aslr_to.DifferentialFreeASRFwdDynamicsModel(state, actuation, terminalCostModel,K,B), dt)

# Creating the shooting problem and the solver
T = 100
x0 = np.array([0, 0., 0., 0., 0,0 ,0,0])
problem = crocoddyl.ShootingProblem(x0, [runningModel] * T, terminalModel)
problem.nthreads = 1  # TODO(cmastalli): Remove after Crocoddyl supports multithreading with Python-derived models
solver = crocoddyl.SolverFDDP(problem)

cameraTF = [1.4, 0., 0.2, 0.5, 0.5, 0.5, 0.5]
if WITHDISPLAY and WITHPLOT:
    display = crocoddyl.GepettoDisplay(two_dof, 4, 4, cameraTF, False)
    solver.setCallbacks([crocoddyl.CallbackLogger(), crocoddyl.CallbackVerbose(), crocoddyl.CallbackDisplay(display)])
elif WITHDISPLAY:
    display = crocoddyl.GepettoDisplay(two_dof, 4, 4, cameraTF, False)
    solver.setCallbacks([crocoddyl.CallbackVerbose(), crocoddyl.CallbackDisplay(display)])
elif WITHPLOT:
    solver.setCallbacks([crocoddyl.CallbackLogger(), crocoddyl.CallbackVerbose()])
else:
    solver.setCallbacks([crocoddyl.CallbackVerbose()])

# Solving the problem with the solver
solver.solve()

# log = solver.getCallbacks()[0]
# u1 , u2 = aslr_to.u_squared(log)
# print("printing usquared")
# print(u1)
# print("______")
# print(u2)
# # Plotting the entire motion
if WITHPLOT:
    log = solver.getCallbacks()[0]
    aslr_to.plotOCSolution(log.xs, log.us, figIndex=1, show=False)
    crocoddyl.plotConvergence(log.costs, log.u_regs, log.x_regs, log.grads, log.stops, log.steps, figIndex=2)

# Display the entire motion
if WITHDISPLAY:
    while True:
        display = crocoddyl.GepettoDisplay(two_dof, floor=False)
        display.displayFromSolver(solver)
        time.sleep(2)

u1=np.array([])
u2=np.array([])

x1=np.array([])
x2=np.array([])

for i in range(len(log.us)):
    u1 = np.append(u1,log.us[i][0])
    u2 = np.append(u2,log.us[i][1])

for i in range(len(log.xs)):
    x1 = np.append(x1,log.xs[i][0])
    x2 = np.append(x2,log.xs[i][1])

t=np.arange(0,T*dt,dt)

savemat("optimised_trajectory.mat", {"q1": x1,"q2":x2,"t":t})
savemat("controls.mat", {"u1": u1,"u2":u2,"t":t})
