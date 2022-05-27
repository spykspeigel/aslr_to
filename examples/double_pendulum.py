import os
import sys
import crocoddyl
import numpy as np
import example_robot_data
import aslr_to
import time
# from pendulum import CostModelDoublePendulum, ActuationModelDoublePendulum
WITHDISPLAY = 'display' in sys.argv or 'CROCODDYL_DISPLAY' in os.environ
WITHPLOT = 'plot' in sys.argv or 'CROCODDYL_PLOT' in os.environ

# Loading the double pendulum model
pendulum = example_robot_data.load('double_pendulum')
model = pendulum.model

state = aslr_to.StateMultibodyASR(model)

actuation = aslr_to.ActuationModelDoublePendulum(state, actLink=0)

nu = actuation.nu

runningCostModel = crocoddyl.CostModelSum(state, nu)
terminalCostModel = crocoddyl.CostModelSum(state, nu)

xActivation = crocoddyl.ActivationModelWeightedQuad(np.array([1e0] *2 + [0] *2 + [1e0] * model.nv + [0]* model.nv))
xResidual = crocoddyl.ResidualModelState(state, state.zero(), nu)
xRegCost = crocoddyl.CostModelResidual(state, xActivation, xResidual)

uResidual = crocoddyl.ResidualModelControl(state, nu)
uActivation = crocoddyl.ActivationModelWeightedQuad(np.array([1.,0.] ))
uRegCost = crocoddyl.CostModelResidual(state, uActivation, uResidual)
xPendCost = aslr_to.CostModelDoublePendulum(state, crocoddyl.ActivationModelWeightedQuad(np.array([1] * 4 + [.1] * 2)), nu)

dt = 1e-2

runningCostModel.addCost("uReg", uRegCost, 1e-1)
runningCostModel.addCost("xReg", xRegCost, 1e-2)
runningCostModel.addCost("xGoalR", xPendCost, 1e-1)

terminalCostModel.addCost("xGoal", xPendCost, 1e4)

K = 1*np.eye(int(state.nv/2))
B = .001*np.eye(int(state.nv/2))

runningModel = aslr_to.IntegratedActionModelEulerASR(
    aslr_to.DifferentialFreeASRFwdDynamicsModel(state, actuation, runningCostModel,K,B), dt)

terminalModel = aslr_to.IntegratedActionModelEulerASR(
    aslr_to.DifferentialFreeASRFwdDynamicsModel(state, actuation, terminalCostModel,K,B), 0)
# Creating the shooting problem and the solver
T = 10
x0 = np.array([3.14, 0., 0., 0., 0,0 ,0,0])
problem = crocoddyl.ShootingProblem(x0, [runningModel] * T, terminalModel)
problem.nthreads = 1  # TODO(cmastalli): Remove after Crocoddyl supports multithreading with Python-derived models
solver = crocoddyl.SolverFDDP(problem)

cameraTF = [1.4, 0., 0.2, 0.5, 0.5, 0.5, 0.5]
if WITHDISPLAY and WITHPLOT:
    display = crocoddyl.GepettoDisplay(pendulum, 4, 4, cameraTF, False)
    solver.setCallbacks([crocoddyl.CallbackLogger(), crocoddyl.CallbackVerbose(), crocoddyl.CallbackDisplay(display)])
elif WITHDISPLAY:
    display = crocoddyl.GepettoDisplay(pendulum, 4, 4, cameraTF, False)
    solver.setCallbacks([crocoddyl.CallbackVerbose(), crocoddyl.CallbackDisplay(display)])
elif WITHPLOT:
    solver.setCallbacks([crocoddyl.CallbackLogger(), crocoddyl.CallbackVerbose()])
else:
    solver.setCallbacks([crocoddyl.CallbackLogger(),crocoddyl.CallbackVerbose()])

# Solving the problem with the solver
solver.solve()
# log = solver.getCallbacks()[0]
# u1 , u2 = aslr_to.u_squared(log)
# print("printing usquared")
# print(aslr_to.u_squared(log))
# # Plotting the entire motion
if WITHPLOT:
    log = solver.getCallbacks()[0]
    aslr_to.plotOCSolution(log.xs, log.us, figIndex=1, show=False)
    crocoddyl.plotConvergence(log.costs, log.u_regs, log.x_regs, log.grads, log.stops, log.steps, figIndex=2)
# Display the entire motion
if WITHDISPLAY:
    while True:
        display = crocoddyl.GepettoDisplay(pendulum, floor=False)
        display.displayFromSolver(solver)
        time.sleep(2)

