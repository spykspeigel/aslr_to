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

#WITHPLOT =True
two_dof = example_robot_data.load('asr_twodof')
robot_model = two_dof.model
robot_model.gravity.linear = np.array([9.81,0,0])
state = crocoddyl.StateMultibody(robot_model)
actuation = aslr_to.ASRActuationCondensed(state, 6)
nu = actuation.nu

runningCostModel = crocoddyl.CostModelSum(state,nu)
terminalCostModel = crocoddyl.CostModelSum(state,nu)

# xResidual = crocoddyl.ResidualModelState(state, nu)
xActivation = crocoddyl.ActivationModelWeightedQuad(np.array([1e0] *2 + [1e0] *2 ))
xResidual = crocoddyl.ResidualModelState(state, state.zero(), nu)
xRegCost = crocoddyl.CostModelResidual(state, xActivation, xResidual)

uActivation = crocoddyl.ActivationModelWeightedQuad(np.array([1e0]*2+[1e1]*2 + [0] * 2 ))
uResidual = crocoddyl.ResidualModelControl(state, nu)
uRegCost = crocoddyl.CostModelResidual(state, uActivation,uResidual)

target = np.array([.0, .0, .4])
framePlacementResidual = crocoddyl.ResidualModelFramePlacement(state, robot_model.getFrameId("EE"),
                                                               pinocchio.SE3(np.eye(3), np.array([.01, .2, .18])), nu)
goalTrackingCost = crocoddyl.CostModelResidual(state, framePlacementResidual)
xRegCost = crocoddyl.CostModelResidual(state, xResidual)
costs = crocoddyl.CostModelSum(state, nu)

feas_residual = aslr_to.VSADynamicsResidualModel(state, nu )
lb = -3.14*np.ones(state.nv)
ub = 3.14*np.ones(state.nv)
activation = crocoddyl.ActivationModelQuadraticBarrier(crocoddyl.ActivationBounds(lb,ub))
feasCost = crocoddyl.CostModelResidual(state,activation ,feas_residual)
costs.addCost("feascost",feasCost,nu)

lamda = 10
Kref = np.zeros(int(nu/2))
vsaCost = aslr_to.CostModelStiffness(state, nu, lamda,Kref)
runningCostModel.addCost("vsa", vsaCost, 1e-3)
# Then let's added the running and terminal cost functions
runningCostModel.addCost("gripperPose", goalTrackingCost, 1e-1)
runningCostModel.addCost("xReg", xRegCost, 1e-3)
runningCostModel.addCost("uReg", uRegCost, 1e-1)
runningCostModel.addCost("feascost", feasCost, 1e2)

terminalCostModel.addCost("gripperPose", goalTrackingCost, 1e4)


dt = 1e-2
runningModel = crocoddyl.IntegratedActionModelEuler(
    crocoddyl.DifferentialActionModelFreeFwdDynamics(state, actuation, runningCostModel), dt)
terminalModel = crocoddyl.IntegratedActionModelEuler(
    crocoddyl.DifferentialActionModelFreeFwdDynamics(state, actuation, terminalCostModel), 0)


l_lim_0=0
runningModel.u_lb = np.array([ -100, -100, -1000,-1000, 0.1, 0.1])
runningModel.u_ub = np.array([ 100, 100, 1000, 1000,100,100])
T = 200

q0 = np.array([0.,0])
x0 = np.concatenate([q0,pinocchio.utils.zero(state.nv)])

problem = crocoddyl.ShootingProblem(x0, [runningModel] * T, terminalModel)

# Creating the DDP solver for this OC problem, defining a logger
solver = crocoddyl.SolverBoxDDP(problem)
cameraTF = [2., 2.68, 0.54, 0.2, 0.62, 0.72, 0.22]


if WITHDISPLAY:
    display = crocoddyl.GepettoDisplay(two_dof)

solver.setCallbacks([crocoddyl.CallbackLogger(), crocoddyl.CallbackVerbose(),inv_dyn.CallbackResidualLogger('feascost') ])

xs = [x0] * (solver.problem.T + 1)
# us = solver.problem.quasiStatic([x0] * solver.problem.T)
us = [np.array([0,0,0,0,10,10])]* (solver.problem.T ) 
solver.th_stop = 1e-7
# Solving it with the DDP algorithm
solver.solve(xs,us,400)

print('Finally reached = ', solver.problem.terminalData.differential.multibody.pinocchio.oMf[robot_model.getFrameId(
    "EE")].translation.T)

log = solver.getCallbacks()[2]
log1 = solver.getCallbacks()[0]
print( aslr_to.u_squared(log1))
aslr_to.plot_theta(log,1)
# Plotting the solution and the DDP convergence
if WITHPLOT:
    log = solver.getCallbacks()[0]
    crocoddyl.plotOCSolution(log.xs, log.us,figIndex=1, show=True)
    # crocoddyl.plotConvergence(log.costs, log.u_regs, log.x_regs, log.grads, log.stops, log.steps, figIndex=2)

# Visualizing the solution in gepetto-viewer
if WITHDISPLAY:
    while True:
        display.displayFromSolver(solver)
        time.sleep(2.0)