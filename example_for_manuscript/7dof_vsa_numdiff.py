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
robot_model.gravity.linear = np.array([9.81,0,0])
state = aslr_to.StateMultibodyASR(robot_model)
actuation = aslr_to.VSAASRActuation(state)

nu = 2*actuation.nu


xActivation = crocoddyl.ActivationModelWeightedQuad(np.array([1e0] * 7 + [1e0] * 7 + [1e0] * robot_model.nv+[1e0]* robot_model.nv))
xResidual = crocoddyl.ResidualModelState(state, state.zero(), nu)
xRegCost = crocoddyl.CostModelResidual(state, xActivation, xResidual)
uActivation = crocoddyl.ActivationModelWeightedQuad(np.array([1e0]*7 + [1e0] * 7 ))
uResidual = crocoddyl.ResidualModelControl(state, nu)
uRegCost = crocoddyl.CostModelResidual(state, uActivation,uResidual)
avg_time = []
for i in range(10):
    # target = np.array([.0, .0, .4])
    target = np.array([np.random.uniform(low=0.1, high=0.2), np.random.uniform(low=0.1,high=0.3), np.random.uniform(low=0.1,high=0.3)])
    framePlacementResidual = aslr_to.ResidualModelFramePlacementASR(state, robot_model.getFrameId("gripper_left_joint"),
                                                                pinocchio.SE3(np.eye(3), target), nu)
    goalTrackingCost = crocoddyl.CostModelResidual(state, framePlacementResidual)
    #xRegCost = crocoddyl.CostModelResidual(state, xResidual)

    runningCostModel = crocoddyl.CostModelSum(state,nu)
    terminalCostModel = crocoddyl.CostModelSum(state,nu)
    # Then let's added the running and terminal cost functions
    runningCostModel.addCost("gripperPose", goalTrackingCost, 1e-1)
    runningCostModel.addCost("xReg", xRegCost, 1e-1)
    runningCostModel.addCost("uReg", uRegCost, 1e-2)
    terminalCostModel.addCost("gripperPose", goalTrackingCost, 4e4)

    K = 10*np.eye(int(state.nv/2))
    B = .001*np.eye(int(state.nv/2))

    dt = 1e-2
    runningModel = aslr_to.IntegratedActionModelEulerASR(
        aslr_to.DifferentialFreeFwdDynamicsModelVSA(state, actuation, runningCostModel,B), dt)
    #runningModel.differential.armature = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.])
    terminalModel = aslr_to.IntegratedActionModelEulerASR(
        aslr_to.DifferentialFreeFwdDynamicsModelVSA(state, actuation, terminalCostModel,B), 0)
    #terminalModel.differential.armature = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.])
    runningModel.u_lb = np.array([ -10, -10, -10,-10,-10,-10,-10,.05, .05, 0.05,.05, .05,0.05,.05])
    runningModel.u_ub = np.array([ 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10])
    T = 200

    q0 = np.array([0.173046, 1., -0.52366, 0., 0., 0.1, -0.005])
    x0 = np.concatenate([q0, q0,pinocchio.utils.zero(state.nv)])

    problem = crocoddyl.ShootingProblem(x0, [runningModel] * T, terminalModel)

    # Creating the DDP solver for this OC problem, defining a logger
    solver = crocoddyl.SolverBoxFDDP(problem)

    solver.setCallbacks([crocoddyl.CallbackLogger(), crocoddyl.CallbackVerbose() ])

    xs = [x0] * (solver.problem.T + 1)
    #us = solver.problem.quasiStatic([x0] * solver.problem.T)

    solver.th_stop = 1e-5
# Solving it with the DDP algorithm
    t1=time.time()
    convergence = solver.solve([],[],100)
    t2=time.time()
    print("time per iteration")
    print((t2-t1)/solver.iter)
    avg_time.append((t2-t1)/solver.iter)

print('Finally reached = ', solver.problem.terminalData.differential.multibody.pinocchio.oMf[robot_model.getFrameId(
    "gripper_left_joint")].translation.T)

# Plotting the solution and the DDP convergence
if WITHPLOT:
    log = solver.getCallbacks()[0]
    aslr_to.plotOCSolution(log.xs, log.us,stiffness= True, figIndex=1, show=True)
    #crocoddyl.plotConvergence(log.costs, log.u_regs, log.x_regs, log.grads, log.stops, log.steps, figIndex=2)

# Visualizing the solution in gepetto-viewer
if WITHDISPLAY:
    while True:
        display.displayFromSolver(solver)
        time.sleep(2.0)
log = solver.getCallbacks()[0]
