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
import statistics
import matplotlib.pyplot as plt
WITHDISPLAY = 'display' in sys.argv or 'CROCODDYL_DISPLAY' in os.environ
WITHPLOT = 'plot' in sys.argv or 'CROCODDYL_PLOT' in os.environ

#WITHPLOT =True
three_dof = example_robot_data.load('three_dof')
robot_model = three_dof.model
state = aslr_to.StateMultibodyASR(robot_model)
actuation = aslr_to.ASRActuation(state)
# actuation = aslr_to.ActuationModelDoublePendulum(state,actLink=0,nu=3)
nu = actuation.nu


costs = []
iterations = []
stops = []
not_working=[]

for i in range(10):
    x=np.random.uniform(low=0.1,high=0.3)
    y=np.random.uniform(low=0.1,high=0.3)
    target = np.array([x, y, .18])
    framePlacementResidual = aslr_to.ResidualModelFramePlacementASR(state, robot_model.getFrameId("EE"),
                                                               pinocchio.SE3(np.eye(3), target), nu)

    runningCostModel = crocoddyl.CostModelSum(state,nu)
    terminalCostModel = crocoddyl.CostModelSum(state,nu)

    xActivation = crocoddyl.ActivationModelWeightedQuad(np.array([1e0] * 3 + [1e0] * 3 + [1e1] * robot_model.nv + [1e1] * robot_model.nv))
    xResidual = crocoddyl.ResidualModelState(state, state.zero(), nu)
    xRegCost = crocoddyl.CostModelResidual(state, xActivation, xResidual)

    xtermActivation = crocoddyl.ActivationModelWeightedQuad(np.array([0] * 3 + [0] * 3 + [1e0] * robot_model.nv + [1e0] * robot_model.nv))
    xtermResidual = crocoddyl.ResidualModelState(state, state.zero(), nu)
    xtermCost = crocoddyl.CostModelResidual(state, xActivation, xResidual)

    uResidual = crocoddyl.ResidualModelControl(state, nu)
    uRegCost = crocoddyl.CostModelResidual(state, uResidual)
    runningCostModel.addCost("xReg", xRegCost, 1e-2)
    runningCostModel.addCost("uReg", uRegCost, 1e-1)
    terminalCostModel.addCost("termVel", xtermCost, 1e-1)

    goalTrackingCost = crocoddyl.CostModelResidual(state, framePlacementResidual)
    #xRegCost = crocoddyl.CostModelResidual(state, xResidual)
    # Then let's added the running and terminal cost functions
    runningCostModel.addCost("gripperPose", goalTrackingCost, 1e-1)
    terminalCostModel.addCost("gripperPose", goalTrackingCost, 1e4)

    K = 5*np.eye(int(state.nv/2))
    B = 1e-3*np.eye(int(state.nv/2))

    dt = 1e-2
    runningModel = aslr_to.IntegratedActionModelEulerASR(
        aslr_to.DifferentialFreeASRFwdDynamicsModel(state, actuation, runningCostModel,K,B), dt)

    terminalModel = aslr_to.IntegratedActionModelEulerASR(
        aslr_to.DifferentialFreeASRFwdDynamicsModel(state, actuation, terminalCostModel,K,B), 0)

    T = 300

    q0 = np.array([0,0,0])
    x0 = np.concatenate([q0, q0,pinocchio.utils.zero(state.nv)])

    problem = crocoddyl.ShootingProblem(x0, [runningModel] * T, terminalModel)

    solver = crocoddyl.SolverFDDP(problem)

    xs = [x0] * (solver.problem.T + 1)

    solver.th_stop = 5e-5

    solver.solve([],[],500)


    if solver.iter<99 and solver.isFeasible:
        iterations.append(solver.iter)
        costs.append(solver.cost)
        stops.append(solver.stop)

    else:
        not_working.append(x0)

avg_iterations = sum(iterations)/len(iterations)
avg_costs = sum(costs)/len(costs)
avg_stops = sum(stops)/len(stops)
std_iterations = statistics.pstdev(iterations)
std_stops = statistics.pstdev(stops)
std_costs = statistics.pstdev(costs)
print("______________average cost_______________")
print(avg_costs)

print("______________average iterations_______________")
print(avg_iterations)

print("______________average stops_______________")
print(avg_stops)

print("______________standard deviation cost_______________")
print(std_costs)

print("______________standard deviation iterations_______________")
print(std_iterations)

print("______________standard deviation stops_______________")
print(std_stops)

print("++___________not working")
print(len(not_working))