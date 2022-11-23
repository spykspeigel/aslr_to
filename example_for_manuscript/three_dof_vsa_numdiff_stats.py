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
actuation = aslr_to.VSAASRActuation(state)

nu = 2*actuation.nu

costs = []
iterations = []
stops = []
not_working=[]

for i in range(20):
    x=np.random.uniform(low=0.1,high=0.3)
    y=np.random.uniform(low=0.1,high=0.3)
    target = np.array([x, y, .18])

    framePlacementResidual = aslr_to.ResidualModelFramePlacementASR(state, robot_model.getFrameId("EE"),
                                                                pinocchio.SE3(np.eye(3), target), nu)

    goalTrackingCost = crocoddyl.CostModelResidual(state, framePlacementResidual)

    xActivation = crocoddyl.ActivationModelWeightedQuad(np.array([1e0] *3 + [1e0] *3 + [1e0] * robot_model.nv + [1e0]* robot_model.nv))
    xResidual = crocoddyl.ResidualModelState(state, state.zero(), nu)
    xRegCost = crocoddyl.CostModelResidual(state, xActivation, xResidual)
    uActivation = crocoddyl.ActivationModelWeightedQuad(np.array([1e0]*3 + [1e-1] * 3 ))
    uResidual = crocoddyl.ResidualModelControl(state, nu)
    uRegCost = crocoddyl.CostModelResidual(state, uActivation,uResidual)
    xtermActivation = crocoddyl.ActivationModelWeightedQuad(np.array([0] * 3 + [0] * 3 + [1e0] * robot_model.nv + [1e0] * robot_model.nv))
    xtermResidual = crocoddyl.ResidualModelState(state, state.zero(), nu)
    xtermCost = crocoddyl.CostModelResidual(state, xActivation, xResidual)

    lamda = 10
    Kref = 5*np.ones(int(nu/2))
    vsaCost = aslr_to.CostModelStiffness(state, nu, lamda,Kref)

    runningCostModel = crocoddyl.CostModelSum(state,nu)
    terminalCostModel = crocoddyl.CostModelSum(state,nu)
    print(nu)
    # Then let's added the running and terminal cost functions
    runningCostModel.addCost("gripperPose", goalTrackingCost, 1e-1)
    runningCostModel.addCost("xReg", xRegCost, 1e-3)
    runningCostModel.addCost("uReg", uRegCost, 1e-1)
    # runningCostModel.addCost("vsa", vsaCost, 1e-2)
    terminalCostModel.addCost("gripperPose", goalTrackingCost, 1e3)
    # terminalCostModel.addCost("xReg", xRegCost, 1e1)
    # terminalCostModel.addCost("termVel", xtermCost, 1e0)

    B = .001*np.eye(int(state.nv/2))


    dt = 1e-2
    runningModel_a =  aslr_to.DifferentialFreeFwdDynamicsModelVSA(state, actuation, runningCostModel,B)
    #runningModel.differential.armature = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.])
    runningModel = aslr_to.IntegratedActionModelEulerASR(aslr_to.NumDiffASRFwdDynamicsModel(runningModel_a, 1e-7),dt)
    terminalModel_a = aslr_to.DifferentialFreeFwdDynamicsModelVSA(state, actuation, runningCostModel,B)
    #terminalModel.differential.armature = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.])
    terminalModel = aslr_to.IntegratedActionModelEulerASR(aslr_to.NumDiffASRFwdDynamicsModel(terminalModel_a, 1e-7),0)

    T = 300

    l_lim_0=0
    print(runningModel_a.nu)
    runningModel.u_lb = np.array([ -100, -100,-100, .05,.05,0.05])
    runningModel.u_ub = np.array([ 100,  100, 100,100,100,100])

    q0 = np.array([.0,.0,0])
    x0 = np.concatenate([q0,np.zeros(3),pinocchio.utils.zero(state.nv)])

    problem = crocoddyl.ShootingProblem(x0, [runningModel] * T, terminalModel)

    # Creating the DDP solver for this OC problem, defining a logger
    solver = crocoddyl.SolverBoxFDDP(problem)

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