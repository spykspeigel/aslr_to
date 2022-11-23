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
import time
WITHDISPLAY = 'display' in sys.argv or 'CROCODDYL_DISPLAY' in os.environ
WITHPLOT = 'plot' in sys.argv or 'CROCODDYL_PLOT' in os.environ

#WITHPLOT =True
talos_arm = example_robot_data.load('talos_arm')
robot_model = talos_arm.model
state = aslr_to.StateMultibodyASR(robot_model)
# actuation = aslr_to.ActuationModelDoublePendulum(state,actLink=0,nu=7)
actuation = aslr_to.ASRActuation(state)
nu = actuation.nu


xActivation = crocoddyl.ActivationModelWeightedQuad(np.array([1e0] * 7 + [1e0] * 7 + [1e0] * robot_model.nv + [1e0] * robot_model.nv))
xResidual = crocoddyl.ResidualModelState(state, state.zero(), nu)
xRegCost = crocoddyl.CostModelResidual(state, xActivation, xResidual)
uResidual = crocoddyl.ResidualModelControl(state, nu)
uRegCost = crocoddyl.CostModelResidual(state, uResidual)
costs = []
iterations = []
stops = []
not_working= []
not_working_stops=[]

t=time.time()
avg_time =[]
for i in range(30):
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


    K = 30*np.eye(int(state.nv/2))
    B = 1e-3*np.eye(int(state.nv/2))

    dt = 1e-2
    runningModel_a = aslr_to.DifferentialFreeASRFwdDynamicsModel(state, actuation, runningCostModel,K,B)
    runningModel = aslr_to.IntegratedActionModelEulerASR( aslr_to.NumDiffASRFwdDynamicsModel(runningModel_a, 1e-4), dt)


    terminalModel_a = aslr_to.DifferentialFreeASRFwdDynamicsModel(state, actuation, terminalCostModel,K,B)
    #terminalModel.differential.armature = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.])
    terminalModel = aslr_to.IntegratedActionModelEulerASR(aslr_to.NumDiffASRFwdDynamicsModel(terminalModel_a, 1e-4),0)
    T = 150

    q0 = np.array([0.173046, 1., -0.52366, 0., 0., 0.1, -0.005])
    x0 = np.concatenate([q0, q0,pinocchio.utils.zero(state.nv)])

    problem = crocoddyl.ShootingProblem(x0, [runningModel] * T, terminalModel)

    # Creating the DDP solver for this OC problem, defining a logger
    solver = crocoddyl.SolverFDDP(problem)
    solver.setCallbacks([crocoddyl.CallbackVerbose()])


    xs = [x0] * (solver.problem.T + 1)
    us = solver.problem.quasiStatic([x0] * solver.problem.T)
    solver.th_stop = 1e-5
    # Solving it with the DDP algorithm
    t1=time.time()
    solver.solve([],[],300)
    t2=time.time()

    if solver.iter<99 and solver.isFeasible:
        iterations.append(solver.iter)
        costs.append(solver.cost)
        stops.append(solver.stoppingCriteria())
    else:
        not_working.append(x0)


    print("time per iteration")
    print((t2-t1)/solver.iter)
    avg_time.append((t2-t1)/solver.iter)

t3=time.time()
print("average time per problem")
print((t2-t)/100)

print("average time per iteration")
print(sum(avg_time)/len(avg_time))

avg_iterations = sum(iterations)/len(iterations)
avg_costs = sum(costs)/len(costs)
avg_stops = sum(stops)/len(stops)
std_iterations = statistics.pstdev(iterations)
std_stops = statistics.pstdev(stops)
std_costs = statistics.pstdev(costs)

print("______________average iterations_______________")
print(avg_iterations)


print("______________standard deviation iterations_______________")
print(std_iterations)

print("______________not working_________________")
print(not_working)