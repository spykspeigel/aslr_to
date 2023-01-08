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
import statistics
WITHDISPLAY = 'display' in sys.argv or 'CROCODDYL_DISPLAY' in os.environ
WITHPLOT = 'plot' in sys.argv or 'CROCODDYL_PLOT' in os.environ

#WITHPLOT =True
two_dof = example_robot_data.load('asr_twodof')
robot_model = two_dof.model
# two_dof = example_robot_data.load('double_pendulum')
# robot_model = two_dof.model

robot_model.gravity.linear = np.array([9.81,0,0])
state = aslr_to.StateMultibodyASR(robot_model)
# actuation = aslr_to.ASRActuation(state)
actuation = aslr_to.ActuationModelDoublePendulum(state,actLink=0,nu=2)
nu = actuation.nu

runningCostModel = crocoddyl.CostModelSum(state,nu)
terminalCostModel = crocoddyl.CostModelSum(state,nu)

xActivation = crocoddyl.ActivationModelWeightedQuad(np.array([1e0] *2 + [1e0] *2 + [1e0] * robot_model.nv + [1e0]* robot_model.nv))
xResidual = crocoddyl.ResidualModelState(state, state.zero(), nu)
xRegCost = crocoddyl.CostModelResidual(state, xActivation, xResidual)
uResidual = crocoddyl.ResidualModelControl(state, nu)
uRegCost = crocoddyl.CostModelResidual(state, uResidual)

framePlacementResidual = aslr_to.ResidualModelFramePlacementASR(state, robot_model.getFrameId("EE"),
                                                               pinocchio.SE3(np.eye(3), np.array([.01, .2, .18])), nu)

# framePlacementResidual = aslr_to.ResidualModelFramePlacementASR(state, robot_model.getFrameId("EE"),
#                                                                pinocchio.SE3(np.eye(3), np.array([.01, .2, .18])), nu)                                                        

goalTrackingCost = crocoddyl.CostModelResidual(state, framePlacementResidual)
#xRegCost = crocoddyl.CostModelResidual(state, xResidual)

# Then let's added the running and terminal cost functions
runningCostModel.addCost("gripperPose", goalTrackingCost, 1e0)
runningCostModel.addCost("xReg", xRegCost, 1e-1)
runningCostModel.addCost("uReg", uRegCost, 1e-1)
terminalCostModel.addCost("gripperPose", goalTrackingCost, 4e5)


K = 5*np.eye(int(state.nv/2))
B = .001*np.eye(int(state.nv/2))

dt = 1e-3
runningModel = aslr_to.IntegratedActionModelEulerASR(
    aslr_to.DifferentialFreeASRFwdDynamicsModel(state, actuation, runningCostModel,K,B), dt)
#runningModel.differential.armature = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.])
terminalModel = aslr_to.IntegratedActionModelEulerASR(
    aslr_to.DifferentialFreeASRFwdDynamicsModel(state, actuation, terminalCostModel,K,B), 0)
#terminalModel.differential.armature = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.])

# runningModel_a = aslr_to.DifferentialFreeASRFwdDynamicsModel(state, actuation, runningCostModel,K,B)
# #runningModel.differential.armature = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.])
# runningModel = aslr_to.IntegratedActionModelEulerASR(aslr_to.NumDiffASRFwdDynamicsModel(runningModel_a, 1e-5),dt)
# terminalModel_a = aslr_to.DifferentialFreeASRFwdDynamicsModel(state, actuation, terminalCostModel,K,B)
# #terminalModel.differential.armature = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.])
# terminalModel = aslr_to.IntegratedActionModelEulerASR(aslr_to.NumDiffASRFwdDynamicsModel(terminalModel_a, 1e-5),0)

T = 400

q0_ = np.array([[0,0],[0,0.01],[0,-0.01],[0.01,0],[0.01,0.01],[-0.01,-0.01],[-0.02,0],[0.02,0],[0,0.02],[0,-0.02],[.02,.02],[-0.02,-0.02]])

costs = []
iterations = []
stops = []
not_working = []
avg_time = []
for i in range(10):
    q0 = q0_[i]
    x0 = np.concatenate([q0,q0,pinocchio.utils.zero(state.nv)])

    problem = crocoddyl.ShootingProblem(x0, [runningModel] * T, terminalModel)

    # Creating the DDP solver for this OC problem, defining a logger
    solver = crocoddyl.SolverFDDP(problem)
    cameraTF = [2., 2.68, 0.54, 0.2, 0.62, 0.72, 0.22]

    solver.setCallbacks([crocoddyl.CallbackLogger(), crocoddyl.CallbackVerbose() ])

    xs = [x0] * (solver.problem.T + 1)
    us = solver.problem.quasiStatic([x0] * solver.problem.T)
    solver.th_stop = 1e-5
    # Solving it with the DDP algorithm
    t1=time.time()
    convergence = solver.solve([],[],500)
    t2=time.time()
    print("time per iteration")
    print((t2-t1)/solver.iter)
    avg_time.append((t2-t1)/solver.iter)


    if solver.iter<150 and solver.isFeasible:
        iterations.append(solver.iter)
        costs.append(solver.cost)
        stops.append(solver.stoppingCriteria())
    else:
        not_working.append(x0)

avg_iterations = sum(iterations)/len(iterations)
std_iterations = statistics.stdev(iterations)

average_time = sum(avg_time)/len(avg_time)
print(average_time)