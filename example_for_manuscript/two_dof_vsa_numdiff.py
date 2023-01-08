import os
import sys
import crocoddyl
import pinocchio
import numpy as np
import aslr_to
import example_robot_data
import time
from scipy.io import savemat
import matplotlib.pyplot as plt
import statistics
WITHDISPLAY = 'display' in sys.argv or 'CROCODDYL_DISPLAY' in os.environ
WITHPLOT = 'plot' in sys.argv or 'CROCODDYL_PLOT' in os.environ

two_dof = example_robot_data.load('asr_twodof')
robot_model = two_dof.model
robot_model.gravity.linear = np.array([9.81,0,0])
state = aslr_to.StateMultibodyASR(robot_model)
actuation = aslr_to.VSAASRActuation(state)
nu = 2*actuation.nu



xActivation = crocoddyl.ActivationModelWeightedQuad(np.array([1e0] *2 + [1e0] *2 + [1e0] * robot_model.nv + [1e0]* robot_model.nv))
xResidual = crocoddyl.ResidualModelState(state, state.zero(), nu)
xRegCost = crocoddyl.CostModelResidual(state, xActivation, xResidual)
uActivation = crocoddyl.ActivationModelWeightedQuad(np.array([1e0]+[1e0] + [1e0] * 2 ))
uResidual = crocoddyl.ResidualModelControl(state, nu)
uRegCost = crocoddyl.CostModelResidual(state, uActivation,uResidual)

lamda = 10
Kref = 0.001*np.ones(int(nu/2))
vsaCost = aslr_to.CostModelStiffness(state, nu, lamda,Kref)

costs = []
iterations = []
stops = []
not_working = []
avg_time = []
for i in range(2):
    print(i)
    target = np.array([np.random.uniform(low=.1,high= .2),np.random.uniform(low=.1,high= .2), .18])
    framePlacementResidual = aslr_to.ResidualModelFramePlacementASR(state, robot_model.getFrameId("EE"),
                                                               pinocchio.SE3(np.eye(3), target), nu)

    goalTrackingCost = crocoddyl.CostModelResidual(state, framePlacementResidual)

    runningCostModel = crocoddyl.CostModelSum(state,nu)
    terminalCostModel = crocoddyl.CostModelSum(state,nu)

    # Then let's added the running and terminal cost functions
    runningCostModel.addCost("gripperPose", goalTrackingCost, 1e-1)
    runningCostModel.addCost("xReg", xRegCost, 1e-2)
    runningCostModel.addCost("uReg", uRegCost, 1e-3)
    # runningCostModel.addCost("vsa", vsaCost, 1e-2)
    terminalCostModel.addCost("gripperPose", goalTrackingCost, 4e4)
    terminalCostModel.addCost("xReg", xRegCost, 1e-2)

    B = .001*np.eye(int(state.nv/2))

    dt = 1e-2
    # runningModel_a = aslr_to.DifferentialFreeFwdDynamicsModelVSA(state, actuation, runningCostModel,B)
    # runningModel = aslr_to.IntegratedActionModelEulerASR(aslr_to.NumDiffASRFwdDynamicsModel(runningModel_a, 1e-6), dt)

    # terminalModel_a = aslr_to.DifferentialFreeFwdDynamicsModelVSA(state, actuation, terminalCostModel,B)
    # terminalModel = aslr_to.IntegratedActionModelEulerASR(aslr_to.NumDiffASRFwdDynamicsModel(terminalModel_a, 1e-6), 0)


    runningModel = aslr_to.IntegratedActionModelEulerASR(
        aslr_to.DifferentialFreeFwdDynamicsModelVSA(state, actuation, runningCostModel,B), dt)
    terminalModel = aslr_to.IntegratedActionModelEulerASR(
        aslr_to.DifferentialFreeFwdDynamicsModelVSA(state, actuation, terminalCostModel,B), 0)

    l_lim_0=0
    runningModel.u_lb = np.array([ -10, -10, .05, .05])
    runningModel.u_ub = np.array([ 10, 10, 7, 7])
    T = 200

    q0 = np.array([.0,.0])
    x0 = np.concatenate([q0,np.zeros(2),pinocchio.utils.zero(state.nv)])

    problem = crocoddyl.ShootingProblem(x0, [runningModel] * T, terminalModel)

    # Creating the DDP solver for this OC problem, defining a logger
    solver = crocoddyl.SolverBoxFDDP(problem)

    solver.setCallbacks([crocoddyl.CallbackLogger(), crocoddyl.CallbackVerbose() ])

    xs = [x0] * (solver.problem.T + 1)
    us = solver.problem.quasiStatic([x0] * solver.problem.T)
    solver.th_stop = 1e-4
    # Solving it with the DDP algorithm
    solver.solve([], [], 400)

    if solver.iter<150 and solver.isFeasible:
        iterations.append(solver.iter)
        costs.append(solver.cost)
        stops.append(solver.stoppingCriteria())
    else:
        not_working.append(x0)
    t1=time.time()
    convergence = solver.solve([],[],500)
    t2=time.time()
    print("time per iteration")
    print((t2-t1)/solver.iter)
    avg_time.append((t2-t1)/solver.iter)


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
