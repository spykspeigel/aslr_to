import os
import sys
import crocoddyl
import pinocchio
import numpy as np
import aslr_to
import example_robot_data
import time
from scipy.io import savemat
import statistics
import matplotlib.pyplot as plt
WITHDISPLAY = 'display' in sys.argv or 'CROCODDYL_DISPLAY' in os.environ
WITHPLOT = 'plot' in sys.argv or 'CROCODDYL_PLOT' in os.environ

two_dof = example_robot_data.load('asr_twodof')
robot_model = two_dof.model
robot_model.gravity.linear = np.array([9.81,0,0])
state = aslr_to.StateMultibodyASR(robot_model)
actuation = aslr_to.ActuationModelDoublePendulum(state, actLink=0, nu=2)
nu = 2*actuation.nu

framePlacementResidual = aslr_to.ResidualModelFramePlacementASR(state, robot_model.getFrameId("EE"),
                                                               pinocchio.SE3(np.eye(3), np.array([.01, .2, .18])), nu)

goalTrackingCost = crocoddyl.CostModelResidual(state, framePlacementResidual)

xActivation = crocoddyl.ActivationModelWeightedQuad(np.array([1e0] *2 + [1e0] *2 + [1e0] * robot_model.nv + [1e0]* robot_model.nv))
xResidual = crocoddyl.ResidualModelState(state, state.zero(), nu)
xRegCost = crocoddyl.CostModelResidual(state, xActivation, xResidual)
uActivation = crocoddyl.ActivationModelWeightedQuad(np.array([1e0]+[1e0] + [1e0] * 2 ))
uResidual = crocoddyl.ResidualModelControl(state, nu)
uRegCost = crocoddyl.CostModelResidual(state, uActivation,uResidual)

lamda = 10
Kref = 3*np.ones(int(nu/2))
vsaCost = aslr_to.CostModelStiffness(state, nu, lamda,Kref)

runningCostModel = crocoddyl.CostModelSum(state,nu)
terminalCostModel = crocoddyl.CostModelSum(state,nu)

# Then let's added the running and terminal cost functions
runningCostModel.addCost("gripperPose", goalTrackingCost, 1e0)
runningCostModel.addCost("xReg", xRegCost, 1e-1)
runningCostModel.addCost("uReg", uRegCost, 1e-1)
# runningCostModel.addCost("vsa", vsaCost, 1e-2)
terminalCostModel.addCost("gripperPose", goalTrackingCost, 4e4)
# terminalCostModel.addCost("xReg", xRegCost, 1e1)

B = .001*np.eye(int(state.nv/2))

dt = 1e-2
runningModel = aslr_to.IntegratedActionModelEulerASR(
    aslr_to.DifferentialFreeFwdDynamicsModelVSA(state, actuation, runningCostModel,B), dt)
terminalModel = aslr_to.IntegratedActionModelEulerASR(
    aslr_to.DifferentialFreeFwdDynamicsModelVSA(state, actuation, terminalCostModel,B), 0)

# runningModel_a = aslr_to.DifferentialFreeFwdDynamicsModelVSA(state, actuation, runningCostModel,B)
# runningModel = aslr_to.IntegratedActionModelEulerASR(aslr_to.NumDiffASRFwdDynamicsModel(runningModel_a, 1e-6), dt)

# terminalModel_a = aslr_to.DifferentialFreeFwdDynamicsModelVSA(state, actuation, terminalCostModel,B)
# terminalModel = aslr_to.IntegratedActionModelEulerASR(aslr_to.NumDiffASRFwdDynamicsModel(terminalModel_a, 1e-6), 0)

l_lim_0=0
runningModel.u_lb = np.array([ -100, -100, .05, 2])
runningModel.u_ub = np.array([ 100, 100, 100, 2])
T = 300


q0_ = np.array([[0,0],[0,-0.01],[0.01,0],[0.01,0.01],[-0.01,-0.01],[-0.02,0],[0.02,0],[0,0.02],[0,-0.02],[.02,.02],[-0.02,-0.02]])

# q0_ = np.array([[.02,.02],[-0.02,-0.02]])
costs = []
iterations = []
stops = []
not_working = []
for i in range(2):
    q0 = q0_[i]
    x0 = np.concatenate([q0,q0,pinocchio.utils.zero(state.nv)])

    problem = crocoddyl.ShootingProblem(x0, [runningModel] * T, terminalModel)

    # Creating the DDP solver for this OC problem, defining a logger
    solver = crocoddyl.SolverBoxFDDP(problem)
    cameraTF = [2., 2.68, 0.54, 0.2, 0.62, 0.72, 0.22]

    solver.setCallbacks([crocoddyl.CallbackLogger(), crocoddyl.CallbackVerbose() ])

    xs = [x0] * (solver.problem.T + 1)
    us = solver.problem.quasiStatic([x0] * solver.problem.T)
    solver.th_stop = 1e-5
    # Solving it with the DDP algorithm
    solver.solve(xs, us, 400)

    if solver.iter<150 and solver.isFeasible:
        iterations.append(solver.iter)
        costs.append(solver.cost)
        stops.append(solver.stoppingCriteria())
    else:
        not_working.append(x0)

avg_iterations = sum(iterations)/len(iterations)
std_iterations = statistics.stdev(iterations)

print(avg_iterations)
print(std_iterations)