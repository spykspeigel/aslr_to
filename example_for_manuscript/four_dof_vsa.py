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
WITHDISPLAY = 'display' in sys.argv or 'CROCODDYL_DISPLAY' in os.environ
WITHPLOT = 'plot' in sys.argv or 'CROCODDYL_PLOT' in os.environ

four_dof = example_robot_data.load('four_dof')
robot_model = four_dof.model

state = aslr_to.StateMultibodyASR(robot_model)
actuation = aslr_to.ActuationModelDoublePendulum(state, actLink=0, nu=4)
nu = 2*actuation.nu

target = np.array([.2, .3, .18])
framePlacementResidual = aslr_to.ResidualModelFramePlacementASR(state, robot_model.getFrameId("EE"),
                                                               pinocchio.SE3(np.eye(3), target), nu)

goalTrackingCost = crocoddyl.CostModelResidual(state, framePlacementResidual)

xActivation = crocoddyl.ActivationModelWeightedQuad(np.array([1e0] *4 + [1e0] *4 + [1e0] * robot_model.nv + [1e0]* robot_model.nv))
xResidual = crocoddyl.ResidualModelState(state, state.zero(), nu)
xRegCost = crocoddyl.CostModelResidual(state, xActivation, xResidual)
uActivation = crocoddyl.ActivationModelWeightedQuad(np.array([1e0]*4 + [1e0] * 4 ))
uResidual = crocoddyl.ResidualModelControl(state, nu)
uRegCost = crocoddyl.CostModelResidual(state, uActivation,uResidual)
xtermActivation = crocoddyl.ActivationModelWeightedQuad(np.array([0] * 4 + [0] * 4 + [1e0] * robot_model.nv + [1e0] * robot_model.nv))
xtermResidual = crocoddyl.ResidualModelState(state, state.zero(), nu)
xtermCost = crocoddyl.CostModelResidual(state, xActivation, xResidual)

lamda = 10
Kref = 5*np.ones(int(nu/2))
vsaCost = aslr_to.CostModelStiffness(state, nu, lamda,Kref)

runningCostModel = crocoddyl.CostModelSum(state,nu)
terminalCostModel = crocoddyl.CostModelSum(state,nu)

# Then let's added the running and terminal cost functions
runningCostModel.addCost("gripperPose", goalTrackingCost, 1e0)
runningCostModel.addCost("xReg", xRegCost, 1e-3)
runningCostModel.addCost("uReg", uRegCost, 1e-1)
# runningCostModel.addCost("vsa", vsaCost, 1e-2)
terminalCostModel.addCost("gripperPose", goalTrackingCost, 4e4)
# terminalCostModel.addCost("xReg", xRegCost, 1e1)
terminalCostModel.addCost("termVel", xtermCost, 1e0)

B = .001*np.eye(int(state.nv/2))

dt = 1e-2
runningModel = aslr_to.IntegratedActionModelEulerASR(
    aslr_to.DifferentialFreeFwdDynamicsModelVSA(state, actuation, runningCostModel,B), dt)
terminalModel = aslr_to.IntegratedActionModelEulerASR(
    aslr_to.DifferentialFreeFwdDynamicsModelVSA(state, actuation, terminalCostModel,B), 0)

l_lim_0=0
runningModel.u_lb = np.array([ -100, -100,-100,-100, .05, 5,.05,5])
runningModel.u_ub = np.array([ 100, 100, 100, 100, 100, 5,100,5])
T = 400

q0 = np.array([.0,.0,0,0])
x0 = np.concatenate([q0,np.zeros(4),pinocchio.utils.zero(state.nv)])

problem = crocoddyl.ShootingProblem(x0, [runningModel] * T, terminalModel)

# Creating the DDP solver for this OC problem, defining a logger
solver = crocoddyl.SolverBoxDDP(problem)
cameraTF = [2., 2.68, 0.54, 0.2, 0.62, 0.72, 0.22]

if WITHDISPLAY:
    display = crocoddyl.GepettoDisplay(four_dof)

solver.setCallbacks([crocoddyl.CallbackLogger(), crocoddyl.CallbackVerbose() ])

xs = [x0] * (solver.problem.T + 1)
us = solver.problem.quasiStatic([x0] * solver.problem.T)
solver.th_stop = 1e-7
# Solving it with the DDP algorithm
solver.solve([], [], 300)

print('Finally reached = ', solver.problem.terminalData.differential.multibody.pinocchio.oMf[robot_model.getFrameId(
    "EE")].translation.T)

log = solver.getCallbacks()[0]
aslr_to.plot_stiffness(log.us)

if WITHPLOT:
    log = solver.getCallbacks()[0]
    aslr_to.plotOCSolution(log.xs,log.us, stiffness=True,figIndex=1, show=True)
    #crocoddyl.plotConvergence(log.costs, log.u_regs, log.x_regs, log.grads, log.stops, log.steps, figIndex=2)

# Visualizing the solution in gepetto-viewer
if WITHDISPLAY:
    while True:
        display.displayFromSolver(solver)
        time.sleep(2.0)

# u1 , u2 = aslr_to.u_squared(log)
print("printing usquared")
print(np.sum(aslr_to.u_squared(log)))
# print("______")
# print(u2)

u1=np.array([])
u2=np.array([])
u3=np.array([])
u4=np.array([])

q1=np.array([])
q2=np.array([])
q3=np.array([])
q4=np.array([])
qm1=np.array([])
qm2=np.array([])
qm3=np.array([])
qm4=np.array([])

v1=np.array([])
v2=np.array([])
v3=np.array([])
v4=np.array([])
vm1=np.array([])
vm2=np.array([])
vm3=np.array([])
vm4=np.array([])

for i in range(len(log.us)):
    u1 = np.append(u1,log.us[i][0])
    u2 = np.append(u2,log.us[i][2])
    u3 = np.append(u3,log.us[i][4])
    u4 = np.append(u4,log.us[i][6])

for i in range(len(log.xs)):
    q1 = np.append(q1,log.xs[i][0])
    q2 = np.append(q2,log.xs[i][1])
    q3 = np.append(q3,log.xs[i][2])
    q4 = np.append(q4,log.xs[i][3])

    qm1 = np.append(qm1,log.xs[i][4])
    qm2 = np.append(qm2,log.xs[i][5])
    qm3 = np.append(qm3,log.xs[i][6])
    qm4 = np.append(qm4,log.xs[i][7])
    v1 = np.append(v1,log.xs[i][8])
    v2 = np.append(v2,log.xs[i][9])
    v3 = np.append(v3,log.xs[i][10])
    v4 = np.append(v4,log.xs[i][11])
    vm1 = np.append(vm1,log.xs[i][12])
    vm2 = np.append(vm2,log.xs[i][13])
    vm3 = np.append(vm3,log.xs[i][14])
    vm4 = np.append(vm4,log.xs[i][15])


t=np.arange(0,T*dt,dt)

savemat("optimised_trajectory_vsa.mat", {"q1": q1,"q2":q2, "q3": q3,"q4":q4,"qm1": qm1,"qm2":qm2,"qm3": qm3,"qm4":qm4,"v1": v1,"v2":v2,"v3": v3,"v4":v4,"vm1": vm1,"vm2":vm2,"vm3": vm3,"vm4":vm4,"t":t})

savemat("controls_vsa.mat", {"u1": u1,"u2":u2,"t":t})
savemat("stiffness_vsa.mat", {"u3": u3,"u4":u4,"t":t})

K=solver.K.tolist()
K_temp = []
for i in range(len(K)):
    K_temp.append(np.linalg.norm(K[i]))

K[-1] = int(K_temp[-2]/K_temp[-3])*K[-2]
K_temp = []
for i in range(len(K)):
    K_temp.append(np.linalg.norm(K[i]))

plt.plot(K_temp)
plt.show()
savemat("underactuated_vsa_fb.mat", {"K":K,"t":t})
