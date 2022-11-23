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

two_dof = example_robot_data.load('asr_twodof')
robot_model = two_dof.model
robot_model.gravity.linear = np.array([9.81,0,0])
state = aslr_to.StateMultibodyASR(robot_model)
actuation = aslr_to.VSAASRActuation(state)
nu = 2*actuation.nu
target = np.array([.01, .2, .18])
framePlacementResidual = aslr_to.ResidualModelFramePlacementASR(state, robot_model.getFrameId("EE"),
                                                               pinocchio.SE3(np.eye(3), target), nu)

goalTrackingCost = crocoddyl.CostModelResidual(state, framePlacementResidual)

xActivation = crocoddyl.ActivationModelWeightedQuad(np.array([1e0] *2 + [1e0] *2 + [1e0] * robot_model.nv + [1e0]* robot_model.nv))
xResidual = crocoddyl.ResidualModelState(state, state.zero(), nu)
xRegCost = crocoddyl.CostModelResidual(state, xActivation, xResidual)
uActivation = crocoddyl.ActivationModelWeightedQuad(np.array([1e0]+[1e0] + [1e0] * 2 ))
uResidual = crocoddyl.ResidualModelControl(state, nu)
uRegCost = crocoddyl.CostModelResidual(state, uActivation,uResidual)



xtermActivation = crocoddyl.ActivationModelWeightedQuad(np.array([0] * 2 + [0] * 2 + [1e0] * robot_model.nv + [1e0] * robot_model.nv))
xtermResidual = crocoddyl.ResidualModelState(state, state.zero(), nu)
xtermCost = crocoddyl.CostModelResidual(state, xActivation, xResidual)

lamda = 10
Kref = 0.001*np.ones(int(nu/2))
vsaCost = aslr_to.CostModelStiffness(state, nu, lamda,Kref)

runningCostModel = crocoddyl.CostModelSum(state,nu)
terminalCostModel = crocoddyl.CostModelSum(state,nu)

# Then let's added the running and terminal cost functions
runningCostModel.addCost("gripperPose", goalTrackingCost, 1e0)
runningCostModel.addCost("xReg", xRegCost, 1e-1)
runningCostModel.addCost("uReg", uRegCost, 5e-3)
# runningCostModel.addCost("vsa", vsaCost, 1e-2)
terminalCostModel.addCost("gripperPose", goalTrackingCost, 4e4)
# terminalCostModel.addCost("xReg", xtermCost, 1e-3)

B = .001*np.eye(int(state.nv/2))

dt = 1e-2
runningModel = aslr_to.IntegratedActionModelEulerASR(
    aslr_to.DifferentialFreeFwdDynamicsModelVSA(state, actuation, runningCostModel,B), dt)
terminalModel = aslr_to.IntegratedActionModelEulerASR(
    aslr_to.DifferentialFreeFwdDynamicsModelVSA(state, actuation, terminalCostModel,B), 0)

l_lim_0=0
runningModel.u_lb = np.array([ -10, -10, .05, .05])
runningModel.u_ub = np.array([ 10, 10, 7, 7])
T = 400

q0 = np.array([.0,.0])
x0 = np.concatenate([q0,np.zeros(2),pinocchio.utils.zero(state.nv)])

problem = crocoddyl.ShootingProblem(x0, [runningModel] * T, terminalModel)

# Creating the DDP solver for this OC problem, defining a logger
solver = crocoddyl.SolverBoxFDDP(problem)
cameraTF = [2., 2.68, 0.54, 0.2, 0.62, 0.72, 0.22]

if WITHDISPLAY:
    display = crocoddyl.GepettoDisplay(two_dof)
    display.robot.viewer.gui.addSphere('world/point', .05, [1., 0., 0., 1.])  # radius = .1, RGBA=1001
    display.robot.viewer.gui.applyConfiguration('world/point',
                                                target.tolist() + [0., 0., 0., 1.])  # xyz+quaternion
    display.robot.viewer.gui.refresh()

solver.setCallbacks([crocoddyl.CallbackLogger(), crocoddyl.CallbackVerbose() ])

xs = [x0] * (solver.problem.T + 1)
us = solver.problem.quasiStatic([x0] * solver.problem.T)
solver.th_stop = 1e-7
# Solving it with the DDP algorithm
solver.solve([], [], 400)

print('Finally reached = ', solver.problem.terminalData.differential.multibody.pinocchio.oMf[robot_model.getFrameId(
    "EE")].translation.T)

log = solver.getCallbacks()[0]
aslr_to.plot_stiffness(log.us)

# Visualizing the solution in gepetto-viewer
if WITHDISPLAY:
    while True:
        display.displayFromSolver(solver)
        time.sleep(2.0)

u1=np.array([])
u2=np.array([])
u3=np.array([])
u4=np.array([])

q1=np.array([])
q2=np.array([])
qm1=np.array([])
qm2=np.array([])
v1=np.array([])
v2=np.array([])
vm1=np.array([])
vm2=np.array([])

for i in range(len(log.us)):
    u1 = np.append(u1,log.us[i][0])
    u2 = np.append(u2,log.us[i][1])
    u3 = np.append(u3,log.us[i][2])
    u4 = np.append(u4,log.us[i][3])

for i in range(len(log.xs)):
    q1 = np.append(q1,log.xs[i][0])
    q2 = np.append(q2,log.xs[i][1])
    qm1 = np.append(qm1,log.xs[i][2])
    qm2 = np.append(qm2,log.xs[i][3])
    v1 = np.append(v1,log.xs[i][4])
    v2 = np.append(v2,log.xs[i][5])
    vm1 = np.append(vm1,log.xs[i][6])
    vm2 = np.append(vm2,log.xs[i][7])


t=np.arange(0,T*dt,dt)

savemat("optimised_trajectory_vsa.mat", {"q1": q1,"q2":q2,"qm1": qm1,"qm2":qm2,"v1": v1,"v2":v2,"vm1": vm1,"vm2":vm2,"t":t})
savemat("controls_vsa.mat", {"u1": u1,"u2":u2,"t":t})
savemat("stiffness_vsa.mat", {"u3": u3,"u4":u4,"t":t})

# if WITHPLOT:
#     log = solver.getCallbacks()[0]
#     aslr_to.plotOCSolution(log.xs, log.us, stiffness=True, figIndex=1, show=True)
#     #crocoddyl.plotConvergence(log.costs, log.u_regs, log.x_regs, log.grads, log.stops, log.steps, figIndex=2)


# u1 , u2 = aslr_to.u_squared(log)
print("printing usquared")
print(np.sum(aslr_to.u_squared(log)))
# print("______")
# print(u2)


K=solver.K.tolist()
K_temp = []
for i in range(len(K)):
    K_temp.append(np.linalg.norm(K[i]))


plt.plot(K_temp)
plt.show()
savemat("fb.mat", {"K":K,"t":t})
