import os
import sys
import crocoddyl
import pinocchio
import numpy as np
# import monoped
# import actuation, slice_model
# from utils import plotOCSolution, plotConvergence, plot_frame_trajectory, animateMonoped, plot_power
# from power_costs import CostModelJointFriction, CostModelJointFrictionSmooth, CostModelJouleDissipation
# import modify_model
import conf
import example_robot_data
import aslr_to
T = conf.T
dt = conf.dt

# GRAVITY
# rmodel.gravity.linear = np.zeros(3)

# INITIAL CONFIGURATION
# to make the robot start with det(J) != 0 more options are given
softleg = example_robot_data.load("softleg")
rmodel = softleg.model
q0 = np.zeros(1 + conf.n_links)
rdata = rmodel.createData()
state = crocoddyl.StateSoftMultibody(rmodel)
K = np.zeros([state.pinocchio.nv,state.pinocchio.nq])
nu = state.nv_m
K[-nu:,-nu:]= 40*np.eye(nu)
B = .001*np.eye(state.nv_m)
actuation = aslr_to.SoftLegActuation(state)

# OPTION 1 Select the angle of the first joint wrt vertical
# angle = np.pi/4
# q0[0] = 2 * np.cos(angle)
# q0[1] = np.pi - angle
# q0[2] = 2 * angle

# OPTION 2 Initial configuration distributing the joints in a semicircle with foot in O (scalable if n_joints > 2)
# q0[0] = 0.16 / np.sin(np.pi/(2 * conf.n_links))
# q0[1:] = np.pi/conf.n_links
# q0[1] = np.pi/2 + np.pi/(2 * conf.n_links)

# OPTION 3 Solo, (the convention used has negative displacements)
q0[0] = 0.16 / np.sin(np.pi/(2 * conf.n_links))
q0[1] = np.pi/4
q0[2] = -np.pi/2

x0 = np.concatenate([q0, np.zeros(rmodel.nv), np.zeros(4)])
# COSTS
# Create a cost model for the running and terminal action model
# Setting the final position goal with variable angle
# angle = np.pi/2
# s = np.sin(angle)
# c = np.cos(angle)
# R = np.matrix([ [c,  0, s],
#                 [0,  1, 0],
#                 [-s, 0, c]
#              ])
# target = np.array([np.sin(angle), 0, np.cos(angle)]))
runningCostModel = crocoddyl.CostModelSum(state, actuation.nu)
terminalCostModel = crocoddyl.CostModelSum(state, actuation.nu)
target = np.array(conf.target)
footName = "softleg_1_contact_link"
footFrameID = rmodel.getFrameId(footName)

Pref = crocoddyl.FrameTranslation(footFrameID, target)
# If also the orientation is useful for the task use
footTrackingCost = crocoddyl.CostModelFrameTranslation(state, Pref, actuation.nu)
Vref = crocoddyl.FrameMotion(footFrameID, pinocchio.Motion(np.zeros(6)))
footFinalVelocity = crocoddyl.CostModelFrameVelocity(state, Vref, actuation.nu)
# simulating the cost on the power with a cost on the control

# PENALIZATIONS
bounds = crocoddyl.ActivationBounds(np.concatenate([np.zeros(1), -1e3* np.ones(state.nx-1)]), 1e3*np.ones(state.nx))
stateAct = crocoddyl.ActivationModelWeightedQuadraticBarrier(bounds, np.concatenate([np.ones(1), np.zeros(state.nx - 1)]))
nonPenetration = crocoddyl.CostModelState(state, stateAct, np.zeros(state.nx), actuation.nu)

# MAXIMIZATION
jumpBounds = crocoddyl.ActivationBounds(-1e3*np.ones(state.nx), np.concatenate([np.zeros(1), +1e3* np.ones(state.nx-1)]))
jumpAct = crocoddyl.ActivationModelWeightedQuadraticBarrier(bounds, np.concatenate([-np.ones(1), np.zeros(state.nx - 1)]))
maximizeJump = crocoddyl.CostModelState(state, jumpAct, np.ones(state.nx), actuation.nu)

# CONTACT MODEL
contactModel = crocoddyl.ContactModelMultiple(state, actuation.nu)
contact_location = crocoddyl.FrameTranslation(footFrameID, np.array([0., 0., 0.]))
supportContactModel = crocoddyl.ContactModel2D(state, contact_location, actuation.nu, np.array([0., 1/dt])) # makes the velocity drift disappear in one timestep
contactModel.addContact("foot_contact", supportContactModel)

# FRICTION CONE
# the friction cone can also have the [min, maximum] force parameters
# 4 is the number of faces for the approximation
mu = 0.7
normalDirection = np.array([0, 0, 1])
minForce = 0

# Creating the action model for the KKT dynamics with simpletic Euler integration scheme
contactCostModel = crocoddyl.CostModelSum(state, actuation.nu)
# contactCostModel.addCost('frictionCone', frictionCone, 1e-6)
# contactCostModel.addCost('joule_dissipation', joule_dissipation, 5e-3)
# contactCostModel.addCost('joint_friction', joint_friction, 5e-3)
contactCostModel.addCost('velocityRegularization', v2, 1e-1)
contactCostModel.addCost('nonPenetration', nonPenetration, 1e5)

contactDifferentialModel = aslr_to.DifferentialContactASLRFwdDynModel(state,
        actuation,
        contactModel,
        contactCostModel,
        K, B) # bool enable force
contactPhase = crocoddyl.IntegratedActionModelEuler(contactDifferentialModel, dt)

# runningCostModel.addCost("joule_dissipation", joule_dissipation, 5e-3)
# runningCostModel.addCost('joint_friction', joint_friction, 5e-3)
# runningCostModel.addCost("velocityRegularization", v2, 1e0)
runningCostModel.addCost("nonPenetration", nonPenetration, 1e6)
# runningCostModel.addCost("maxJump", maximizeJump, 1e2)
terminalCostModel.addCost("footPose", footTrackingCost, 5e3)
# terminalCostModel.addCost("footVelocity", footFinalVelocity, 1e0)

runningModel = crocoddyl.IntegratedActionModelEuler(
    crocoddyl.DifferentialActionModelFreeFwdDynamics(state, actuation, runningCostModel), dt)
terminalModel = crocoddyl.IntegratedActionModelEuler(crocoddyl.DifferentialActionModelFreeFwdDynamics(state, actuation, terminalCostModel), 0.)

runningModel.u_lb = -rmodel.effortLimit[-actuation.nu:]
runningModel.u_ub = rmodel.effortLimit[-actuation.nu:]

# Setting the nodes of the problem with a sliding variable
ratioContactTotal = 0.4/(conf.dt*T) # expressed as ratio in [s]
contactNodes = int(conf.T * ratioContactTotal)
flyingNodes = conf.T - contactNodes
problem_with_contact = crocoddyl.ShootingProblem(x0,
                                                [contactPhase] * contactNodes + [runningModel] * flyingNodes,
                                                terminalModel)
problem_without_contact = crocoddyl.ShootingProblem(x0, [runningModel] * T, terminalModel)

# SOLVE
ddp = crocoddyl.SolverFDDP(problem_with_contact)
ddp.setCallbacks([crocoddyl.CallbackLogger(), crocoddyl.CallbackVerbose(),])
# Additionally also modify ddp.th_stop and ddp.th_grad
ddp.th_stop = 1e-9
ddp.solve([],[], maxiter = int(1e3))
ddp.rmodel = rmodel

# SHOWING THE RESULTS
plotOCSolution(ddp)
plotConvergence(ddp)
plot_frame_trajectory(ddp, ['FL_HFE', 'FL_KFE', 'FL_FOOT'], trid = False)
animateMonoped(ddp, saveAnimation=False)
plot_power(ddp)

# CHECK THE CONTACT FORCE FRICTION CONE CONDITION

r_data = rmodel.createData()
contactFrameID = rmodel.getFrameId(footName)
Fx_, Fz_ = list([] for _ in range(2))
for i in range(int(conf.T*ratioContactTotal)):
        # convert the contact information to dictionary
        contactData = ddp.problem.runningDatas[i].differential.multibody.contacts.contacts['foot_contact']
        for force, vector in zip(contactData.f.linear, [Fx_, [], Fz_]):
                vector.append(force)
ratio = np.array(Fx_)/np.array(Fz_)
percentageContactViolation=len(ratio[ratio>cone.mu])/contactNodes*100
#assert((ratio<cone.mu)).all(), 'The friction cone condition is violated for {:0.1f}% of the contact phase ({:0.3f}s)'.format(percentageContactViolation, len(ratio[ratio>cone.mu])*conf.dt)
import matplotlib.pyplot as plt
Fz_clean=Fz_
Fz_clean.remove(max(Fz_))
plt.plot(Fz_clean)
plt.title('$F_z$')
plt.ylabel('[N]')
plt.show()

# POLISHING THE SOLUTION
# xs=ddp.xs
# us=ddp.us
# ddp2 = crocoddyl.SolverBoxFDDP(problem_with_contact)
# ddp2.setCallbacks([crocoddyl.CallbackLogger(), crocoddyl.CallbackVerbose(),])
# ddp2.th_stop = 1e-6
# ddp2.solve(xs, us, maxiter = int(2e2))
# ddp2.rmodel = rmodel

# plotOCSolution(ddp2)
# plotConvergence(ddp2)
# plot_power(ddp2)
# animateMonoped(ddp2)

# CHECKING THE PARTIAL DERIVATIVES
# runningModel.differential.costs.removeCost('joule_dissipation')
mnd = crocoddyl.DifferentialActionModelNumDiff(runningModel.differential)
dnd = mnd.createData()
m=runningModel.differential
d=m.createData()
x=ddp.xs[3];u=ddp.us[3]

cm=m.costs.costs['joule_dissipation'].cost
#cm.gamma=1
#cm.T_mu=1
#cm.n[:]=1
#cm.K[:,:]=1

# CHECK THE GRADIENT  AND THE HESSIAN
n_joints = conf.n_links
x=.5*np.ones(2 * (n_joints + 1))
u=np.ones(n_joints)
m.calc(d,x,u)
mnd.calc(dnd,x,u)
m.calcDiff(d,x,u)
mnd.calcDiff(dnd,x,u)
lu=d.Lu.copy()
lx=d.Lx.copy()
m.costs.costs['joule_dissipation'].cost

eps=1e-6

# Lx, Lu
print(d.Lx-dnd.Lx)
print(d.Lu-dnd.Lu)
# Lux
m.calc(d,x+np.concatenate((np.zeros(n_joints + 1), eps * np.ones(n_joints + 1))),u)
m.calcDiff(d,x+np.concatenate((np.zeros(n_joints + 1), eps * np.ones(n_joints + 1))),u)
if n_joints > 1:
    print(np.vstack((np.zeros((n_joints + 2, n_joints)),np.diag((d.Lu-lu)/eps)))-d.Lxu)
else:
    print(np.hstack((0,(d.Lu-lu)/eps))-d.Lxu)
# Lxx
m.calc(d, x + np.concatenate((np.zeros(n_joints + 1), eps * np.ones(n_joints + 1))), u)
m.calcDiff(d,x+np.concatenate((np.zeros(n_joints + 1), eps * np.ones(n_joints + 1))), u)
print(np.diag((d.Lx-lx)/eps)-d.Lxx)
# Luu
m.calc(d,x,u+eps*np.ones(n_joints))
m.calcDiff(d,x,u+eps*np.ones(n_joints))
print(np.diag((d.Lu-lu)/eps)-d.Luu)

def print_data(data):
    print('Lx', data.Lx)
    print('Lu', data.Lu)
    print('Lxx', data.Lxx)
    print('Lxu', data.Lxu)
    print('Luu', data.Luu)
    print('Fu', data.Fu)
    print('Fx', data.Fx)
    print('cost', data.cost)