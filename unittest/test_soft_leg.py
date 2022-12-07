import sys
import unittest

import crocoddyl
import example_robot_data
import pinocchio
import numpy as np
import aslr_to
from test_utils_ex import NUMDIFF_MODIFIER, assertNumDiff

ROBOT_MODEL = example_robot_data.load("softleg").model
STATE = crocoddyl.StateSoftMultibody(ROBOT_MODEL)
K = np.zeros([STATE.pinocchio.nv,STATE.pinocchio.nv])
nu = STATE.nv_m
K[-nu:,-nu:]= 30*np.eye(nu)
B = .01*np.eye(STATE.nv_m)
ACTUATION = aslr_to.SoftLegActuation(STATE)

nu = ACTUATION.nu

CONTACTS = crocoddyl.ContactModelMultiple(STATE, nu)

SUPPORT_FEET = [
    ROBOT_MODEL.getFrameId('softleg_1_contact_link')]

for i in SUPPORT_FEET:
    xref = crocoddyl.FrameTranslation(i, np.array([0., 0., 0.]))
    supportContactModel = crocoddyl.ContactModel2D(STATE, xref, nu, np.array([0., 50.]))
    CONTACTS.addContact(ROBOT_MODEL.frames[i].name + "_contact", supportContactModel)
COSTS = crocoddyl.CostModelSum(STATE, nu)

# mu, R = 0.7, np.eye(3)
# for i in SUPPORT_FEET:
#     frictionCone = crocoddyl.CostModelResidual(
#         STATE, crocoddyl.ResidualModelContactForce(STATE, i, pinocchio.Force.Zero(), 3,ACTUATION.nu))
#     COSTS.addCost(ROBOT_MODEL.frames[i].name + "_frictionCone", frictionCone, 1)

# for i in SUPPORT_FEET:
#     cone = crocoddyl.FrictionCone(R, mu, 4, False)
#     frictionCone = crocoddyl.CostModelResidual(
#         STATE, crocoddyl.ActivationModelQuadraticBarrier(crocoddyl.ActivationBounds(cone.lb, cone.ub)),
#         crocoddyl.ResidualModelContactFrictionCone(STATE, i, cone, nu))
#     COSTS.addCost(ROBOT_MODEL.frames[i].name + "_frictionCone", frictionCone, 1e1)

# for i, p in zip(swingFootIds, feetPos0):
#     # Defining a foot swing task given the step length
#     # resKnot = numKnots % 2
#     phKnots = numKnots / 2
#     if k < phKnots:
#         dp = np.array([stepLength * (k + 1) / numKnots, 0., stepHeight * k / phKnots])
#     elif k == phKnots:
#         dp = np.array([stepLength * (k + 1) / numKnots, 0., stepHeight])
#     else:
#         dp = np.array(
#             [stepLength * (k + 1) / numKnots, 0., stepHeight * (1 - float(k - phKnots) / phKnots)])
#     tref = p + dp

#     swingFootTask += [[i, pinocchio.SE3(np.eye(3), tref)]]

# for i in swingFootTask:
#     frameTranslationResidual = crocoddyl.ResidualModelFrameTranslation(STATE, i[0], i[1].translation,
#                                                                         nu)
#     frameVelocityResidual = crocoddyl.ResidualModelFrameVelocity(STATE, i[0], pinocchio.Motion.Zero(),
#                                                                     pinocchio.LOCAL, nu)
#     footTrack = crocoddyl.CostModelResidual(STATE, frameTranslationResidual)
#     impulseFootVelCost = crocoddyl.CostModelResidual(STATE, frameVelocityResidual)
#     COSTS.addCost(ROBOT_MODEL.frames[i[0]].name + "_footTrack", footTrack, 1e7)
#     COSTS.addCost(ROBOT_MODEL.frames[i[0]].name + "_impulseVel", impulseFootVelCost, 1e6)

# rdata = ROBOT_MODEL.createData()
# q0 = ROBOT_MODEL.referenceConfigurations["standing"]
# pinocchio.forwardKinematics(ROBOT_MODEL,rdata , q0)
# pinocchio.updateFramePlacements(ROBOT_MODEL, rdata)
# com0 = pinocchio.centerOfMass(ROBOT_MODEL, rdata, q0)

# comResidual = crocoddyl.ResidualModelCoMPosition(STATE, com0, nu)
# comTrack = crocoddyl.CostModelResidual(STATE, comResidual)
# COSTS.addCost("comTrack", comTrack, 1e5)

dMODEL = aslr_to.DifferentialContactASLRFwdDynModel(STATE, ACTUATION, CONTACTS, COSTS,K,B)

MODEL = crocoddyl.IntegratedActionModelEuler(dMODEL, 1e-2)

x = MODEL.state.rand()
u = np.random.rand(MODEL.nu)
DATA = MODEL.createData()

MODEL_ND = crocoddyl.ActionModelNumDiff( MODEL)
MODEL_ND.disturbance *= 10
DATA_ND = MODEL_ND.createData()
MODEL.calc( DATA,  x,  u)
MODEL.calcDiff( DATA,  x,  u)
MODEL_ND.calc(DATA_ND,  x,  u)
MODEL_ND.calcDiff(DATA_ND,  x,  u)

assertNumDiff( DATA.Fu, DATA_ND.Fu, NUMDIFF_MODIFIER *
                MODEL_ND.disturbance)  # threshold was 2.7e-2, is now 2.11e-4 (see assertNumDiff.__doc__)
assertNumDiff( DATA.Fx, DATA_ND.Fx, NUMDIFF_MODIFIER *
                MODEL_ND.disturbance)  # threshold was 2.7e-2, is now 2.11e-4 (see assertNumDiff.__doc__)

assertNumDiff(DATA.Lu, DATA_ND.Lu, NUMDIFF_MODIFIER *
                MODEL_ND.disturbance)  # threshold was 2.7e-2, is now 2.11e-4 (see assertNumDiff.__doc__)
assertNumDiff(DATA.Lx, DATA_ND.Lx, NUMDIFF_MODIFIER *
                MODEL_ND.disturbance)  # threshold was 2.7e-2, is now 2.11e-4 (see assertNumDiff.__doc__)

# assertNumDiff(DATA.Lx[:18], DATA_ND.Lx[:18], NUMDIFF_MODIFIER *
#                 MODEL_ND.disturbance)  # threshold was 2.7e-2, is now 2.11e-4 (see assertNumDiff.__doc__)

# assertNumDiff(DATA.Lx[36:], DATA_ND.Lx[36:], NUMDIFF_MODIFIER *
#                 MODEL_ND.disturbance)  # threshold was 2.7e-2, is now 2.11e-4 (see assertNumDiff.__doc__)
