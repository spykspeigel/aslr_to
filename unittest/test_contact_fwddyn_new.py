import sys
import unittest

import crocoddyl
import example_robot_data
import pinocchio
import numpy as np
import aslr_to
from test_utils_ex import NUMDIFF_MODIFIER, assertNumDiff

ROBOT_MODEL = example_robot_data.load("anymal").model
STATE = crocoddyl.StateSoftMultibody(ROBOT_MODEL)
K = np.zeros([STATE.pinocchio.nv,STATE.pinocchio.nv])
K[-12:,-12:]=1*np.eye(12)
B = .01*np.eye(STATE.nv_m)
nu =12
ACTUATION = aslr_to.FreeFloatingActuationNew(STATE,nu)

SUPPORT_FEET = [
    ROBOT_MODEL.getFrameId('LF_FOOT'),
    ROBOT_MODEL.getFrameId('RF_FOOT'),
    ROBOT_MODEL.getFrameId('LH_FOOT'),
    ROBOT_MODEL.getFrameId('RH_FOOT')
]

nu = ACTUATION.nu + STATE.nv_m

CONTACTS = crocoddyl.ContactModelMultiple(STATE, nu)

SUPPORT_FEET = [
    ROBOT_MODEL.getFrameId('LF_FOOT')]

for i in SUPPORT_FEET:
    xref = crocoddyl.FrameTranslation(i, np.array([0., 0., 0.]))
    supportContactModel = crocoddyl.ContactModel3D(STATE, xref, nu, np.array([0., 50.]))
    CONTACTS.addContact(ROBOT_MODEL.frames[i].name + "_contact", supportContactModel)
COSTS = crocoddyl.CostModelSum(STATE, nu)

mu, R = 0.7, np.eye(3)
for i in SUPPORT_FEET:
    frictionCone = crocoddyl.CostModelResidual(
        STATE, crocoddyl.ResidualModelContactForce(STATE, i, pinocchio.Force.Zero(), 3,nu))
    # COSTS.addCost(ROBOT_MODEL.frames[i].name + "_frictionCone", frictionCone, 1)

for i in SUPPORT_FEET:
    cone = crocoddyl.FrictionCone(R, mu, 4, False)
    frictionCone = crocoddyl.CostModelResidual(
        STATE, crocoddyl.ActivationModelQuadraticBarrier(crocoddyl.ActivationBounds(cone.lb, cone.ub)),
        crocoddyl.ResidualModelContactFrictionCone(STATE, i, cone, nu))
    COSTS.addCost(ROBOT_MODEL.frames[i].name + "_frictionCone", frictionCone, 1e1)

CONSTRAINTS = crocoddyl.ConstraintModelManager(STATE, nu)

CONSTRAINTS.addConstraint(
    "rnea",
    crocoddyl.ConstraintModelResidual(STATE, aslr_to.ResidualModelTauCouple(STATE, nu, K)))

MODEL = aslr_to.DifferentialContactASLRFwdDynModelNew(STATE, ACTUATION, CONTACTS, COSTS,K,B)

x = MODEL.state.rand()
u = np.random.rand(MODEL.nu)
DATA = MODEL.createData()

MODEL_ND = crocoddyl.DifferentialActionModelNumDiff( MODEL)
MODEL_ND.disturbance *= 10
DATA_ND = MODEL_ND.createData()
MODEL.calc( DATA,  x,  u)
MODEL.calcDiff( DATA,  x,  u)
MODEL_ND.calc(DATA_ND,  x,  u)
MODEL_ND.calcDiff(DATA_ND,  x,  u)

assertNumDiff(DATA.Lx, DATA_ND.Lx, NUMDIFF_MODIFIER *
                MODEL_ND.disturbance)  # threshold was 2.7e-2, is now 2.11e-4 (see assertNumDiff.__doc__)
assertNumDiff( DATA.Fx, DATA_ND.Fx, NUMDIFF_MODIFIER *
                MODEL_ND.disturbance)  # threshold was 2.7e-2, is now 2.11e-4 (see assertNumDiff.__doc__)

assertNumDiff( DATA.Hx, DATA_ND.Hx, NUMDIFF_MODIFIER *
                MODEL_ND.disturbance)  # threshold was 2.7e-2, is now 2.11e-4 (see assertNumDiff.__doc__)

assertNumDiff( DATA.Fu, DATA_ND.Fu, NUMDIFF_MODIFIER *
                MODEL_ND.disturbance)  # threshold was 2.7e-2, is now 2.11e-4 (see assertNumDiff.__doc__)
assertNumDiff(DATA.Lu, DATA_ND.Lu, NUMDIFF_MODIFIER *
                MODEL_ND.disturbance)  # threshold was 2.7e-2, is now 2.11e-4 (see assertNumDiff.__doc__)

# assertNumDiff(DATA.Lx[:18], DATA_ND.Lx[:18], NUMDIFF_MODIFIER *
#                 MODEL_ND.disturbance)  # threshold was 2.7e-2, is now 2.11e-4 (see assertNumDiff.__doc__)

# assertNumDiff(DATA.Lx[36:], DATA_ND.Lx[36:], NUMDIFF_MODIFIER *
#                 MODEL_ND.disturbance)  # threshold was 2.7e-2, is now 2.11e-4 (see assertNumDiff.__doc__)
