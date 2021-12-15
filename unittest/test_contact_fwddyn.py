import sys
import unittest

import crocoddyl
import example_robot_data
import pinocchio
import numpy as np
import aslr_to
from test_utils_ex import NUMDIFF_MODIFIER, assertNumDiff

ROBOT_MODEL = example_robot_data.load("anymal").model
STATE = aslr_to.StateMultiASR(ROBOT_MODEL)
ACTUATION = aslr_to.ASRFreeFloatingActuation(STATE)
SUPPORT_FEET = [
    ROBOT_MODEL.getFrameId('LF_FOOT'),
    ROBOT_MODEL.getFrameId('RF_FOOT'),
    ROBOT_MODEL.getFrameId('LH_FOOT'),
    ROBOT_MODEL.getFrameId('RH_FOOT')
]
nu = ACTUATION.nu
CONTACTS = crocoddyl.ContactModelMultiple(STATE, nu)
for i in SUPPORT_FEET:
    xref = crocoddyl.FrameTranslation(i, np.array([0., 0., 0.]))
    supportContactModel = crocoddyl.ContactModel3D(STATE, xref, nu, np.array([0., 50.]))
    CONTACTS.addContact(ROBOT_MODEL.frames[i].name + "_contact", supportContactModel)
COSTS = crocoddyl.CostModelSum(STATE, nu)

mu, R = 0.7, np.eye(3)

for i in SUPPORT_FEET:
    cone = crocoddyl.FrictionCone(R, mu, 4, False)
    frictionCone = crocoddyl.CostModelResidual(
        STATE, crocoddyl.ActivationModelQuadraticBarrier(crocoddyl.ActivationBounds(cone.lb, cone.ub)),
        crocoddyl.ResidualModelContactFrictionCone(STATE, i, cone, ACTUATION.nu, CONTACTS.nc))
    COSTS.addCost(ROBOT_MODEL.frames[i].name + "_frictionCone", frictionCone, 1e1)

q0 = ROBOT_MODEL.referenceConfigurations["standing"]
defaultState=np.concatenate([q0, np.zeros(ROBOT_MODEL.nv), np.zeros(24)])
stateWeights = np.array([0.] * 3 + [500.] * 3 + [0.01] * (ROBOT_MODEL.nv - 6) + [10.] * 6 + [1.] *
                        (ROBOT_MODEL.nv - 6) + [0.]*2*STATE.nv_m)
stateResidual = crocoddyl.ResidualModelState(STATE, defaultState, nu)
stateActivation = crocoddyl.ActivationModelWeightedQuad(stateWeights**2)
ctrlWeights = np.array( [1e0] * nu )
ctrlResidual = crocoddyl.ResidualModelControl(STATE, nu)
ctrlActivation = crocoddyl.ActivationModelWeightedQuad(ctrlWeights**2)
stateReg = crocoddyl.CostModelResidual(STATE, stateActivation, stateResidual)
ctrlReg = crocoddyl.CostModelResidual(STATE, ctrlActivation, ctrlResidual)
COSTS.addCost("stateReg", stateReg, 1e1)
COSTS.addCost("ctrlReg", ctrlReg, 1e-1)

MODEL = aslr_to.DifferentialContactASLRFwdDynModel(STATE, ACTUATION, CONTACTS, COSTS)

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

assertNumDiff( DATA.Fu, DATA_ND.Fu, NUMDIFF_MODIFIER *
                MODEL_ND.disturbance)  # threshold was 2.7e-2, is now 2.11e-4 (see assertNumDiff.__doc__)
assertNumDiff( DATA.Fx, DATA_ND.Fx, NUMDIFF_MODIFIER *
                MODEL_ND.disturbance)  # threshold was 2.7e-2, is now 2.11e-4 (see assertNumDiff.__doc__)

assertNumDiff(DATA.Lu, DATA_ND.Lu, NUMDIFF_MODIFIER *
                MODEL_ND.disturbance)  # threshold was 2.7e-2, is now 2.11e-4 (see assertNumDiff.__doc__)
assertNumDiff(DATA.Lx[:], DATA_ND.Lx[:], NUMDIFF_MODIFIER *
                MODEL_ND.disturbance)  # threshold was 2.7e-2, is now 2.11e-4 (see assertNumDiff.__doc__)
