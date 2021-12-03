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
    supportContactModel = aslr_to.Contact3DModelASLR(STATE, xref, nu, np.array([0., 50.]))
    CONTACTS.addContact(ROBOT_MODEL.frames[i].name + "_contact", supportContactModel)
COSTS = crocoddyl.CostModelSum(STATE, nu)
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