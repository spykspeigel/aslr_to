import sys
from telnetlib import DM
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

ACTUATION = aslr_to.ASRFreeFloatingActuation(STATE,K,B)

SUPPORT_FEET = [
    ROBOT_MODEL.getFrameId('LF_FOOT'),
    ROBOT_MODEL.getFrameId('RF_FOOT'),
    ROBOT_MODEL.getFrameId('LH_FOOT'),
    ROBOT_MODEL.getFrameId('RH_FOOT')
]

nu = ACTUATION.nu

CONTACTS = crocoddyl.ContactModelMultiple(STATE, nu)

SUPPORT_FEET = [
    ROBOT_MODEL.getFrameId('LF_FOOT')]

for i in SUPPORT_FEET:
    xref = crocoddyl.FrameTranslation(i, np.array([0., 0., 0.]))
    supportContactModel = crocoddyl.ContactModel3D(STATE, xref, nu, np.array([0., 50.]))
    CONTACTS.addContact(ROBOT_MODEL.frames[i].name + "_contact", supportContactModel)
COSTS = crocoddyl.CostModelSum(STATE, nu)

dMODEL = aslr_to.DifferentialContactASLRFwdDynModel(STATE, ACTUATION, CONTACTS, COSTS,K,B)

MODEL = crocoddyl.IntegratedActionModelEuler(dMODEL, 1e-2)

x = MODEL.state.rand()
u = np.random.rand(MODEL.nu)
DATA = MODEL.createData()

MODEL_ND = crocoddyl.ActionModelNumDiff( MODEL)
# MODEL_ND.disturbance = 1000
DATA_ND = MODEL_ND.createData()
MODEL.calc( DATA,  x,  u)
MODEL.calcDiff( DATA,  x,  u)
MODEL_ND.calc(DATA_ND,  x,  u)
MODEL_ND.calcDiff(DATA_ND,  x,  u)

assertNumDiff( DATA.Fu, DATA_ND.Fu, NUMDIFF_MODIFIER *
                MODEL_ND.disturbance)  # threshold was 2.7e-2, is now 2.11e-4 (see assertNumDiff.__doc__)
assertNumDiff( DATA.Fx[36:48,36:48], DATA_ND.Fx[36:48,36:48], NUMDIFF_MODIFIER *
                MODEL_ND.disturbance)  # threshold was 2.7e-2, is now 2.11e-4 (see assertNumDiff.__doc__)
