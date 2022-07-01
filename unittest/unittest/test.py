import crocoddyl
import example_robot_data
import pinocchio
import aslr_to
import numpy as np
from test_utils_ex import NUMDIFF_MODIFIER, assertNumDiff

ROBOT_MODEL = example_robot_data.load("anymal").model
robot_data = ROBOT_MODEL.createData()
STATE = aslr_to.StateMultiASR(ROBOT_MODEL)
K = np.zeros([STATE.pinocchio.nv,STATE.pinocchio.nv])
K[-12:,-12:]=10*np.eye(12)
B = .001*np.eye(STATE.nv_m)
ACTUATION = aslr_to.ASRFreeFloatingActuation(STATE,K,B)

nq, nv = STATE.nq_l, STATE.nv_l
x = STATE.rand()

SUPPORT_FEET = [
    ROBOT_MODEL.getFrameId('LF_FOOT')
]
# SUPPORT_FEET = [
#     ROBOT_MODEL.getFrameId('LF_FOOT'),
#     ROBOT_MODEL.getFrameId('RF_FOOT'),
#     ROBOT_MODEL.getFrameId('LH_FOOT'),
#     ROBOT_MODEL.getFrameId('RH_FOOT')
# ]
nu = ACTUATION.nu
CONTACTS = crocoddyl.ContactModelMultiple(STATE, nu)
for i in SUPPORT_FEET:
    xref = crocoddyl.FrameTranslation(i, np.array([0., 0., 0.]))
    Contact = crocoddyl.ContactModel3D(STATE, xref, nu, np.array([0., 50.]))
    CONTACTS.addContact(ROBOT_MODEL.frames[i].name + "_contact", Contact)
COSTS = crocoddyl.CostModelSum(STATE, nu)

for i in SUPPORT_FEET:
    # cone = crocoddyl.FrictionCone(R, mu, 4, False)
    frictionCone = crocoddyl.CostModelResidual(
        STATE, crocoddyl.ResidualModelContactForce(STATE, i, pinocchio.Force.Zero(), 3,ACTUATION.nu))
    COSTS.addCost(ROBOT_MODEL.frames[i].name + "_frictionCone", frictionCone, 1)

# mu, R = 0.7, np.eye(3)
# for i in SUPPORT_FEET:
#     cone = crocoddyl.FrictionCone(R, mu, 4, False)
#     frictionCone = crocoddyl.CostModelResidual(
#         STATE, crocoddyl.ActivationModelQuadraticBarrier(crocoddyl.ActivationBounds(cone.lb, cone.ub)),
#         crocoddyl.ResidualModelContactFrictionCone(STATE, i, cone, nu))
#     COSTS.addCost(ROBOT_MODEL.frames[i].name + "_frictionCone", frictionCone, 1)

MODEL = aslr_to.DifferentialContactASLRFwdDynModel(STATE, ACTUATION, CONTACTS, COSTS)

#analytical da0_dx
u = np.random.rand(ACTUATION.nu)
DATA = MODEL.createData()
MODEL.calc( DATA,  x, u)
MODEL.calcDiff( DATA,  x, u)



MODEL_ND = crocoddyl.DifferentialActionModelNumDiff( MODEL)
# MODEL_ND.disturbance = 1000
DATA_ND = MODEL_ND.createData()

MODEL_ND.calc(DATA_ND,  x,  u)
MODEL_ND.calcDiff(DATA_ND,  x,  u)

assertNumDiff(DATA.Lx[:3], DATA_ND.Lx[:3], NUMDIFF_MODIFIER *
                MODEL_ND.disturbance)  # threshold was 2.7e-2, is now 2.11e-4 (see assertNumDiff.__doc__)

assertNumDiff(DATA.Lx[:18], DATA_ND.Lx[:18], NUMDIFF_MODIFIER *
                MODEL_ND.disturbance)  # threshold was 2.7e-2, is now 2.11e-4 (see assertNumDiff.__doc__)

assertNumDiff(DATA.Lx[36:], DATA_ND.Lx[36:], NUMDIFF_MODIFIER *
                MODEL_ND.disturbance)  # threshold was 2.7e-2, is now 2.11e-4 (see assertNumDiff.__doc__)
