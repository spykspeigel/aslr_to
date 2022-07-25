import crocoddyl
import example_robot_data
import pinocchio
import aslr_to
import numpy as np
from test_utils_ex import NUMDIFF_MODIFIER, assertNumDiff

anymal = example_robot_data.load("softleg")
ROBOT_MODEL = anymal.model
robot_data = ROBOT_MODEL.createData()
STATE = crocoddyl.StateSoftMultibody(ROBOT_MODEL)
K = np.zeros([STATE.pinocchio.nv,STATE.pinocchio.nv])
nu = STATE.nv_m
K[-nu:,-nu:]= 30*np.eye(nu)
B = .01*np.eye(STATE.nv_m)
ACTUATION = aslr_to.SoftLegActuation(STATE)

nq, nv = STATE.nq_l, STATE.nv_l
x = np.array([0.1,0.14,0.14,0,0,0,0,0,0,0])
pinocchio.forwardKinematics(ROBOT_MODEL, robot_data, x[:nq], x[nq:nq+nv],
                            pinocchio.utils.zero(nv))
pinocchio.computeJointJacobians(ROBOT_MODEL, robot_data)
pinocchio.updateFramePlacements(ROBOT_MODEL, robot_data)
pinocchio.computeForwardKinematicsDerivatives(ROBOT_MODEL, robot_data, x[:nq], x[nq:nq+nv],
                                                pinocchio.utils.zero(nv))

SUPPORT_FEET = [
    ROBOT_MODEL.getFrameId('softleg_1_contact_link')]

nu = ACTUATION.nu
CONTACTS = crocoddyl.ContactModelMultiple(STATE, nu)
for i in SUPPORT_FEET:
    xref = crocoddyl.FrameTranslation(i, np.array([0., 0., 0.]))
    Contact = crocoddyl.ContactModel3D(STATE, xref, nu, np.array([0., 50.]))
    CONTACTS.addContact(ROBOT_MODEL.frames[i].name + "_contact", Contact)
COSTS = crocoddyl.CostModelSum(STATE, nu)
# for i in SUPPORT_FEET:
#     # cone = crocoddyl.FrictionCone(R, mu, 4, False)
#     frictionCone = crocoddyl.CostModelResidual(
#         STATE, crocoddyl.ResidualModelContactForce(STATE, i, pinocchio.Force.Zero(), 3,ACTUATION.nu+12))
#     COSTS.addCost(ROBOT_MODEL.frames[i].name + "_frictionCone", frictionCone, 1)

mu, R = 0.7, np.eye(3)
for i in SUPPORT_FEET:
    cone = crocoddyl.FrictionCone(R, mu, 4, False)
    frictionCone = crocoddyl.CostModelResidual(
        STATE, crocoddyl.ActivationModelQuadraticBarrier(crocoddyl.ActivationBounds(cone.lb, cone.ub)),
        crocoddyl.ResidualModelContactFrictionCone(STATE, i, cone, nu))
    COSTS.addCost(ROBOT_MODEL.frames[i].name + "_frictionCone", frictionCone, 1)

MODEL = aslr_to.DifferentialContactASLRFwdDynModel(STATE, ACTUATION, CONTACTS, COSTS,K,B)

#analytical da0_dx
u = np.random.rand(nu)
DATA = MODEL.createData()
MODEL.calc( DATA,  x, u)
MODEL.calcDiff( DATA,  x, u)
df_dx = DATA.df_dx
c = DATA.cost
Lx = DATA.Lx
res = DATA.costs.costs.todict()['softleg_1_contact_link_frictionCone'].residual.r
Rx = DATA.costs.costs.todict()['softleg_1_contact_link_frictionCone'].residual.Rx
Jc = DATA.multibody.contacts.contacts.todict()['softleg_1_contact_link_contact'].Jc
#computing numerical diff da0_dx
disturbance =1e-8
dx = np.zeros(STATE.ndx)
df_dx_nd = np.zeros([3,STATE.ndx])
dres_dx_nd = np.zeros([3,STATE.ndx])
Lx_nd = np.zeros([STATE.ndx,])
f = DATA.multibody.contacts.contacts.todict()['softleg_1_contact_link_contact'].f.linear
temp_lx = np.zeros(STATE.ndx)

for i in range(STATE.ndx):
    dx[i] = disturbance
    x_p = STATE.integrate(x,dx)
    dx=np.zeros(STATE.ndx)
    # robot_data = ROBOT_MODEL.createData()
    pinocchio.forwardKinematics(ROBOT_MODEL, robot_data, x_p[:nq], x_p[nq:nq+nv],
                            pinocchio.utils.zero(nv))
    pinocchio.updateFramePlacements(ROBOT_MODEL, robot_data)
    pinocchio.computeJointJacobians(ROBOT_MODEL, robot_data)
    pinocchio.computeAllTerms(ROBOT_MODEL, robot_data, x_p[:nq], x_p[nq:nq+nv])
    pinocchio.computeCentroidalMomentum(ROBOT_MODEL, robot_data, x_p[:nq], x_p[nq:nq+nv])
    pinocchio.computeForwardKinematicsDerivatives(ROBOT_MODEL, robot_data, x_p[:nq], x_p[nq:nq+nv],
                                                    pinocchio.utils.zero(nv))
    DATA_ND =MODEL.createData() 
    MODEL.calc( DATA_ND,  x_p, u)
    f_new = DATA_ND.multibody.contacts.contacts.todict()['softleg_1_contact_link_contact'].f.linear
    res_new = DATA_ND.costs.costs.todict()['softleg_1_contact_link_frictionCone'].residual.r
    Jc_new = DATA_ND.multibody.contacts.contacts.todict()['softleg_1_contact_link_contact'].Jc
    print("new Jc is being printed")
    # print((f_new-f)/disturbance)
    print(np.linalg.norm(Jc-Jc_new)/disturbance)
    df_dx_nd[:,i]=(f_new-f)/disturbance
    # dres_dx_nd[:,i] = (res_new-res)/disturbance
    Lx_nd[i] = (DATA_ND.cost-DATA.cost)/disturbance
    # Lx_nd[i] = (0.5*np.dot(res_new,res_new) - 0.5*np.dot(res,res))/disturbance
    # temp_lx[i] = np.dot(dres_dx_nd[:,i].T, res_new) 

# # print(0.5*np.dot(res,res))
# # print(DATA.cost)
# # print(dres_dx_nd)
# # print(np.dot(dres_dx_nd.T,res))
# # print(Lx_nd)
# # print(Lx)
# print(np.linalg.norm(Lx-np.dot(dres_dx_nd.T,res)))
print(np.linalg.norm(Lx-Lx_nd))
# # print(temp_lx)
print(np.linalg.norm(df_dx_nd[:,:3]-df_dx[:,:3]))
# # print(np.linalg.norm(temp_lx-Lx))
# # print(np.linalg.norm(dres_dx_nd-Rx))

# MODEL_ND = crocoddyl.DifferentialActionModelNumDiff( MODEL)
# # MODEL_ND.disturbance = 1000
# DATA_ND = MODEL_ND.createData()

# MODEL_ND.calc(DATA_ND,  x,  u)
# MODEL_ND.calcDiff(DATA_ND,  x,  u)

# assertNumDiff(DATA.Lx, DATA_ND.Lx, NUMDIFF_MODIFIER *
#                 MODEL_ND.disturbance)  # threshold was 2.7e-2, is now 2.11e-4 (see assertNumDiff.__doc__)

# assertNumDiff(DATA.Lx[:18], DATA_ND.Lx[:18], NUMDIFF_MODIFIER *
#                 MODEL_ND.disturbance)  # threshold was 2.7e-2, is now 2.11e-4 (see assertNumDiff.__doc__)

# assertNumDiff(DATA.Lx[36:], DATA_ND.Lx[36:], NUMDIFF_MODIFIER *
#                 MODEL_ND.disturbance)  # threshold was 2.7e-2, is now 2.11e-4 (see assertNumDiff.__doc__)
