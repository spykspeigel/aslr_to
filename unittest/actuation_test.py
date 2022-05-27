from lzma import MODE_FAST
import crocoddyl
import example_robot_data
import pinocchio
import aslr_to
import numpy as np

two_dof = example_robot_data.load('asr_twodof')
ROBOT_MODEL = two_dof.model
robot_data = ROBOT_MODEL.createData()
STATE = aslr_to.StateMultibodyASR(ROBOT_MODEL)
ACTUATION = aslr_to.QbActuationModel(STATE)
x = STATE.rand()

nq=STATE.nq
nv=STATE.nv
nq_l = int(nq/2)
nv_l = int(nv/2)
q_l = x[:nq_l]
# q_m = x[nq_l:nq]
v_l = x[-nv:-nv_l]
# v_m = x[-nv_l:]
x_l = np.hstack([q_l,v_l])

pinocchio.forwardKinematics(ROBOT_MODEL, robot_data, x_l[:nq_l], v_l,
                            pinocchio.utils.zero(nv_l))
pinocchio.computeJointJacobians(ROBOT_MODEL, robot_data)
pinocchio.updateFramePlacements(ROBOT_MODEL, robot_data)
pinocchio.computeForwardKinematicsDerivatives(ROBOT_MODEL, robot_data, x_l[:nq_l], v_l,
                                                pinocchio.utils.zero(nv_l))

nu = ACTUATION.nu

#analytical dA_dx
u = np.random.rand(ACTUATION.nu)
DATA = ACTUATION.createData()
ACTUATION.calc( DATA,  x, u)
ACTUATION.calcDiff( DATA,  x, u)
dK_dx = DATA.dK_dx
dtau_dx = DATA.dtau_dx
dK_du = DATA.dK_du
dtau_du = DATA.dtau_du
#computing numerical diff dA_dx
disturbance =2e-8
dx = np.zeros(STATE.ndx)
dA_dx_nd = np.zeros([4,STATE.ndx])
K = DATA.K
tau = DATA.tau
dK_dx_nd = np.zeros([2,8])
dtau_dx_nd = np.zeros([4,8])
for i in range(STATE.ndx):
    dx[i] = disturbance
    x_p = STATE.integrate(x,dx)
    dx=np.zeros(STATE.ndx)
    DATA_ND =ACTUATION.createData() 
    ACTUATION.calc( DATA_ND,  x_p, u)
    dK_dx_nd[:,i]=(-K + DATA_ND.K)/disturbance
    dtau_dx_nd[:,i]=(-tau + DATA_ND.tau)/disturbance

dK_du_nd = np.zeros([2,nu])
dtau_du_nd = np.zeros([4,nu])
du = np.zeros(nu)
for i in range(nu):
    du[i] = disturbance
    DATA_ND =ACTUATION.createData() 
    ACTUATION.calc( DATA_ND,  x, u+du)
    du = np.zeros(nu)
    dK_du_nd[:,i]=(-K + DATA_ND.K)/disturbance
    dtau_du_nd[:,i]=(-tau + DATA_ND.tau)/disturbance

# print(dtau_dx)
# print(dtau_dx_nd)
print(np.linalg.norm(dtau_dx_nd-dtau_dx))
# print(dK_dx)
# print(dK_dx_nd)
print(np.linalg.norm(dtau_du-dtau_du_nd))

from test_utils_ex import NUMDIFF_MODIFIER, assertNumDiff
# assertNumDiff(dK_dx,dK_dx_nd, NUMDIFF_MODIFIER *
#                 2e-8)  # threshold was 2.7e-2, is now 2.11e-4 (see assertNumDiff.__doc__)

assertNumDiff(dtau_du,dtau_du_nd, NUMDIFF_MODIFIER *
                2e-8)  # threshold was 2.7e-2, is now 2.11e-4 (see assertNumDiff.__doc__)

assertNumDiff(dtau_dx,dtau_dx_nd, NUMDIFF_MODIFIER *
                2e-8)  # threshold was 2.7e-2, is now 2.11e-4 (see assertNumDiff.__doc__)
