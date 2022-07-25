###################################3
####
####Deprecated code
####
#####################################


import numpy as np
import pinocchio
import crocoddyl
import scipy.linalg as scl


#   state class for soft actuated robots. x = [q_l,q_m,v_l,v_m] . 
#   q_l : configuration vector of link side
#   q_m : configuration vector of motor side
#   v_l : q_dot of link side 
#   v_m : \theta_dot of the motor side

class StateMultibodyASRFixed(crocoddyl.StateAbstract):
    def __init__(self, pinocchioModel):
        crocoddyl.StateAbstract.__init__(self, 2*(pinocchioModel.nq + pinocchioModel.nv), 4 * pinocchioModel.nv)
        self.pinocchio = pinocchioModel
        self.nq_l = pinocchioModel.nq
        self.nv_l = pinocchioModel.nv
        self.nq_m = pinocchioModel.nv
        self.nv_m = pinocchioModel.nv
    def zero(self):
        nq_l = self.nq_l
        nv_l = self.nv_l
        nq_m = self.nq_m
        nv_m = self.nv_m
        q_l = pinocchio.neutral(self.pinocchio)
        v_l = pinocchio.utils.zero(nv_l)
        q_m = pinocchio.neutral(self.pinocchio)
        v_m = pinocchio.utils.zero(nv_m)
        return np.concatenate([q_l,q_m, v_l, v_m])

    def rand(self):
        nq_l = self.nq_l
        nv_l = self.nv_l
        nq_m = self.nq_m
        nv_m = self.nv_m
        q_l = pinocchio.randomConfiguration(self.pinocchio)
        q_m = pinocchio.randomConfiguration(self.pinocchio)
        v_l = pinocchio.utils.rand(nv_l)
        v_m = pinocchio.utils.rand(nv_m)
        return np.concatenate([q_l, v_l, q_m, v_m])

    def diff(self, x0, x1):
        nq_l = self.nq_l
        nv_l = self.nv_l
        nq_m = self.nq_m
        nv_m = self.nv_m
        nl = nq_l+nv_l 
        q0_l = x0[:nq_l]
        v0_l = x0[nq_l:nl]
        q0_m = x0[nl:-nv_m]
        v0_m = x0[-nv_m:]
        q1_l = x1[:nq_l]
        v1_l = x1[nq_l:nl]
        q1_m = x1[nl:-nv_m]
        v1_m = x1[-nv_m:]
        dq_l = pinocchio.difference(self.pinocchio, q0_l, q1_l)
        dq_m = pinocchio.difference(self.pinocchio, q0_m, q1_m)
        return np.concatenate([dq_l, v1_l - v0_l, dq_m, v1_m - v0_m])

    def integrate(self, x, dx):
        nq_l = self.nq_l
        nv_l = self.nv_l
        nq_m = self.nq_m
        nv_m = self.nv_m
        nl = nq_l+nv_l 
        q_l = x[:nq_l]
        v_l = x[nq_l:nl]
        q_m = x[nl:-nv_m]
        v_m = x[-nv_m:]
        dq_l = dx[:nq_l]
        dv_l = dx[nq_l:nl]
        dq_m = dx[nl:-nv_m]
        dv_m = dx[-nv_m:]

        qn_l = pinocchio.integrate(self.pinocchio, q_l, dq_l)
        qn_m = pinocchio.integrate(self.pinocchio, q_m, dq_m)

        return np.concatenate([qn_l, v_l + dv_l, qn_m, v_m + dv_m])

    def Jdiff(self, x1, x2, firstsecond=crocoddyl.Jcomponent.both):
        if firstsecond == crocoddyl.Jcomponent.both:
            return [self.Jdiff(x1, x2, crocoddyl.Jcomponent.first), self.Jdiff(x1, x2, crocoddyl.Jcomponent.second)]

        if firstsecond == crocoddyl.Jcomponent.first:
            nq_l = self.nq_l
            nv_l = self.nv_l
            nq_m = self.nq_m
            nv_m = self.nv_m
            nl = nq_l+nv_l 
            dx = self.diff(x2, x1)

            q_l = x2[:nq_l]
            q_m = x2[nl:-nv_m]
            dq_l = dx[:nq_l]
            dq_m = dx[nl:-nv_m]

            Jdq_l = pinocchio.dIntegrate(self.pinocchio, q_l, dq_l)[1]
            Jdq_m = pinocchio.dIntegrate(self.pinocchio, q_m, dq_m)[1]
            return np.matrix(-scl.block_diag(np.linalg.inv(Jdq_l), np.eye(nv_l), np.linalg.inv(Jdq_m), np.eye(nv_m)))
        elif firstsecond == crocoddyl.Jcomponent.second:
            nq_l = self.nq_l
            nv_l = self.nv_l
            nq_m = self.nq_m
            nv_m = self.nv_m
            nl = nq_l+nv_l 
            dx = self.diff(x1, x2)

            q_l = x1[:nq_l]
            q_m = x1[nl:-nv_m]
            dq_l = dx[:nq_l]
            dq_m = dx[nl:-nv_m]

            Jdq_l = pinocchio.dIntegrate(self.pinocchio, q_l, dq_l)[1]
            Jdq_m = pinocchio.dIntegrate(self.pinocchio, q_m, dq_m)[1]
            return np.matrix(scl.block_diag(np.linalg.inv(Jdq_l),np.eye(self.nv_l),np.linalg.inv(Jdq_m), np.eye(self.nv_m)))

    def Jintegrate(self, x, dx, firstsecond=crocoddyl.Jcomponent.both):
        if firstsecond == crocoddyl.Jcomponent.both:
            return [
                self.Jintegrate(x, dx, crocoddyl.Jcomponent.first),
                self.Jintegrate(x, dx, crocoddyl.Jcomponent.second)
            ]
        nq=self.nq
        nv=self.nv
        q_l = x[:int(nq/2)]
        q_m = x[int(nq/2):nq]
        dq_l = dx[:int(nq/2)]
        dq_m = dx[int(nq/2):nq]
        Jq_l, Jdq_l = pinocchio.dIntegrate(self.pinocchio, q_l, dq_l)
        Jq_m, Jdq_m = pinocchio.dIntegrate(self.pinocchio, q_m, dq_m)
        if firstsecond == crocoddyl.Jcomponent.first:
            return np.matrix(scl.block_diag(np.linalg.inv(Jq_l), np.eye(self.nv_l), np.linalg.inv(Jq_m), np.eye(self.nv_m)))
        elif firstsecond == crocoddyl.Jcomponent.second:
            return np.matrix(scl.block_diag(np.linalg.inv(Jdq_l), np.eye(self.nv_l), np.linalg.inv(Jdq_m), np.eye(self.nv_m)))
