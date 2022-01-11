import numpy as np
import pinocchio
import crocoddyl
import scipy.linalg as scl


#   state class for soft actuated robots. x = [q_l,q_m,v_l,v_m] . 
#   q_l : configuration vector of link side
#   q_m : configuration vector of motor side
#   v_l : q_dot of link side 
#   v_m : \theta_dot of the motor side

class StateMultibodyASR(crocoddyl.StateAbstract):
    def __init__(self, pinocchioModel):
        crocoddyl.StateAbstract.__init__(self, 2*(pinocchioModel.nq + pinocchioModel.nv), 4 * pinocchioModel.nv)
        self.pinocchio = pinocchioModel

    def zero(self):
        q_l = pinocchio.neutral(self.pinocchio)
        v_l = pinocchio.utils.zero(int(self.nv/2))
        q_m = pinocchio.neutral(self.pinocchio)
        v_m = pinocchio.utils.zero(int(self.nv/2))
        return np.concatenate([q_l,q_m, v_l, v_m])

    def rand(self):
        q_l = pinocchio.randomConfiguration(self.pinocchio)
        q_m = pinocchio.randomConfiguration(self.pinocchio)
        v_l = pinocchio.utils.rand(int(self.nv/2))
        v_m = pinocchio.utils.rand(int(self.nv/2))
        return np.concatenate([q_l, q_m, v_l, v_m])

    def diff(self, x0, x1):
        nq=self.nq
        nv=self.nv
        q0_l = x0[:int(nq/2)]
        q0_m = x0[int(nq/2):nq]
        v0_l = x0[-nv:-int(nv/2)]
        v0_m = x0[-int(nv/2):]
        q1_l = x1[:int(nq/2)]
        q1_m = x1[int(nq/2):nq]
        v1_l = x1[-nv:-int(nv/2)]
        v1_m = x1[-int(nv/2):]
        dq_l = pinocchio.difference(self.pinocchio, q0_l, q1_l)
        dq_m = q1_m - q0_m
        return np.concatenate([dq_l, dq_m, v1_l - v0_l, v1_m - v0_m])

    def integrate(self, x, dx):
        nq=self.nq
        nv=self.nv
        q_l = x[:int(nq/2)]
        q_m = x[int(nq/2):nq]
        v_l = x[nv:-int(nv/2)]
        v_m = x[-int(nv/2):]
        dq_l = dx[:int(nq/2)]
        dq_m = dx[int(nq/2):nq]
        dv_l = dx[-nv:-int(nv/2)]
        dv_m = dx[-int(nv/2):]
        qn_l = pinocchio.integrate(self.pinocchio, q_l, dq_l)
        qn_m =  q_m+ dq_m

        return np.concatenate([qn_l, qn_m, v_l + dv_l, v_m + dv_m])

    def Jdiff(self, x1, x2, firstsecond=crocoddyl.Jcomponent.both):
        if firstsecond == crocoddyl.Jcomponent.both:
            return [self.Jdiff(x1, x2, crocoddyl.Jcomponent.first), self.Jdiff(x1, x2, crocoddyl.Jcomponent.second)]

        if firstsecond == crocoddyl.Jcomponent.first:
            nq=self.nq
            nv=self.nv
            dx = self.diff(x2, x1)

            q_l = x2[:int(nq/2)]
            q_m = x2[int(nq/2):nq]
            dq_l = dx[:int(nq/2)]
            dq_m = dx[int(nq/2):nq]

            Jdq_l = pinocchio.dIntegrate(self.pinocchio, q_l, dq_l)[1]
            # Jdq_m = pinocchio.dIntegrate(self.pinocchio, q_m, dq_m)[1]
            return np.matrix(-scl.block_diag(np.linalg.inv(Jdq_l), np.eye(nv+int(nq/2))))
        elif firstsecond == crocoddyl.Jcomponent.second:
            nq=self.nq
            nv=self.nv
            dx = self.diff(x1, x2)
            q_l = x1[:int(nq/2)]
            q_m = x1[int(nq/2):nq]
            dq_l = dx[:int(nq/2)]
            dq_m = dx[int(nq/2):nq]
            Jdq_l = pinocchio.dIntegrate(self.pinocchio, q_l, dq_l)[1]
            # Jdq_m = pinocchio.dIntegrate(self.pinocchio, q_m, dq_m)[1]
            return np.matrix(scl.block_diag(np.linalg.inv(Jdq_l), np.eye(self.nv+int(nq/2))))

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
        # Jq_m, Jdq_m = pinocchio.dIntegrate(self.pinocchio, q_m, dq_m)
        if firstsecond == crocoddyl.Jcomponent.first:
            return np.matrix(scl.block_diag(Jq_l,np.eye(nv+int(nq/2))))
        elif firstsecond == crocoddyl.Jcomponent.second:
            return np.matrix(scl.block_diag(Jdq_l, np.eye(nv+int(nq/2))))
