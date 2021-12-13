import numpy as np
import pinocchio
import crocoddyl
import scipy.linalg as scl


#   state class for soft actuated robots. x = [q_l,v_l,q_m,v_m] . 
#   q_l : configuration vector of link side
#   q_m : configuration vector of motor side
#   v_l : q_dot of link side 
#   v_m : \theta_dot of the motor side

class StateMultiASR(crocoddyl.StateMultibody,crocoddyl.StateAbstract):
    def __init__(self, pinocchioModel):
        crocoddyl.StateMultibody.__init__(self, pinocchioModel)
        crocoddyl.StateAbstract.__init__(self, 3*(pinocchioModel.nv)-12 + pinocchioModel.nq , 4 * pinocchioModel.nv-12)
        self.nq_m = pinocchioModel.nv-6
        self.nv_m = pinocchioModel.nv -6
        self.nv_l = pinocchioModel.nv
        self.nq_l = pinocchioModel.nq
        self.ndx = 4 * pinocchioModel.nv-12
        self.nx = 3*(pinocchioModel.nv)-12 + pinocchioModel.nq
        self.nq = self.nq_m +self.nq_l
        self.nv = self.nv_m +self.nv_l
    def zero(self):
        q_l = pinocchio.neutral(self.pinocchio)
        v_l = pinocchio.utils.zero(self.pinocchio.nv)
        q_m = np.zeros(self.nq_m)
        v_m = np.zeros(self.nq_m)
        return np.concatenate([q_l,v_l,q_m,v_m])

    def rand(self):
        q_l = pinocchio.randomConfiguration(self.pinocchio)
        q_m = pinocchio.utils.rand(self.nq_m)
        v_l = pinocchio.utils.rand(self.pinocchio.nv)
        v_m = pinocchio.utils.rand(self.nq_m)
        return np.concatenate([q_l, v_l,q_m, v_m])

    def diff(self, x0, x1):
        nq_l = self.pinocchio.nq
        nv_l = self.pinocchio.nv
        nl = self.nq_l+ self.nv_l
        q0_l = x0[:nq_l]
        v0_l = x0[nq_l:nl]
        q0_m = x0[nl:-self.nv_m]
        v0_m = x0[-self.nv_m:]

        q1_l = x1[:nq_l]
        v1_l = x1[nq_l:nl]
        q1_m = x1[nl:-self.nv_m]
        v1_m = x1[-self.nv_m:]

        dq_l = pinocchio.difference(self.pinocchio, q0_l, q1_l)
        dq_m = q1_m - q0_m

        return np.concatenate([dq_l,  v1_l - v0_l, dq_m, v1_m - v0_m])

    def integrate(self, x, dx):
        nq_l = self.pinocchio.nq
        nv_l = self.pinocchio.nv
        nl = self.nq_l+ self.nv_l

        q_l = x[:nq_l]
        v_l = x[nq_l:nl]
        q_m = x[nl:-self.nv_m]
        v_m = x[-self.nv_m:]
        
        dq_l = dx[:nv_l]
        dv_l = dx[nv_l:2*nv_l]
        dq_m = dx[2*nv_l:-self.nv_m]
        dv_m = dx[-self.nv_m:]

        qn_l = pinocchio.integrate(self.pinocchio, q_l, dq_l)
        qn_m = q_m+ dq_m
        return np.concatenate([qn_l,  v_l + dv_l, qn_m, v_m + dv_m])

    def Jdiff(self, x1, x2, firstsecond=crocoddyl.Jcomponent.both):
        if firstsecond == crocoddyl.Jcomponent.both:
            return [self.Jdiff(x1, x2, crocoddyl.Jcomponent.first), self.Jdiff(x1, x2, crocoddyl.Jcomponent.second)]

        if firstsecond == crocoddyl.Jcomponent.first:
            nq_l = self.pinocchio.nq
            nv_l = self.pinocchio.nv
            dx = self.diff(x2, x1)

            q_l = x2[:nq_l]
            dq_l = dx[:nv_l]

            Jdq_l = pinocchio.dIntegrate(self.pinocchio, q_l, dq_l)[0]
            return np.matrix(-scl.block_diag(np.linalg.inv(Jdq_l), np.eye(self.nv_l), np.eye(self.nv_m), np.eye(self.nv_m)))
        elif firstsecond == crocoddyl.Jcomponent.second:
            nq_l = self.pinocchio.nq
            nv_l = self.pinocchio.nv
            dx = self.diff(x1, x2)

            q_l = x1[:nq_l]
            dq_l = dx[:nv_l]

            Jdq_l = pinocchio.dIntegrate(self.pinocchio, q_l, dq_l)[1]
            return np.matrix(scl.block_diag(np.linalg.inv(Jdq_l), np.eye(self.nv_l), np.eye(self.nv_m), np.eye(self.nv_m)))

    def Jintegrate(self, x, dx, firstsecond=crocoddyl.Jcomponent.both):
        if firstsecond == crocoddyl.Jcomponent.both:
            return [
                self.Jintegrate(x, dx, crocoddyl.Jcomponent.first),
                self.Jintegrate(x, dx, crocoddyl.Jcomponent.second)
            ]
        nq_l = self.pinocchio.nq
        nv_l = self.pinocchio.nv
        
        q_l = x[:nq_l]
        dq_l = dx[:nv_l]

        Jq_l, Jdq_l = pinocchio.dIntegrate(self.pinocchio, q_l, dq_l)
        if firstsecond == crocoddyl.Jcomponent.first:
            return np.matrix(scl.block_diag(Jq_l, np.eye(self.nv_l), np.eye(self.nv_m),np.eye(self.nv_m)))
        elif firstsecond == crocoddyl.Jcomponent.second:
            return np.matrix(scl.block_diag(Jdq_l, np.eye(self.nv_l), np.eye(self.nv_m), np.eye(self.nv_m)))
        
    def JintegrateTransport(self, x, dx,  Jin,  firstsecond=crocoddyl.Jcomponent.first):

        nv_l = self.pinocchio.nv
        if firstsecond == crocoddyl.Jcomponent.first:
            Jin = pinocchio.dIntegrateTransport(self.pinocchio, x[:self.pinocchio.nq], dx[:nv_l], Jin[:nv_l,:], pinocchio.ArgumentPosition.ARG0)

        elif firstsecond == crocoddyl.Jcomponent.second:
            Jin = pinocchio.dIntegrateTransport(self.pinocchio, x[:self.pinocchio.nq], dx[:nv_l], Jin[:nv_l,:], pinocchio.ArgumentPosition.ARG1)
        
        return Jin
        