import crocoddyl
import pinocchio
import numpy as np


class Contact3DModelASLR(crocoddyl.ContactModelAbstract):
    def __init__(self, state, xref, nu, gains=[0., 0.]):
        crocoddyl.ContactModelAbstract.__init__(self, state, 3, nu)
        self.xref = xref
        self.gains = gains
        self.joint = state.pinocchio.frames[xref.id].parent

    def calc(self, data, x):
        assert (self.xref.translation is not None or self.gains[0] == 0.)
        v = pinocchio.getFrameVelocity(self.state.pinocchio, data.pinocchio, self.xref.id)
        data.vw[:] = v.angular
        data.vv[:] = v.linear

        fJf = pinocchio.getFrameJacobian(self.state.pinocchio, data.pinocchio, self.xref.id,
                                         pinocchio.ReferenceFrame.LOCAL)
        data.Jc = fJf[:3, :]
        data.Jw[:, :self.state.nv_l] = fJf[3:, :]

        data.a0[:] = pinocchio.getFrameAcceleration(self.state.pinocchio, data.pinocchio,
                                                    self.xref.id).linear + np.cross(data.vw, data.vv)
        if self.gains[0] != 0.:
            data.a0[:] += np.asscalar(
                self.gains[0]) * (data.pinocchio.oMf[self.xref.id].translation - self.xref.translation)
        if self.gains[1] != 0.:
            data.a0[:] += np.asscalar(self.gains[1]) * data.vv

    def calcDiff(self, data, x):
        v_partial_dq, a_partial_dq, a_partial_dv, a_partial_da = pinocchio.getJointAccelerationDerivatives(
            self.state.pinocchio, data.pinocchio, self.joint, pinocchio.ReferenceFrame.LOCAL)

        data.vv_skew = pinocchio.skew(data.vv)
        data.vw_skew = pinocchio.skew(data.vw)
        fXjdv_dq = np.dot(data.fXj, v_partial_dq)
        da0_dq = np.dot(data.fXj, a_partial_dq)[:3, :]
        da0_dq += np.dot(data.vw_skew, fXjdv_dq[:3, :])
        da0_dq -= np.dot(data.vv_skew, fXjdv_dq[3:, :])
        da0_dv = np.dot(data.fXj, a_partial_dv)[:3, :]
        da0_dv += np.dot(data.vw_skew, data.Jc)

        da0_dv -= np.dot(data.vv_skew, data.Jw[:, :self.state.nv_l])

        if np.asscalar(self.gains[0]) != 0.:
            R = data.pinocchio.oMf[self.xref.id].rotation
            da0_dq += np.asscalar(self.gains[0]) * np.dot(
                R,
                pinocchio.getFrameJacobian(self.state.pinocchio, data.pinocchio, self.xref.id,
                                           pinocchio.ReferenceFrame.LOCAL)[:3, :])
        if np.asscalar(self.gains[1]) != 0.:
            da0_dq += np.asscalar(self.gains[1]) * np.dot(data.fXj[:3, :], v_partial_dq)
            da0_dv += np.asscalar(self.gains[1]) * np.dot(data.fXj[:3, :], a_partial_da)
        data.da0_dx[:, :self.state.nv_l] = da0_dq
        data.da0_dx[:, self.state.nv_m+self.state.nv_l:-self.state.nv_m] = da0_dv

    def updateForce(self, data, force):
    #       if (force.size() != 3) {
    # throw_pretty("Invalid argument: "
    #              << "lambda has wrong dimension (it should be 3)");
        data.f = data.jMf.act(pinocchio.Force(np.append(force, np.zeros(3))))

    
    def createData(self, data):
        data = Contact3DDataASLR(self, data)
        return data


class Contact3DDataASLR(crocoddyl.ContactDataAbstract):
    def __init__(self, model, data):
        crocoddyl.ContactDataAbstract.__init__(self, model, data)
        self.fXj = model.state.pinocchio.frames[model.xref.id].placement.inverse().action
        self.vw = np.zeros(3)
        self.vv = np.zeros(3)
        self.Jw = np.zeros((3, model.state.nv))
        self.vv_skew = np.zeros((3, 3))
        self.vw_skew = np.zeros((3, 3))
