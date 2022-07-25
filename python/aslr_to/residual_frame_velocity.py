import numpy as np
import pinocchio
import crocoddyl

class ResidualFrameVelocityASR(crocoddyl.ResidualModelAbstract):
    def __init__(self, state, activation=None, frame_id=None, velocity=None, nu=None):
        crocoddyl.ResidualModelAbstract.__init__(self, state, 6, nu, True, True, False)
        self._frame_id = frame_id
        self._velocity = velocity

    def calc(self, data, x, u):
        data.residual.r[:] = (
            pinocchio.getFrameVelocity(self.state.pinocchio, data.shared.pinocchio, self._frame_id, pinocchio.LOCAL) -
            self._velocity).vector

    def calcDiff(self, data, x, u):
        v_partial_dq, v_partial_dv = pinocchio.getJointVelocityDerivatives(self.state.pinocchio, data.shared.pinocchio,
                                                                           data.joint, pinocchio.ReferenceFrame.LOCAL)

        self.activation.calcDiff(data.activation, data.residual.r)
        data.residual.Rx[:,:int(self.state.nq/2)] = np.hstack([np.dot(data.fXj, v_partial_dq), np.dot(data.fXj, v_partial_dv)])

    def createData(self, collector):
        data = FrameVelocityCostDataDerived(self, collector)
        return data


class FrameVelocityCostDataDerived(crocoddyl.CostDataAbstract):

    def __init__(self, model, collector):
        crocoddyl.CostDataAbstract.__init__(self, model, collector)
        self.fXj = model.state.pinocchio.frames[model._frame_id].placement.inverse().action
        self.joint = model.state.pinocchio.frames[model._frame_id].parent
