import numpy as np
import pinocchio
import crocoddyl


#Currently only supports Fully actuated case
class ResidualModelFramePlacementASR(crocoddyl.ResidualModelAbstract):
    def __init__(self, state, frame_id=None, placement=None, nu=None):
        crocoddyl.ResidualModelAbstract.__init__(self, state, 6, nu,True, False, False)
        self._frame_id = frame_id
        self._placement = placement

    def calc(self, data, x, u):
        data.rMf = self._placement.inverse() * data.shared.pinocchio.oMf[self._frame_id]
        data.r[:] = pinocchio.log(data.rMf).vector

    def calcDiff(self, data, x, u):
        pinocchio.updateFramePlacements(self.state.pinocchio, data.shared.pinocchio)

        data.rJf[:, :] = pinocchio.Jlog6(data.rMf)
        data.fJf[:, :] = pinocchio.getFrameJacobian(self.state.pinocchio, data.shared.pinocchio, self._frame_id,
                                                    pinocchio.ReferenceFrame.LOCAL)
        data.J[:, :] = np.dot(data.rJf, data.fJf)
        data.Rx[:,:int(self.state.nq/2)] = data.J

    def createData(self, collector):
        data = ResidualDataFramePlacementASR(self, collector)
        return data


class ResidualDataFramePlacementASR(crocoddyl.ResidualDataAbstract):
    def __init__(self, model, collector):
        crocoddyl.ResidualDataAbstract.__init__(self, model, collector)
        self.rMf = pinocchio.SE3.Identity()
        self.rJf = pinocchio.Jlog6(self.rMf)
        self.fJf = np.zeros([6, int(model.state.nv/2)])
        self.rJf = np.zeros((6, 6))
        self.J = np.zeros((6, int(model.state.nv/2)))
