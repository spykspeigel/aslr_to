import numpy as np
import pinocchio
import crocoddyl


class IntegratedActionModelEulerASR(crocoddyl.ActionModelAbstract):
    def __init__(self, diffModel, timeStep=1e-3, withCostResiduals=True):
        crocoddyl.ActionModelAbstract.__init__(self, diffModel.state, int(diffModel.state.nv/2), diffModel.nr)
        self.differential = diffModel
        self.withCostResiduals = withCostResiduals
        self.timeStep = timeStep

    def calc(self, data, x, u=None):
        nq, dt = self.state.nq, self.timeStep
        self.differential.calc(data.differential, x, u)
        acc=data.differential.xout
        if self.withCostResiduals:
            data.r = data.differential.r
        data.cost = data.differential.cost
        # data.xnext[nq:] = x[nq:] + acc*dt
        # data.xnext[:nq] = pinocchio.integrate(self.differential.pinocchio,
        #                                       a2m(x[:nq]),a2m(data.xnext[nq:]*dt)).flat
        data.dx = np.concatenate([x[nq:] * dt + acc * dt**2, acc * dt])
        data.xnext[:] = self.differential.state.integrate(x, data.dx)

        return data.xnext, data.cost

    def calcDiff(self, data, x, u=None):
        nv, dt = self.state.nv, self.timeStep
        self.differential.calcDiff(data.differential, x, u)
        dxnext_dx, dxnext_ddx = self.state.Jintegrate(x, data.dx)
        da_dx, da_du = data.differential.Fx, data.differential.Fu
        ddx_dx = np.vstack([da_dx * dt, da_dx])
        ddx_dx[range(nv), range(nv, 2 * nv)] += 1
        data.Fx[:, :] = dxnext_dx + dt * np.dot(dxnext_ddx, ddx_dx)
        ddx_du = np.vstack([da_du * dt, da_du])
        data.Fu[:, :] = dt * np.dot(dxnext_ddx, ddx_du)
        data.Lx[:] = data.differential.Lx
        data.Lu[:] = data.differential.Lu
        data.Lxx[:, :] = data.differential.Lxx
        data.Lxu[:, :] = data.differential.Lxu
        data.Luu[:, :] = data.differential.Luu

    def createData(self):
        data = IntegratedActionDataEulerASR(self)
        return data


class IntegratedActionDataEulerASR(crocoddyl.ActionDataAbstract):
    def __init__(self, model):
        crocoddyl.ActionDataAbstract.__init__(self, model)
        self.differential = model.differential.createData()
