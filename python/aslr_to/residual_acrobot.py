import numpy as np
import crocoddyl
import pinocchio

class ResidualModelDoublePendulum(crocoddyl.ResidualModelAbstract):
    def __init__(self, state, nu):
        crocoddyl.ResidualModelAbstract.__init__(self, state, 6, nu, True, True, False)

    def calc(self, data, x, u):
        c1, c2 = np.cos(x[0]), np.cos(x[1])
        s1, s2 = np.sin(x[0]), np.sin(x[1])
        data.r[:] = np.array([s1, s2, 1 + c1, 1 - c2, x[4], x[5]])

    def calcDiff(self, data, x, u):
        c1, c2 = np.cos(x[0]), np.cos(x[1])
        s1, s2 = np.sin(x[0]), np.sin(x[1])

        data.Rx[:2, :2] = np.diag([c1, c2])
        data.Rx[2:4, :2] = np.diag([-s1, s2])
        data.Rx[4:6, 4:6] = np.diag([1, 1])

    def createData(self, collector):
        data = ResidualDataDoublePendulum(self, collector)
        return data


class ResidualDataDoublePendulum(crocoddyl.ResidualDataAbstract):
    def __init__(self, model, collector):
        crocoddyl.ResidualDataAbstract.__init__(self, model, collector)
