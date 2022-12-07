import numpy as np
import pinocchio
import crocoddyl

class RigidLegActuation(crocoddyl.ActuationModelAbstract):

    def __init__(self, state):
        crocoddyl.ActuationModelAbstract.__init__(self, state, 2)

    def calc(self, data, x, u):
        data.tau = np.hstack([np.zeros(1), u[:]])

    def calcDiff(self, data, x, u):
        data.dtau_du[1:, :] = np.eye(2)
