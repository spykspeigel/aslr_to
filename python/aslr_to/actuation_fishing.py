import numpy as np
import pinocchio
import crocoddyl

class ASRFishing(crocoddyl.ActuationModelAbstract):
    def __init__(self, state):
        crocoddyl.ActuationModelAbstract.__init__(self, state, 1)

    def calc(self, data, x, u):
        data.tau[0] = u

    def calcDiff(self, data, x, u):
        data.dtau_du[0] =  1

