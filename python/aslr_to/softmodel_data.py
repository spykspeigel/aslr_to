import numpy as np
import pinocchio
import crocoddyl
import aslr_to

class SoftModelData:
    def __init__(self, nv_m):
        self.theta = np.zeros(nv_m)
        self.theta_dot = np.zeros(nv_m)
        self.theta_ddot = np.zeros(nv_m)
        self.K = np.zeros([nv_m,nv_m])
        self.nv_m = nv_m
class DataCollectorSoftModel(crocoddyl.DataCollectorAbstract):
    def __init__(self, motor):
        crocoddyl.DataCollectorAbstract.__init__(self)
        # SoftModelData.__init__(self,motor.nv_m)
        self.motor = motor

