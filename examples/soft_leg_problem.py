import crocoddyl
import pinocchio
import numpy as np
import aslr_to


class SimpleMonopedProblem:
    def __init__(self, rmodel, lfFoot, rfFoot, lhFoot, rhFoot,S=None):
        self.rmodel = rmodel
        self.rdata = rmodel.createData()
        self.state = crocoddyl.StateSoftMultibody(self.rmodel)
        self.K = np.zeros([self.state.pinocchio.nv,self.state.pinocchio.nq])
        nu = self.state.nv_m
        self.K[-nu:,-nu:]= 30*np.eye(nu)
        self.B = .01*np.eye(self.state.nv_m)
        self.actuation = aslr_to.ASRFreeFloatingActuation(self.state,self.K,self.B)
        if S is not None:
            self.S = S
        else:
            self.S = np.eye(12)
        # Getting the frame id for all the legs
        self.lfFootId = self.rmodel.getFrameId(lfFoot)
        self.rfFootId = self.rmodel.getFrameId(rfFoot)
        self.lhFootId = self.rmodel.getFrameId(lhFoot)
        self.rhFootId = self.rmodel.getFrameId(rhFoot)
        # Defining default state
        q0 = self.rmodel.referenceConfigurations["standing"]
        self.rmodel.defaultState = np.concatenate([q0, np.zeros(self.rmodel.nv), np.zeros(24)])
        self.firstStep = True
        # Defining the friction coefficient and normal
        self.mu = 0.7
        self.Rsurf = np.eye(3)
