import numpy as np
import pinocchio
import crocoddyl
import aslr_to

class DataCollectorSoftActMultibody(crocoddyl.DataCollectorActMultibody, aslr_to.DataCollectorSoftModel):
    def __init__(self, pinocchio, actuation, motor):
        crocoddyl.DataCollectorActMultibody.__init__(self,pinocchio,actuation) 
        aslr_to.DataCollectorSoftModel.__init__(self,motor)

class DataCollectorSoftActMultibodyInContact(crocoddyl.DataCollectorActMultibodyInContact, aslr_to.DataCollectorSoftModel):
    def __init__(self, pinocchio, actuation, motor, contacts):
        crocoddyl.DataCollectorActMultibodyInContact.__init__(pinocchio,actuation,contacts)
        aslr_to.DataCollectorSoftModel.__init__(motor)