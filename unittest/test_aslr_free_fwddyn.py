import sys
import unittest

import crocoddyl
import pinocchio
import example_robot_data
import numpy as np
import inv_dyn
from python import aslr_to
from test_utils_ex import NUMDIFF_MODIFIER, assertNumDiff


class ASLRFreeFwdDynamicsTestCase(unittest.TestCase):
    MODEL = None

    def setUp(self):
        self.x = self.MODEL.state.rand()
        self.u = np.random.rand(self.MODEL.nu)
        self.DATA = self.MODEL.createData()

    def test_calcDiff_against_numdiff(self):
        MODEL_ND = crocoddyl.DifferentialActionModelNumDiff(self.MODEL)
        MODEL_ND.disturbance *= 10
        DATA_ND = MODEL_ND.createData()
        self.MODEL.calc(self.DATA, self.x, self.u)
        self.MODEL.calcDiff(self.DATA, self.x, self.u)
        MODEL_ND.calc(DATA_ND, self.x, self.u)
        MODEL_ND.calcDiff(DATA_ND, self.x, self.u)
        assertNumDiff(self.DATA.Fx, DATA_ND.Fx, NUMDIFF_MODIFIER *
                      MODEL_ND.disturbance)  # threshold was 2.7e-2, is now 2.11e-4 (see assertNumDiff.__doc__)
        assertNumDiff(self.DATA.Fu, DATA_ND.Fu, NUMDIFF_MODIFIER *
                      MODEL_ND.disturbance)  # threshold was 2.7e-2, is now 2.11e-4 (see assertNumDiff.__doc__)
        assertNumDiff(self.DATA.Lx, DATA_ND.Lx, NUMDIFF_MODIFIER *
                      MODEL_ND.disturbance)  # threshold was 2.7e-2, is now 2.11e-4 (see assertNumDiff.__doc__)
        assertNumDiff(self.DATA.Lu, DATA_ND.Lu, NUMDIFF_MODIFIER *
                      MODEL_ND.disturbance)  # threshold was 2.7e-2, is now 2.11e-4 (see assertNumDiff.__doc__)
        assertNumDiff(self.DATA.Gx, DATA_ND.Gx, NUMDIFF_MODIFIER *
                      MODEL_ND.disturbance)  # threshold was 2.7e-2, is now 2.11e-4 (see assertNumDiff.__doc__)
        assertNumDiff(self.DATA.Gu, DATA_ND.Gu, NUMDIFF_MODIFIER *
                      MODEL_ND.disturbance)  # threshold was 2.7e-2, is now 2.11e-4 (see assertNumDiff.__doc__)
        assertNumDiff(self.DATA.Hx, DATA_ND.Hx, NUMDIFF_MODIFIER *
                      MODEL_ND.disturbance)  # threshold was 2.7e-2, is now 2.11e-4 (see assertNumDiff.__doc__)
        assertNumDiff(self.DATA.Hu, DATA_ND.Hu, NUMDIFF_MODIFIER *
                      MODEL_ND.disturbance)  # threshold was 2.7e-2, is now 2.11e-4 (see assertNumDiff.__doc__)


class ASLRFreeDynamicsTest(ASLRFreeFwdDynamicsTestCase):
    ROBOT_MODEL = example_robot_data.load('talos_arm').model
    STATE = aslr_to.StateMultibodyASLR(ROBOT_MODEL)
    ACTUATION = aslr_to.ASLRActuation(STATE)
    nu = ACTUATION.nu 
    COSTS = crocoddyl.CostModelSum(STATE, nu)
    MODEL = aslr_to.DifferentialFreeASLRFwdDynamicsModel(STATE, ACTUATION, COSTS)


if __name__ == '__main__':
    test_classes_to_run = [ASLRFreeDynamicsTest]
    loader = unittest.TestLoader()
    suites_list = []
    for test_class in test_classes_to_run:
        suite = loader.loadTestsFromTestCase(test_class)
        suites_list.append(suite)
    big_suite = unittest.TestSuite(suites_list)
    runner = unittest.TextTestRunner()
    results = runner.run(big_suite)
    sys.exit(not results.wasSuccessful())
