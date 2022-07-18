import crocoddyl
import example_robot_data
import pinocchio
import aslr_to
import numpy as np
import aslr_to
from test_utils_ex import NUMDIFF_MODIFIER, assertNumDiff

robot_model = example_robot_data.load("anymal").model
state = crocoddyl.StateSoftMultibody(robot_model)
state_nd = crocoddyl.StateNumDiff(state)
x1 = state.rand()
x2 = state.rand()
dx_test = pinocchio.utils.rand(state.ndx)
dx = state.diff(x1,x2)
xi = state.integrate(x1,dx)

xi_test = state.integrate(x1,dx_test)


Jfirst = state.Jdiff(x1, x2, crocoddyl.Jcomponent.first)
Jsecond = state.Jdiff(x1, x2, crocoddyl.Jcomponent.second)
Jfirst_nd = state_nd.Jdiff(x1, x2, crocoddyl.Jcomponent.first)
Jsecond_nd = state_nd.Jdiff(x1, x2, crocoddyl.Jcomponent.second)

Jintfirst = state.Jintegrate(x1, dx, crocoddyl.Jcomponent.first)
Jintsecond = state.Jintegrate(x1, dx, crocoddyl.Jcomponent.second)
Jintfirst_nd = state_nd.Jintegrate(x1, dx, crocoddyl.Jcomponent.first)
Jintsecond_nd = state_nd.Jintegrate(x1, dx, crocoddyl.Jcomponent.second)

assertNumDiff( xi, x2, NUMDIFF_MODIFIER *
                1e-4)  # threshold was 2.7e-2, is now 2.11e-4 (see assertNumDiff.__doc__)

assertNumDiff( state.diff(x1,xi_test), dx_test, NUMDIFF_MODIFIER *
                1e-4)  # threshold was 2.7e-2, is now 2.11e-4 (see assertNumDiff.__doc__)

assertNumDiff( np.array(Jfirst.tolist()[0]), np.array(Jfirst_nd.tolist()[0]), NUMDIFF_MODIFIER *
                1e-10)  # threshold was 2.7e-2, is now 2.11e-4 (see assertNumDiff.__doc__)

assertNumDiff( np.array(Jsecond.tolist()[0])[36:48,36:48], np.array(Jsecond_nd.tolist()[0])[36:48,36:48], NUMDIFF_MODIFIER *
                1e-10)  # threshold was 2.7e-2, is now 2.11e-4 (see assertNumDiff.__doc__)

assertNumDiff( Jintfirst, np.array(Jintfirst_nd.tolist()[0]), NUMDIFF_MODIFIER *
                1e-10)  # threshold was 2.7e-2, is now 2.11e-4 (see assertNumDiff.__doc__)

assertNumDiff( Jintsecond, np.array(Jintsecond_nd.tolist()[0]), NUMDIFF_MODIFIER *
                1e-10)  # threshold was 2.7e-2, is now 2.11e-4 (see assertNumDiff.__doc__)
