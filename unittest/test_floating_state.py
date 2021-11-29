import crocoddyl
import example_robot_data
import pinocchio
import aslr_to
import numpy as np
import aslr_to
from test_utils_ex import NUMDIFF_MODIFIER, assertNumDiff

robot_model = example_robot_data.load("anymal").model
state = aslr_to.StateMultiASR(robot_model)

x1 = state.rand()
x2 = state.rand()
dx_test = pinocchio.utils.rand(state.ndx_)
dx = state.diff(x1,x2)
print(dx.shape)
print(x1.shape)
xi = state.integrate(x1,dx)

xi_test = state.integrate(x1,dx_test)


Jfirst = state.Jdiff(x1, x2, crocoddyl.Jcomponent.first)
Jsecond = state.Jdiff(x1, x2, crocoddyl.Jcomponent.second)
Jfirst_test = state.Jdiff(x1, x2)[0]
Jsecond_test = state.Jdiff(x1, x2)[1]

assertNumDiff( xi, x2, NUMDIFF_MODIFIER *
                1e-4)  # threshold was 2.7e-2, is now 2.11e-4 (see assertNumDiff.__doc__)

assertNumDiff( state.diff(xi_test,x1), dx_test, NUMDIFF_MODIFIER *
                1e-4)  # threshold was 2.7e-2, is now 2.11e-4 (see assertNumDiff.__doc__)

assertNumDiff( Jfirst, Jfirst_test, NUMDIFF_MODIFIER *
                1e-4)  # threshold was 2.7e-2, is now 2.11e-4 (see assertNumDiff.__doc__)

assertNumDiff( Jsecond, Jsecond_test, NUMDIFF_MODIFIER *
                1e-4)  # threshold was 2.7e-2, is now 2.11e-4 (see assertNumDiff.__doc__)