import sys
import unittest
from random import randint
import numpy as np

import crocoddyl
import aslr_to

RUNNING_MODEL = crocoddyl.IntegratedActionModelEuler(aslr_to.ConstrainedDifferentialLQRModel(4, 4,4,4), 5e-3)
TERMINAL_MODEL = crocoddyl.IntegratedActionModelEuler(aslr_to.ConstrainedDifferentialLQRModel(4, 4,4,4))
SOLVER = aslr_to.SolverINTRO

T = randint(1, 50)
# T = 50
state = TERMINAL_MODEL.state
xs = []
us = []
xs.append(state.rand())
for _ in range(T):
    xs.append(state.rand())
    us.append(np.random.rand(TERMINAL_MODEL.nu))
PROBLEM = crocoddyl.ShootingProblem(xs[0], [RUNNING_MODEL] * T, TERMINAL_MODEL)
PROBLEM.nthreads = 1
solver = SOLVER(PROBLEM)

solver.setCallbacks([crocoddyl.CallbackVerbose()])
solver.th_stop = 1e-10
solver.solve(xs, us, 10)

print(solver.iter)
print(solver.stepLength)
