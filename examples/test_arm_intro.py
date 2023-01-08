import crocoddyl
import pinocchio
import numpy as np
import example_robot_data
import aslr_to
import sys
import os
import time
WITHDISPLAY = 'display' in sys.argv or 'CROCODDYL_DISPLAY' in os.environ
WITHPLOT = 'plot' in sys.argv or 'CROCODDYL_PLOT' in os.environ

robot = example_robot_data.load("talos_arm")
rmodel = robot.model

state = crocoddyl.StateMultibody(rmodel)
DT = 1e-2
T = 10
actuation = crocoddyl.ActuationModelFull(state)
# Defining the problem
target = np.array([0., 0., 0.4])
# if WITHDISPLAY:
#     display = crocoddyl.GepettoDisplay(robot)
#     display.robot.viewer.gui.addSphere('world/point', .05, [1., 0., 0., 1.])  # radius = .1, RGBA=1001
#     display.robot.viewer.gui.applyConfiguration('world/point', target.tolist() + [0., 0., 0., 1.])  # xyz+quaternion
#     display.robot.viewer.gui.refresh()

# q0 = np.array([2., 1.5, -2., 0., 0., 0., 0.])

q0 = np.array([0.173046, 1., -0.52366, 0., 0., 0.1, -0.005])
x0 = np.concatenate([q0, np.zeros(state.nv)])


nu = actuation.nu 
runningCostModel = crocoddyl.CostModelSum(state, nu)
terminalCostModel = crocoddyl.CostModelSum(state, nu)

goalTrackingCost = crocoddyl.CostModelResidual(
    state,
    crocoddyl.ResidualModelFrameTranslation(state, rmodel.getFrameId("gripper_left_joint"), target,
                                            nu))
xRegCost = crocoddyl.CostModelResidual(state, crocoddyl.ResidualModelState(state, nu))
uWeights = np.array([0.] * actuation.nu)
uRegCost = crocoddyl.CostModelResidual(
    state,
    crocoddyl.ActivationModelWeightedQuad(uWeights**2),
    crocoddyl.ResidualModelControl(state, nu),
)

# Then let's added the running and terminal cost functions
#runningCostModel.addCost("gripperPose", goalTrackingCost, 1.)
runningCostModel.addCost("stateReg", xRegCost, 1e-4)
runningCostModel.addCost("ctrlReg", uRegCost, 1e-4)
#terminalCostModel.addCost("gripperPose", goalTrackingCost, 1e3)

runningConstraintModel = crocoddyl.ConstraintModelManager(state, actuation.nu)

terminalConstraintModel = crocoddyl.ConstraintModelManager(state, actuation.nu)

terminalConstraintModel.addConstraint(
    "posture",
    crocoddyl.ConstraintModelResidual(
        state,
        crocoddyl.ResidualModelFrameTranslation(state, rmodel.getFrameId("gripper_left_joint"),
                                                 target, actuation.nu),0.0001*np.ones(3),.001*np.ones(3)))
        # crocoddyl.ResidualModelControl(state, nu),0.1*np.ones(7),3*np.ones(7)))

# Create the action model
runningModel_1 = crocoddyl.IntegratedActionModelEuler(
    crocoddyl.DifferentialActionModelFreeFwdDynamics(state, actuation, runningCostModel), DT)
runningModel_2 = crocoddyl.IntegratedActionModelEuler(
    crocoddyl.DifferentialActionModelFreeFwdDynamics(state, actuation, runningCostModel,terminalConstraintModel), DT)
terminalModel = crocoddyl.IntegratedActionModelEuler(
    crocoddyl.DifferentialActionModelFreeFwdDynamics(state, actuation, terminalCostModel,
                                                    terminalConstraintModel))

problem = crocoddyl.ShootingProblem(x0, [runningModel_2] * int(T), terminalModel)

solver = aslr_to.SolverINTRO(problem)
solver.setCallbacks([crocoddyl.CallbackLogger(), crocoddyl.CallbackVerbose()])

solver.solve([],[],400)


if WITHDISPLAY:
    display = crocoddyl.GepettoDisplay(robot)
    display.robot.viewer.gui.addSphere('world/point', .05, [1., 0., 0., 1.])  # radius = .1, RGBA=1001
    display.robot.viewer.gui.applyConfiguration('world/point',
                                                target.tolist() + [0., 0., 0., 1.])  # xyz+quaternion
    display.robot.viewer.gui.refresh()

if WITHDISPLAY:
    while True:
        display.displayFromSolver(solver)
        time.sleep(2.0)
