import crocoddyl
import pinocchio
import numpy as np
import aslr_to


class SimpleQuadrupedalGaitProblem:
    def __init__(self, rmodel, lfFoot, rfFoot, lhFoot, rhFoot):
        self.rmodel = rmodel
        self.rdata = rmodel.createData()
        self.state = crocoddyl.StateMultibody(self.rmodel)
        self.K = np.zeros([self.state.pinocchio.nv,self.state.pinocchio.nq])
        nu = self.state.nv_m
        self.K[-nu:,-nu:]= 10*np.eye(nu)
        self.B = .01*np.eye(self.state.nv_m)
        self.actuation = aslr_to.ASRFreeFloatingActuation(self.state,self.K,self.B)
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


    def createCoMProblem(self, x0, comGoTo, timeStep, numKnots):
        """ Create a shooting problem for a CoM forward/backward task.

        :param x0: initial state
        :param comGoTo: initial CoM motion
        :param timeStep: step time for each knot
        :param numKnots: number of knots per each phase
        :return shooting problem
        """
        # Compute the current foot positions
        q0 = x0[:self.rmodel.nq]
        pinocchio.forwardKinematics(self.rmodel, self.rdata, q0)
        pinocchio.updateFramePlacements(self.rmodel, self.rdata)
        com0 = pinocchio.centerOfMass(self.rmodel, self.rdata, q0)

        # Defining the action models along the time instances
        comModels = []

        # Creating the action model for the CoM task
        comForwardModels = [
            self.createSwingFootModel(timeStep, [self.lfFootId, self.rfFootId, self.lhFootId, self.rhFootId])
            for k in range(numKnots)
        ]
        comForwardTermModel = self.createSwingFootModel(timeStep,
                                                        [self.lfFootId, self.rfFootId, self.lhFootId, self.rhFootId],
                                                        com0 + np.array([comGoTo, 0., 0.]))
        comForwardTermModel.differential.costs.costs['comTrack'].weight = 1e6

        comBackwardModels = [
            self.createSwingFootModel(timeStep, [self.lfFootId, self.rfFootId, self.lhFootId, self.rhFootId])
            for k in range(numKnots)
        ]
        comBackwardTermModel = self.createSwingFootModel(timeStep,
                                                         [self.lfFootId, self.rfFootId, self.lhFootId, self.rhFootId],
                                                         com0 + np.array([-comGoTo, 0., 0.]))
        comBackwardTermModel.differential.costs.costs['comTrack'].weight = 1e6

        # Adding the CoM tasks
        comModels += comForwardModels + [comForwardTermModel]
        comModels += comBackwardModels + [comBackwardTermModel]

        # Defining the shooting problem
        problem = crocoddyl.ShootingProblem(x0, comModels, comModels[-1])
        return problem

    def createCoMGoalProblem(self, x0, comGoTo, timeStep, numKnots):
        """ Create a shooting problem for a CoM position goal task.

        :param x0: initial state
        :param comGoTo: CoM position change target
        :param timeStep: step time for each knot
        :param numKnots: number of knots per each phase
        :return shooting problem
        """
        # Compute the current foot positions
        q0 = self.rmodel.referenceConfigurations["standing"]
        pinocchio.forwardKinematics(self.rmodel, self.rdata, q0)
        pinocchio.updateFramePlacements(self.rmodel, self.rdata)
        com0 = pinocchio.centerOfMass(self.rmodel, self.rdata, q0)

        # Defining the action models along the time instances
        comModels = []

        # Creating the action model for the CoM task
        comForwardModels = [
            self.createSwingFootModel(
                timeStep,
                [self.lfFootId, self.rfFootId, self.lhFootId, self.rhFootId],
            ) for k in range(numKnots)
        ]
        comForwardTermModel = self.createSwingFootModel(timeStep,
                                                        [self.lfFootId, self.rfFootId, self.lhFootId, self.rhFootId],
                                                        com0 + np.array([comGoTo, 0., 0.]))
        comForwardTermModel.differential.costs.costs['comTrack'].weight = 1e6

        # Adding the CoM tasks
        comModels += comForwardModels + [comForwardTermModel]

        # Defining the shooting problem
        problem = crocoddyl.ShootingProblem(x0, comModels, comModels[-1])
        return problem


    def createSwingFootModel(self, timeStep, supportFootIds, comTask=None, swingFootTask=None):
        """ Action model for a swing foot phase.

        :param timeStep: step duration of the action model
        :param supportFootIds: Ids of the constrained feet
        :param comTask: CoM task
        :param swingFootTask: swinging foot task
        :return action model for a swing foot phase
        """
        # Creating a 3D multi-contact model, and then including the supporting
        # foot
        nu = self.actuation.nu
        contactModel = crocoddyl.ContactModelMultiple(self.state, nu)
        for i in supportFootIds:
            supportContactModel = crocoddyl.ContactModel3D(self.state, i, np.array([0., 0., 0.]), nu,
                                                           np.array([0., 50.]))
            contactModel.addContact(self.rmodel.frames[i].name + "_contact", supportContactModel)

        # Creating the cost model for a contact phase
        costModel = crocoddyl.CostModelSum(self.state, nu)
        if isinstance(comTask, np.ndarray):
            comResidual = crocoddyl.ResidualModelCoMPosition(self.state, comTask, nu)
            comTrack = crocoddyl.CostModelResidual(self.state, comResidual)
            costModel.addCost("comTrack", comTrack, 1e6)
        for i in supportFootIds:
            cone = crocoddyl.FrictionCone(self.Rsurf, self.mu, 4, False)
            coneResidual = crocoddyl.ResidualModelContactFrictionCone(self.state, i, cone,
                                                                               self.actuation.nu)
            coneActivation = crocoddyl.ActivationModelQuadraticBarrier(crocoddyl.ActivationBounds(cone.lb, cone.ub))
            frictionCone = crocoddyl.CostModelResidual(self.state, coneActivation, coneResidual)
            costModel.addCost(self.rmodel.frames[i].name + "_frictionCone", frictionCone, 1e1)
        if swingFootTask is not None:
            for i in swingFootTask:
                frameTranslationResidual = crocoddyl.ResidualModelFrameTranslation(self.state, i[0], i[1].translation,
                                                                                   nu)
                footTrack = crocoddyl.CostModelResidual(self.state, frameTranslationResidual)
                costModel.addCost(self.rmodel.frames[i[0]].name + "_footTrack", footTrack, 1e6)

        stateWeights = np.array([1e-1] * 3 + [500.] * 3 + [0.01] * (self.rmodel.nv - 6) + [10.] * 6 + [1.] *
                                (self.rmodel.nv - 6) + [1e-1]*2*self.state.nv_m)
        stateResidual = crocoddyl.ResidualModelState(self.state, self.rmodel.defaultState, nu)
        stateActivation = crocoddyl.ActivationModelWeightedQuad(stateWeights**2)
        ctrlWeights = np.array( [1e0] * nu )
        ctrlResidual = crocoddyl.ResidualModelControl(self.state, nu)
        ctrlActivation = crocoddyl.ActivationModelWeightedQuad(ctrlWeights**2)
        stateReg = crocoddyl.CostModelResidual(self.state, stateActivation, stateResidual)
        ctrlReg = crocoddyl.CostModelResidual(self.state, ctrlActivation, ctrlResidual)
        costModel.addCost("stateReg", stateReg, 1e1)
        costModel.addCost("ctrlReg", ctrlReg, 1e-1)

        # lb = np.concatenate([self.state.lb[1:self.state.nv + 1], self.state.lb[-self.state.nv:]])
        # ub = np.concatenate([self.state.ub[1:self.state.nv + 1], self.state.ub[-self.state.nv:]])
        # stateBoundsResidual = crocoddyl.ResidualModelState(self.state, nu)
        # stateBoundsActivation = crocoddyl.ActivationModelQuadraticBarrier(crocoddyl.ActivationBounds(lb, ub))
        # stateBounds = crocoddyl.CostModelResidual(self.state, stateBoundsActivation, stateBoundsResidual)
        # costModel.addCost("stateBounds", stateBounds, 1e3)

        # Creating the action model for the KKT dynamics with simpletic Euler
        # integration scheme
        dmodel = aslr_to.DifferentialContactASLRFwdDynModel(self.state, self.actuation, contactModel, costModel, self.K, self.B)
        # print(dmodel.nu)
        model = crocoddyl.IntegratedActionModelEuler(dmodel, timeStep)
        return model

