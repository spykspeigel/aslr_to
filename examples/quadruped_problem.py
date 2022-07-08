import crocoddyl
import pinocchio
import numpy as np
import aslr_to


class SimpleQuadrupedalGaitProblem:
    def __init__(self, rmodel, lfFoot, rfFoot, lhFoot, rhFoot):
        self.rmodel = rmodel
        self.rdata = rmodel.createData()
        self.state = crocoddyl.StateSoftMultibody(self.rmodel)
        self.K = np.zeros([self.state.pinocchio.nv,self.state.pinocchio.nq])
        nu = self.state.nv_m
        self.K[-nu:,-nu:]= 10*np.eye(nu)
        self.B = .001*np.eye(self.state.nv_m)
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
            costModel.addCost("comTrack", comTrack, 1e4)
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
                costModel.addCost(self.rmodel.frames[i[0]].name + "_footTrack", footTrack, 1e5)

        stateWeights = np.array([0.] * 3 + [500.] * 3 + [0.01] * (self.rmodel.nv - 6) + [10.] * 6 + [1.] *
                                (self.rmodel.nv - 6) + [1e0]*self.state.nv_m+ [1e-1]*self.state.nv_m)
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
        # costModel.addCost("stateBounds", stateBounds, 1e2)

        # Creating the action model for the KKT dynamics with simpletic Euler
        # integration scheme
        dmodel = aslr_to.DifferentialContactASLRFwdDynModel(self.state, self.actuation, contactModel, costModel, self.K, self.B)
        # print(dmodel.nu)
        model = crocoddyl.IntegratedActionModelEuler(dmodel, timeStep)
        return model


    def createWalkingProblem(self, x0, stepLength, stepHeight, timeStep, stepKnots, supportKnots):
        """ Create a shooting problem for a simple walking gait.

        :param x0: initial state
        :param stepLength: step length
        :param stepHeight: step height
        :param timeStep: step time for each knot
        :param stepKnots: number of knots for step phases
        :param supportKnots: number of knots for double support phases
        :return shooting problem
        """
        # Compute the current foot positions
        q0 = x0[:self.rmodel.nq]
        pinocchio.forwardKinematics(self.rmodel, self.rdata, q0)
        pinocchio.updateFramePlacements(self.rmodel, self.rdata)
        rfFootPos0 = self.rdata.oMf[self.rfFootId].translation
        rhFootPos0 = self.rdata.oMf[self.rhFootId].translation
        lfFootPos0 = self.rdata.oMf[self.lfFootId].translation
        lhFootPos0 = self.rdata.oMf[self.lhFootId].translation
        comRef = (rfFootPos0 + rhFootPos0 + lfFootPos0 + lhFootPos0) / 4
        comRef[2] = pinocchio.centerOfMass(self.rmodel, self.rdata, q0)[2].item()

        # Defining the action models along the time instances
        loco3dModel = []
        doubleSupport = [
            self.createSwingFootModel(
                timeStep,
                [self.lfFootId, self.rfFootId, self.lhFootId, self.rhFootId],
            ) for k in range(supportKnots)
        ]
        if self.firstStep is True:
            rhStep = self.createFootstepModels(comRef, [rhFootPos0], 0.5 * stepLength, stepHeight, timeStep, stepKnots,
                                                [self.lfFootId, self.rfFootId, self.lhFootId], [self.rhFootId])
            rfStep = self.createFootstepModels(comRef, [rfFootPos0], 0.5 * stepLength, stepHeight, timeStep, stepKnots,
                                               [self.lfFootId, self.lhFootId, self.rhFootId], [self.rfFootId])
            self.firstStep = False
        else:
            rhStep = self.createFootstepModels(comRef, [rhFootPos0], stepLength, stepHeight, timeStep, stepKnots,
                                               [self.lfFootId, self.rfFootId, self.lhFootId], [self.rhFootId])
            rfStep = self.createFootstepModels(comRef, [rfFootPos0], stepLength, stepHeight, timeStep, stepKnots,
                                               [self.lfFootId, self.lhFootId, self.rhFootId], [self.rfFootId])
        lhStep = self.createFootstepModels(comRef, [lhFootPos0], stepLength, stepHeight, timeStep, stepKnots,
                                           [self.lfFootId, self.rfFootId, self.rhFootId], [self.lhFootId])
        lfStep = self.createFootstepModels(comRef, [lfFootPos0], stepLength, stepHeight, timeStep, stepKnots,
                                           [self.rfFootId, self.lhFootId, self.rhFootId], [self.lfFootId])
        loco3dModel += doubleSupport + rhStep + rfStep
        loco3dModel += doubleSupport + lhStep #+ lfStep

        problem = crocoddyl.ShootingProblem(x0, loco3dModel, loco3dModel[-1])
        return problem


    def createFootstepModels(self, comPos0, feetPos0, stepLength, stepHeight, timeStep, numKnots, supportFootIds,
                             swingFootIds):
        """ Action models for a footstep phase.

        :param comPos0, initial CoM position
        :param feetPos0: initial position of the swinging feet
        :param stepLength: step length
        :param stepHeight: step height
        :param timeStep: time step
        :param numKnots: number of knots for the footstep phase
        :param supportFootIds: Ids of the supporting feet
        :param swingFootIds: Ids of the swinging foot
        :return footstep action models
        """
        numLegs = len(supportFootIds) + len(swingFootIds)
        comPercentage = float(len(swingFootIds)) / numLegs

        # Action models for the foot swing
        footSwingModel = []
        for k in range(numKnots):
            swingFootTask = []
            for i, p in zip(swingFootIds, feetPos0):
                # Defining a foot swing task given the step length
                # resKnot = numKnots % 2
                phKnots = numKnots / 2
                if k < phKnots:
                    dp = np.array([stepLength * (k + 1) / numKnots, 0., stepHeight * k / phKnots])
                elif k == phKnots:
                    dp = np.array([stepLength * (k + 1) / numKnots, 0., stepHeight])
                else:
                    dp = np.array(
                        [stepLength * (k + 1) / numKnots, 0., stepHeight * (1 - float(k - phKnots) / phKnots)])
                tref = p + dp

                swingFootTask += [[i, pinocchio.SE3(np.eye(3), tref)]]

            comTask = np.array([stepLength * (k + 1) / numKnots, 0., 0.]) * comPercentage + comPos0
            footSwingModel += [
                self.createSwingFootModel(timeStep, supportFootIds, comTask=comTask, swingFootTask=swingFootTask)
            ]

        # Action model for the foot switch
        footSwitchModel = self.createFootSwitchModel(supportFootIds, swingFootTask,False)

        # Updating the current foot position for next step
        comPos0 += [stepLength * comPercentage, 0., 0.]
        for p in feetPos0:
            p += [stepLength, 0., 0.]
        return footSwingModel + [footSwitchModel]

    def createJumpingProblem(self, x0, jumpHeight, jumpLength, timeStep, groundKnots, flyingKnots):
        q0 = x0[:self.rmodel.nq]
        pinocchio.forwardKinematics(self.rmodel, self.rdata, q0)
        pinocchio.updateFramePlacements(self.rmodel, self.rdata)
        rfFootPos0 = self.rdata.oMf[self.rfFootId].translation
        rhFootPos0 = self.rdata.oMf[self.rhFootId].translation
        lfFootPos0 = self.rdata.oMf[self.lfFootId].translation
        lhFootPos0 = self.rdata.oMf[self.lhFootId].translation
        df = jumpLength[2] - rfFootPos0[2]
        rfFootPos0[2] = 0.
        rhFootPos0[2] = 0.
        lfFootPos0[2] = 0.
        lhFootPos0[2] = 0.
        comRef = (rfFootPos0 + rhFootPos0 + lfFootPos0 + lhFootPos0) / 4
        comRef[2] = pinocchio.centerOfMass(self.rmodel, self.rdata, q0)[2].item()

        loco3dModel = []
        takeOff = [
            self.createSwingFootModel(
                timeStep,
                [self.lfFootId, self.rfFootId, self.lhFootId, self.rhFootId],
            ) for k in range(groundKnots)
        ]
        flyingUpPhase = [
            self.createSwingFootModel(
                timeStep, [],
                np.array([jumpLength[0], jumpLength[1], jumpLength[2] + jumpHeight]) * (k + 1) / flyingKnots + comRef)
            for k in range(flyingKnots)
        ]
        flyingDownPhase = []
        for k in range(flyingKnots):
            flyingDownPhase += [self.createSwingFootModel(timeStep, [])]

        f0 = jumpLength
        footTask = [[self.lfFootId, pinocchio.SE3(np.eye(3), lfFootPos0 + f0)],
                    [self.rfFootId, pinocchio.SE3(np.eye(3), rfFootPos0 + f0)],
                    [self.lhFootId, pinocchio.SE3(np.eye(3), lhFootPos0 + f0)],
                    [self.rhFootId, pinocchio.SE3(np.eye(3), rhFootPos0 + f0)]]
        landingPhase = [
            self.createFootSwitchModel([self.lfFootId, self.rfFootId, self.lhFootId, self.rhFootId], footTask)
        ]
        f0[2] = df
        landed = [
            self.createSwingFootModel(timeStep, [self.lfFootId, self.rfFootId, self.lhFootId, self.rhFootId],
                                      comTask=comRef + f0) for k in range(groundKnots)
        ]
        loco3dModel += takeOff
        loco3dModel += flyingUpPhase
        loco3dModel += flyingDownPhase
        loco3dModel += landingPhase
        loco3dModel += landed

        problem = crocoddyl.ShootingProblem(x0, loco3dModel, loco3dModel[-1])
        return problem

    def createTrottingProblem(self, x0, stepLength, stepHeight, timeStep, stepKnots, supportKnots):
        """ Create a shooting problem for a simple trotting gait.

        :param x0: initial state
        :param stepLength: step length
        :param stepHeight: step height
        :param timeStep: step time for each knot
        :param stepKnots: number of knots for step phases
        :param supportKnots: number of knots for double support phases
        :return shooting problem
        """
        # Compute the current foot positions
        q0 = x0[:self.rmodel.nq]
        pinocchio.forwardKinematics(self.rmodel, self.rdata, q0)
        pinocchio.updateFramePlacements(self.rmodel, self.rdata)
        rfFootPos0 = self.rdata.oMf[self.rfFootId].translation
        rhFootPos0 = self.rdata.oMf[self.rhFootId].translation
        lfFootPos0 = self.rdata.oMf[self.lfFootId].translation
        lhFootPos0 = self.rdata.oMf[self.lhFootId].translation
        comRef = (rfFootPos0 + rhFootPos0 + lfFootPos0 + lhFootPos0) / 4
        comRef[2] = pinocchio.centerOfMass(self.rmodel, self.rdata, q0)[2].item()

        # Defining the action models along the time instances
        loco3dModel = []
        doubleSupport = [
            self.createSwingFootModel(
                timeStep,
                [self.lfFootId, self.rfFootId, self.lhFootId, self.rhFootId],
            ) for k in range(supportKnots)
        ]
        if self.firstStep is True:
            rflhStep = self.createFootstepModels(comRef, [rfFootPos0, lhFootPos0], 0.5 * stepLength, stepHeight,
                                                 timeStep, stepKnots, [self.lfFootId, self.rhFootId],
                                                 [self.rfFootId, self.lhFootId])
            self.firstStep = False
        else:
            rflhStep = self.createFootstepModels(comRef, [rfFootPos0, lhFootPos0], stepLength, stepHeight, timeStep,
                                                 stepKnots, [self.lfFootId, self.rhFootId],
                                                 [self.rfFootId, self.lhFootId])
        lfrhStep = self.createFootstepModels(comRef, [lfFootPos0, rhFootPos0], stepLength, stepHeight, timeStep,
                                             stepKnots, [self.rfFootId, self.lhFootId], [self.lfFootId, self.rhFootId])

        loco3dModel += doubleSupport + rflhStep
        loco3dModel += doubleSupport + lfrhStep

        problem = crocoddyl.ShootingProblem(x0, loco3dModel[:-1], loco3dModel[-1])
        return problem

    def createFootSwitchModel(self, supportFootIds, swingFootTask, pseudoImpulse=False):
        """ Action model for a foot switch phase.

        :param supportFootIds: Ids of the constrained feet
        :param swingFootTask: swinging foot task
        :param pseudoImpulse: true for pseudo-impulse models, otherwise it uses the impulse model
        :return action model for a foot switch phase
        """
        if pseudoImpulse:
            return self.createPseudoImpulseModel(supportFootIds, swingFootTask)
        else:
            return self.createImpulseModel(supportFootIds, swingFootTask)

    def createPseudoImpulseModel(self, supportFootIds, swingFootTask):
        """ Action model for pseudo-impulse models.

        A pseudo-impulse model consists of adding high-penalty cost for the contact velocities.
        :param supportFootIds: Ids of the constrained feet
        :param swingFootTask: swinging foot task
        :return pseudo-impulse differential action model
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
                frameVelocityResidual = crocoddyl.ResidualModelFrameVelocity(self.state, i[0], pinocchio.Motion.Zero(),
                                                                             pinocchio.LOCAL, nu)
                footTrack = crocoddyl.CostModelResidual(self.state, frameTranslationResidual)
                impulseFootVelCost = crocoddyl.CostModelResidual(self.state, frameVelocityResidual)
                costModel.addCost(self.rmodel.frames[i[0]].name + "_footTrack", footTrack, 1e6)
                costModel.addCost(self.rmodel.frames[i[0]].name + "_impulseVel", impulseFootVelCost, 1e7)

        stateWeights = np.array([0.] * 3 + [500.] * 3 + [10] * (self.rmodel.nv - 6) + [10.] * 6 + [10.] *
                                (self.rmodel.nv - 6) + [1e0]*2*self.state.nv_m)
        stateResidual = crocoddyl.ResidualModelState(self.state, self.rmodel.defaultState, nu)
        stateActivation = crocoddyl.ActivationModelWeightedQuad(stateWeights**2)
        ctrlWeights = np.array( [1e0] * nu )
        ctrlResidual = crocoddyl.ResidualModelControl(self.state, nu)
        ctrlActivation = crocoddyl.ActivationModelWeightedQuad(ctrlWeights**2)
        stateReg = crocoddyl.CostModelResidual(self.state, stateActivation, stateResidual)
        ctrlReg = crocoddyl.CostModelResidual(self.state, ctrlActivation, ctrlResidual)
        costModel.addCost("stateReg", stateReg, 1e1)
        costModel.addCost("ctrlReg", ctrlReg, 1e-3)

        # lb = np.concatenate([self.state.lb[1:self.state.nv + 1], self.state.lb[-self.state.nv:]])
        # ub = np.concatenate([self.state.ub[1:self.state.nv + 1], self.state.ub[-self.state.nv:]])
        # stateBoundsResidual = crocoddyl.ResidualModelState(self.state, nu)
        # stateBoundsActivation = crocoddyl.ActivationModelQuadraticBarrier(crocoddyl.ActivationBounds(lb, ub))
        # stateBounds = crocoddyl.CostModelResidual(self.state, stateBoundsActivation, stateBoundsResidual)
        # costModel.addCost("stateBounds", stateBounds, 1e1)

        # Creating the action model for the KKT dynamics with simpletic Euler
        # integration scheme
        dmodel = aslr_to.DifferentialContactASLRFwdDynModel(self.state, self.actuation, contactModel, costModel, self.K, self.B,self.S)
        model = crocoddyl.IntegratedActionModelEuler(dmodel, 0.)
        return model

    def createImpulseModel(self, supportFootIds, swingFootTask, JMinvJt_damping=1e-12, r_coeff=0.0):
        """ Action model for impulse models.

        An impulse model consists of describing the impulse dynamics against a set of contacts.
        :param supportFootIds: Ids of the constrained feet
        :param swingFootTask: swinging foot task
        :return impulse action model
        """
        # Creating a 3D multi-contact model, and then including the supporting foot
        nu = self.actuation.nu
        impulseModel = crocoddyl.ImpulseModelMultiple(self.state)
        for i in supportFootIds:
            supportContactModel = crocoddyl.ImpulseModel3D(self.state, i)
            impulseModel.addImpulse(self.rmodel.frames[i].name + "_impulse", supportContactModel)

        # Creating the cost model for a contact phase
        costModel = crocoddyl.CostModelSum(self.state, 0)
        if swingFootTask is not None:
            for i in swingFootTask:
                frameTranslationResidual = crocoddyl.ResidualModelFrameTranslation(self.state, i[0], i[1].translation,
                                                                                   0)
                footTrack = crocoddyl.CostModelResidual(self.state, frameTranslationResidual)
                costModel.addCost(self.rmodel.frames[i[0]].name + "_footTrack", footTrack, 1e7)

        stateWeights = np.array([0.] * 3 + [500.] * 3 + [10] * (self.rmodel.nv - 6) + [10.] * 6 + [1.] *
                                (self.rmodel.nv - 6) + [1e0]*2*self.state.nv_m)
        stateResidual = crocoddyl.ResidualModelState(self.state, self.rmodel.defaultState,0)
        stateActivation = crocoddyl.ActivationModelWeightedQuad(stateWeights**2)
        stateReg = crocoddyl.CostModelResidual(self.state, stateActivation, stateResidual)
        costModel.addCost("stateReg", stateReg, 1e1)

        # Creating the action model for the KKT dynamics with simpletic Euler
        # integration scheme
        model = crocoddyl.ActionModelImpulseFwdDynamics(self.state, impulseModel, costModel)
        model.JMinvJt_damping = JMinvJt_damping
        model.r_coeff = r_coeff
        return model
