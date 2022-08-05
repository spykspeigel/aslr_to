import crocoddyl
import pinocchio
import numpy as np
import aslr_to

import copy 
class RigidMonopedProblem:
    def __init__(self, rmodel, rhFoot):
        self.rmodel = rmodel
        self.rdata = rmodel.createData()
        self.state = crocoddyl.StateMultibody(self.rmodel)
        self.K = np.zeros([self.state.pinocchio.nv,self.state.pinocchio.nq])
        nu = self.state.nv
        self.K[-nu:,-nu:]= 100*np.eye(nu)
        self.B = .001*np.eye(self.state.nv)
        self.actuation = aslr_to.RigidLegActuation(self.state)

        # Getting the frame id for all the legs
        self.FootId = self.rmodel.getFrameId('softleg_1_contact_link')
        # Defining default state
        # q0 = self.rmodel.referenceConfigurations["standing"]
        self.q0 =np.array([0,0,0])

        angle = np.pi/4
        # self.q0[0] = .1 * np.cos(angle)
        # self.q0[1] = np.pi - angle
        # self.q0[2] = .1 * angle

        # OPTION 2 Initial configuration distributing the joints in a semicircle with foot in O (scalable if n_joints > 2)
        self.q0[0] = .36
        self.q0[1] = -np.pi/3
        self.q0[2] = np.pi/3

        # OPTION 3 Solo, (the convention used has negative displacements)
        # q0[0] = 0.16 / np.sin(np.pi/(2 * 2))
        # q0[1] = np.pi/4
        # q0[2] = -np.pi/2

        self.rmodel.defaultState = np.concatenate([self.q0, np.zeros(self.rmodel.nv)])
        print(self.rmodel.defaultState.shape)
        self.firstStep = True
        # Defining the friction coefficient and normal
        self.mu = 0.7
        self.Rsurf = np.eye(3)

    def createJumpingProblem(self, x0, jumpHeight, jumpLength, timeStep, groundKnots, flyingKnots):
        q0 = x0[:self.rmodel.nq]
        pinocchio.forwardKinematics(self.rmodel, self.rdata, q0)
        pinocchio.updateFramePlacements(self.rmodel, self.rdata)
        FootPos0 = self.rdata.oMf[self.FootId].translation
        df = jumpLength[2] - FootPos0[2]

        FootPos0[2] = 0.
        # comRef = copy.copy(FootPos0 )
        comRef = pinocchio.centerOfMass(self.rmodel, self.rdata, q0)
        loco3dModel = []
        takeOff = [
            self.createSwingFootModel(
                timeStep,
                [ self.FootId],
            ) for k in range(groundKnots)
        ]

        flyingUpPhase = [
            self.createSwingFootModel(
                timeStep, [])
                
                for k in range(int(flyingKnots/1.2))
        ]
        flyingUpPhase2 = [
            self.createSwingFootModel(
                timeStep, [],
                comTask = np.array([jumpLength[0], jumpLength[1], jumpLength[2]+jumpHeight]) * (k + 1) / flyingKnots+ comRef)
                for k in range(flyingKnots-int(flyingKnots/1.2))
        ]

        flyingDownPhase = []
        for k in range(flyingKnots):
            flyingDownPhase += [self.createSwingFootModel(timeStep, [])]
        f0 = jumpLength

        footTask = [[self.FootId, pinocchio.SE3(np.eye(3), FootPos0)]]
        landingPhase = [
            self.createFootSwitchModel([ self.FootId], footTask,True)
        ]
        f0[2] = df

        # return
        landed = [
            self.createSwingFootModel(timeStep, [ self.FootId],
                                      comTask=comRef) for k in range(groundKnots)
        ]
        loco3dModel += takeOff
        loco3dModel += flyingUpPhase
        loco3dModel += flyingUpPhase2 
        # loco3dModel += flyingDownPhase
        # loco3dModel += landingPhase
        # loco3dModel += landed

        problem = crocoddyl.ShootingProblem(x0, loco3dModel, loco3dModel[-1])
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
            xref = crocoddyl.FrameTranslation(i, np.array([0., 0., 0.]))
            supportContactModel = crocoddyl.ContactModel2D(self.state, xref, nu, np.array([0., 50.]))
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

        # stateWeights = np.array([0.] * 3 + [500.] * 3 + [0.01] * (self.rmodel.nv - 6) + [10.] * 6 + [1.] *
        #                         (self.rmodel.nv - 6) + [1e0]*self.state.nv+ [1e-1]*self.state.nv)
        stateWeights = np.array([0.]   + [1] * (self.rmodel.nv - 1) + [1.] * 1 + [1.] *
                                (self.rmodel.nv - 1) )
        stateResidual = crocoddyl.ResidualModelState(self.state, self.rmodel.defaultState, nu)
        stateActivation = crocoddyl.ActivationModelWeightedQuad(stateWeights**2)
        ctrlWeights = np.array( [1e0] * nu )
        ctrlResidual = crocoddyl.ResidualModelControl(self.state, nu)
        ctrlActivation = crocoddyl.ActivationModelWeightedQuad(ctrlWeights**2)
        stateReg = crocoddyl.CostModelResidual(self.state, stateActivation, stateResidual)
        ctrlReg = crocoddyl.CostModelResidual(self.state, ctrlActivation, ctrlResidual)
        costModel.addCost("stateReg", stateReg, 1e0)
        costModel.addCost("ctrlReg", ctrlReg, 1e-1)

        # lb = np.concatenate([self.state.lb[1:self.state.nv + 1], self.state.lb[-self.state.nv:]])
        # ub = np.concatenate([self.state.ub[1:self.state.nv + 1], self.state.ub[-self.state.nv:]])
        # stateBoundsResidual = crocoddyl.ResidualModelState(self.state, nu)
        # stateBoundsActivation = crocoddyl.ActivationModelQuadraticBarrier(crocoddyl.ActivationBounds(lb, ub))
        # stateBounds = crocoddyl.CostModelResidual(self.state, stateBoundsActivation, stateBoundsResidual)
        # costModel.addCost("stateBounds", stateBounds, 1e2)

        # Creating the action model for the KKT dynamics with simpletic Euler
        # integration scheme
        dmodel = crocoddyl.DifferentialActionModelContactFwdDynamics(self.state, self.actuation, contactModel,
                                                                     costModel, 0., True)
        model = crocoddyl.IntegratedActionModelEuler(dmodel, timeStep)
        return model

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
            xref = crocoddyl.FrameTranslation(i, np.array([0., 0., 0.]))
            supportContactModel = crocoddyl.ContactModel2D(self.state, xref, nu, np.array([0., 50.]))
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

        stateWeights = np.array([0.]   + [0.01] * (self.rmodel.nv - 1) + [10.] * 1 + [1.] *
                                (self.rmodel.nv - 1))
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
        dmodel = crocoddyl.DifferentialActionModelContactFwdDynamics(self.state, self.actuation, contactModel,
                                                                     costModel, 0., True)
        # dmodel = aslr_to.DifferentialContactASLRFwdDynModel(self.state, self.actuation, contactModel, costModel, self.K, self.B)
        # print(dmodel.nu)
        model = crocoddyl.IntegratedActionModelEuler(dmodel, 0.)
        return model
