    def max_jump(self, x0, jumpHeight, timestep, groundKnots, flyingKnots):
        # runningCostModel = crocoddyl.CostModelSum(state, self.actuation.nu)
        # terminalCostModel = crocoddyl.CostModelSum(state, self.actuation.nu)
        q0 = x0[:self.rmodel.nq]
        pinocchio.forwardKinematics(self.rmodel, self.rdata, q0)
        pinocchio.updateFramePlacements(self.rmodel, self.rdata)
        comRef = pinocchio.centerOfMass(self.rmodel, self.rdata, q0)
        FootPos0 = self.rdata.oMf[self.FootId].translation
        target = FootPos0 + np.array([0,0, jumpHeight])
        Pref = crocoddyl.FrameTranslation(self.FootId, target)
        # If also the orientation is useful for the task use
        footTrackingCost = crocoddyl.CostModelFrameTranslation(self.state, Pref, self.actuation.nu)
        Vref = crocoddyl.FrameMotion(self.FootId, pinocchio.Motion(np.zeros(6)))
        footFinalVelocity = crocoddyl.CostModelFrameVelocity(self.state, Vref, self.actuation.nu)
        
        # PENALIZATIONS
        bounds = crocoddyl.ActivationBounds(np.concatenate([np.zeros(1), -1e3* np.ones(self.state.nx-1)]), 1e3*np.ones(self.state.nx))
        stateAct = crocoddyl.ActivationModelWeightedQuadraticBarrier(bounds, np.concatenate([np.ones(1), np.zeros(self.state.nx - 1)]))
        nonPenetration = crocoddyl.CostModelState(self.state, stateAct, np.zeros(self.state.nx), self.actuation.nu)

        # MAXIMIZATION
        jumpBounds = crocoddyl.ActivationBounds(-1e3*np.ones(self.state.nx), np.concatenate([np.zeros(1), +1e3* np.ones(self.state.nx-1)]))
        jumpAct = crocoddyl.ActivationModelWeightedQuadraticBarrier(bounds, np.concatenate([-np.ones(1), np.zeros(self.state.nx - 1)]))
        maximizeJump = crocoddyl.CostModelState(self.state, jumpAct, np.ones(self.state.nx), self.actuation.nu)

        # CONTACT MODEL
        contactModel = crocoddyl.ContactModelMultiple(self.state, self.actuation.nu)
        contact_location = crocoddyl.FrameTranslation(self.FootId, np.array([0., 0., 0.]))
        supportContactModel = crocoddyl.ContactModel2D(self.state, contact_location, self.actuation.nu, np.array([0., 1/timestep])) # makes the velocity drift disappear in one timestep
        contactModel.addContact("foot_contact", supportContactModel)

        contactCostModel = crocoddyl.CostModelSum(self.state, self.actuation.nu)
        # FRICTION CONE
        for i in [self.FootId]:
            cone = crocoddyl.FrictionCone(self.Rsurf, self.mu, 4, False)
            coneResidual = crocoddyl.ResidualModelContactFrictionCone(self.state, i, cone,
                                                                               self.actuation.nu)
            coneActivation = crocoddyl.ActivationModelQuadraticBarrier(crocoddyl.ActivationBounds(cone.lb, cone.ub))
            frictionCone = crocoddyl.CostModelResidual(self.state, coneActivation, coneResidual)
            contactCostModel.addCost(self.rmodel.frames[i].name + "_frictionCone", frictionCone, 1e1)

        contactCostModel.addCost('nonPenetration', nonPenetration, 1e5)

        contactDifferentialModel = aslr_to.DifferentialContactASLRFwdDynModel(self.state,
                self.actuation,
                contactModel,
                contactCostModel,
                self.K, self.B) 
        contactPhase = crocoddyl.IntegratedActionModelEuler(contactDifferentialModel, timestep)
        
        flyingDownPhase = []
        for k in range(flyingKnots):
            flyingDownPhase += [self.createSwingFootModel(timestep, [])]
        footTask = [[self.FootId, pinocchio.SE3(np.eye(3), FootPos0)]]
        landingPhase = [
            self.createFootSwitchModel([ self.FootId], footTask,True)
        ]

        landed = [
            self.createSwingFootModel(timestep, [ self.FootId],
                                      comTask=comRef) for k in range(groundKnots)
        ]
        
        flyingDownPhase[-1].differential.costs.addCost("footPose", footTrackingCost, 5e3)

        loco3dModel = [contactPhase] * groundKnots + flyingDownPhase# +landingPhase + landed
        # runningCostModel.addCost("joule_dissipation", joule_dissipation, 5e-3)
        # runningCostModel.addCost('joint_friction', joint_friction, 5e-3)
        # runningCostModel.addCost("velocityRegularization", v2, 1e0)
        # runningCostModel.addCost("nonPenetration", nonPenetration, 1e6)
        # runningCostModel.addCost("maxJump", maximizeJump, 1e2)
        # terminalCostModel.addCost("footPose", footTrackingCost, 5e3)
        # terminalCostModel.addCost("footVelocity", footFinalVelocity, 1e0)

        # runningModel = crocoddyl.IntegratedActionModelEuler(
        #     crocoddyl.DifferentialActionModelFreeFwdDynamics(state, self.actuation, runningCostModel), dt)
        # terminalModel = crocoddyl.IntegratedActionModelEuler(crocoddyl.DifferentialActionModelFreeFwdDynamics(state, self.actuation, terminalCostModel), 0.)

        # runningModel.u_lb = -rmodel.effortLimit[-self.actuation.nu:]
        # runningModel.u_ub = rmodel.effortLimit[-self.actuation.nu:]
        problem = crocoddyl.ShootingProblem(x0, loco3dModel, loco3dModel[-1])
        return problem