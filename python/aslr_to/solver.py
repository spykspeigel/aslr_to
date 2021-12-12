import numpy as np
import crocoddyl
import scipy.linalg as scl

def rev_enumerate(lname):
    return reversed(list(enumerate(lname)))


def raiseIfNan(A, error=None):
    if error is None:
        error = scl.LinAlgError("NaN in array")
    if np.any(np.isnan(A)) or np.any(np.isinf(A)) or np.any(abs(np.asarray(A)) > 1e30):
        raise error


class DDPASLR(crocoddyl.SolverAbstract):
    def __init__(self, shootingProblem):
        crocoddyl.SolverAbstract.__init__(self, shootingProblem)
        self.allocateData()  # TODO remove it?

        self.isFeasible = False
        self.alphas = [2**(-n) for n in range(10)]
        self.th_grad = 1e-12

        self.callbacks = None
        self.x_reg = 0
        self.u_reg = 0
        self.reg_incFactor = 10
        self.reg_decFactor = 10
        self.reg_max = 1e9
        self.reg_min = 1e-9
        self.th_step = .5

    def solve(self, init_xs=[], init_us=[], maxiter=100, isFeasible=False, regInit=None):
        self.setCandidate(init_xs, init_us, isFeasible)
        self.x_reg = regInit if regInit is not None else self.reg_min
        self.u_reg = regInit if regInit is not None else self.reg_min
        self.wasFeasible = False
        for i in range(maxiter):
            print(i)
            recalc = True
            while True:
                try:
                    self.computeDirection(recalc=recalc)
                except ArithmeticError:
                    print("insde recalc")
                    recalc = False
                    self.increaseRegularization()
                    if self.x_reg == self.reg_max:
                        print("disnt work")
                        return self.xs, self.us, False
                    else:
                        continue
                break
            self.d = self.expectedImprovement()
            d1, d2 = np.asscalar(self.d[0]), np.asscalar(self.d[1])
            for a in self.alphas:
                try:
                    self.dV = self.tryStep(a)
                except ArithmeticError:
                    continue
                self.dV_exp = a * (d1 + .5 * d2 * a)
                if self.dV_exp >= 0:
                    if d1 < self.th_grad or not self.isFeasible or self.dV > self.th_acceptStep * self.dV_exp:
                        self.wasFeasible = self.isFeasible
                        self.setCandidate(self.xs_try, self.us_try, True)
                        self.cost = self.cost_try
                        break
            if a > self.th_step:
                self.decreaseRegularization()
            if a == self.alphas[-1]:
                self.increaseRegularization()
                if self.x_reg == self.reg_max:
                    return self.xs, self.us, False
            self.stepLength = a
            self.iter = i
            self.stop = self.stoppingCriteria()

            if self.getCallbacks() is not None:
                print('hey')
                [c(self) for c in self.getCallbacks()]

            if self.wasFeasible and self.stop < self.th_stop:
                print("5th check")
                return self.xs, self.us, True
        return self.xs, self.us, False

    def computeDirection(self, recalc=True):
        if recalc:
            self.calcDiff()
        self.backwardPass()
        return [np.nan] * (self.problem.T + 1), self.k, self.Vx

    def tryStep(self, stepLength=1):
        self.forwardPass(stepLength)
        return self.cost - self.cost_try

    def stoppingCriteria(self):
        return sum([np.dot(q.T, q) for q in self.Qu])

    def expectedImprovement(self):
        d1 = sum([np.dot(q.T, k) for q, k in zip(self.Qu, self.k)])
        d2 = sum([-np.dot(k.T, np.dot(q, k)) for q, k in zip(self.Quu, self.k)])
        return np.array([d1, d2])

    def calcDiff(self):
        if self.iter == 0:
            self.problem.calc(self.xs, self.us)

        self.problem.calcDiff(self.xs, self.us)

        self.cost = self.problem.calcDiff(self.xs, self.us)
        if not self.isFeasible:
            self.fs[0] = self.problem.runningModels[0].state.diff(self.xs[0], self.problem.x0)
            for i, (m, d, x) in enumerate(zip(self.problem.runningModels, self.problem.runningDatas, self.xs[1:])):
                self.fs[i + 1] = m.state.diff(x, d.xnext)
        return self.cost

    def backwardPass(self):
        self.Vx[-1][:] = self.problem.terminalData.Lx
        self.Vxx[-1][:, :] = self.problem.terminalData.Lxx

        if self.x_reg != 0:
            ndx = self.problem.terminalModel.state.ndx
            self.Vxx[-1][range(ndx), range(ndx)] += self.x_reg

        # Compute and store the Vx gradient at end of the interval (rollout state)
        if not self.isFeasible:
            self.Vx[-1] += np.dot(self.Vxx[-1], self.fs[-1])

        for t, (model, data) in rev_enumerate(zip(self.problem.runningModels, self.problem.runningDatas)):
            self.Qxx[t][:, :] = data.Lxx + np.dot(data.Fx.T, np.dot(self.Vxx[t + 1], data.Fx))
            self.Qxu[t][:, :] = data.Lxu + np.dot(data.Fx.T, np.dot(self.Vxx[t + 1], data.Fu))
            self.Quu[t][:, :] = data.Luu + np.dot(data.Fu.T, np.dot(self.Vxx[t + 1], data.Fu))
            self.Qx[t][:] = data.Lx + np.dot(data.Fx.T, self.Vx[t + 1])
            self.Qu[t][:] = data.Lu + np.dot(data.Fu.T, self.Vx[t + 1])

            if self.u_reg != 0:
                self.Quu[t][range(model.nu), range(model.nu)] += self.u_reg
            self.computeGains(t)

            self.Vx[t][:] = self.Qx[t] - np.dot(self.K[t].T, self.Qu[t])
            self.Vxx[t][:, :] = self.Qxx[t] - np.dot(self.Qxu[t], self.K[t])
            self.Vxx[t][:, :] = 0.5 * (self.Vxx[t][:, :] + self.Vxx[t][:, :].T)  # ensure symmetric

            if self.x_reg != 0:
                self.Vxx[t][range(model.state.ndx), range(model.state.ndx)] += self.x_reg

            # Compute and store the Vx gradient at end of the interval (rollout state)
            if not self.isFeasible:
                self.Vx[t] += np.dot(self.Vxx[t], self.fs[t])

            raiseIfNan(self.Vxx[t], ArithmeticError('backward error'))
            raiseIfNan(self.Vx[t], ArithmeticError('backward error'))
    def forwardPass(self, stepLength, warning='ignore'):
        xs, us = self.xs, self.us
        xtry, utry = self.xs_try, self.us_try
        ctry = 0
        for t, (m, d) in enumerate(zip(self.problem.runningModels, self.problem.runningDatas)):
            utry[t] = us[t] - self.k[t] * stepLength - np.dot(self.K[t], m.state.diff(xs[t], xtry[t]))
            with np.warnings.catch_warnings():
                np.warnings.simplefilter(warning)
                m.calc(d, xtry[t], utry[t])
                xnext, cost = d.xnext, d.cost
            xtry[t + 1] = xnext.copy()  # not sure copy helpful here.
            ctry += cost
            raiseIfNan([ctry, cost], ArithmeticError('forward error'))
            raiseIfNan(xtry[t + 1], ArithmeticError('forward error'))
        with np.warnings.catch_warnings():
            np.warnings.simplefilter(warning)
            self.problem.terminalModel.calc(self.problem.terminalData, xtry[-1])
            ctry += self.problem.terminalData.cost
        raiseIfNan(ctry, ArithmeticError('forward error'))
        self.cost_try = ctry
        return xtry, utry, ctry

    def computeGains(self, t):
        # try:
        #     print("hey")
        #     if self.Quu[t].shape[0] > 0:
        Lb = scl.cho_factor(self.Quu[t])
        self.K[t][:, :] = scl.cho_solve(Lb, self.Qux[t])
        self.k[t][:] = scl.cho_solve(Lb, self.Qu[t])
        #     else:
        #         pass
        # except scl.LinAlgError:
        #     raise ArithmeticError('backward error')

    def increaseRegularization(self):
        self.x_reg *= self.reg_incFactor
        if self.x_reg > self.reg_max:
            self.x_reg = self.reg_max
        self.u_reg = self.x_reg

    def decreaseRegularization(self):
        self.x_reg /= self.reg_decFactor
        if self.x_reg < self.reg_min:
            self.x_reg = self.reg_min
        self.u_reg = self.x_reg

    def allocateData(self):
        models = self.problem.runningModels.tolist() + [self.problem.terminalModel]
        self.Vxx = [np.zeros([m.state.ndx, m.state.ndx]) for m in models]
        self.Vx = [np.zeros([m.state.ndx]) for m in models]

        self.Q = [np.zeros([m.state.ndx + m.nu, m.state.ndx + m.nu]) for m in self.problem.runningModels]
        self.q = [np.zeros([m.state.ndx + m.nu]) for m in self.problem.runningModels]
        self.Qxx = [Q[:m.state.ndx, :m.state.ndx] for m, Q in zip(self.problem.runningModels, self.Q)]
        self.Qxu = [Q[:m.state.ndx, m.state.ndx:] for m, Q in zip(self.problem.runningModels, self.Q)]
        self.Qux = [Qxu.T for m, Qxu in zip(self.problem.runningModels, self.Qxu)]
        self.Quu = [Q[m.state.ndx:, m.state.ndx:] for m, Q in zip(self.problem.runningModels, self.Q)]
        self.Qx = [q[:m.state.ndx] for m, q in zip(self.problem.runningModels, self.q)]
        self.Qu = [q[m.state.ndx:] for m, q in zip(self.problem.runningModels, self.q)]

        self.K = [np.zeros([m.nu, m.state.ndx]) for m in self.problem.runningModels]
        self.k = [np.zeros([m.nu]) for m in self.problem.runningModels]

        self.xs_try = [self.problem.x0] + [np.nan * self.problem.x0] * self.problem.T
        self.us_try = [np.nan] * self.problem.T
        self.fs = [np.zeros(self.problem.runningModels[0].state.ndx)
                   ] + [np.zeros(m.state.ndx) for m in self.problem.runningModels]
