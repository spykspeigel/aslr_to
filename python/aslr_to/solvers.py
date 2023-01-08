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

class SolverINTRO(crocoddyl.SolverAbstract):
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
            recalc = True
            while True:
                try:
                    self.computeDirection(recalc=recalc)
                except ArithmeticError:
                    recalc = False
                    self.increaseRegularization()
                    if self.x_reg == self.reg_max:
                        return self.xs, self.us, False
                    else:
                        continue
                break
            self.d = self.expectedImprovement()
            d1, d2 = self.d[0].item(), self.d[1].item()

            for a in self.alphas:
                print("he1")
                try:
                    self.dV = self.tryStep(a)
                    print("he2")
                except ArithmeticError:
                    continue
                self.dV_exp = a * (d1 + .5 * d2 * a)
                print(self.dV_exp)
                if self.dV_exp >= 0:
                    print(d1)
                    if d1 < self.th_grad or not self.isFeasible or self.dV > self.th_acceptStep * self.dV_exp:
                        # Accept step
                        print("he3")
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
            if self.getCallbacks is not None:
                [c(self) for c in self.getCallbacks()]
            
            if all(self.tau[0]) < self.th_2:                     # if tau is small enough
                print('Solution found!')
            else:
                # decrease tau and continue
                for i in range(len(self.tau)):
                    self.tau[i] *= self.k_b

            print(self.wasFeasible)
            if self.wasFeasible and self.stop < self.th_stop:
                return self.xs, self.us, True
        return self.xs, self.us, False

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

            self.computeGains(model,data,t)

            self.Vx[t][:] = self.Qx[t]
            self.Vxx[t][:, :] = self.Qxx[t]
            if model.nu != 0:
                self.Vx[t][:] -= np.dot(self.K[t].T, self.Qu[t])
                self.Vx[t][:] -= np.dot(self.Qxu[t], self.k[t])
                self.Vx[t][:] += np.dot(self.K[t].T, np.dot(self.Quu[t], self.k[t]))
                self.Vxx[t][:] -= 2 * np.dot(self.Qxu[t], self.K[t])
                self.Vxx[t][:] += np.dot(self.K[t].T, np.dot(self.Quu[t], self.K[t]))
            self.Vxx[t][:, :] = 0.5 * (self.Vxx[t][:, :] + self.Vxx[t][:, :].T)  # ensure symmetric
            if self.x_reg != 0:
                self.Vxx[t][range(model.state.ndx), range(model.state.ndx)] += self.x_reg

            # Compute and store the Vx gradient at end of the interval (rollout state)
            if not self.isFeasible:
                self.Vx[t] += np.dot(self.Vxx[t], self.fs[t])

            raiseIfNan(self.Vxx[t], ArithmeticError('backward error'))
            raiseIfNan(self.Vx[t], ArithmeticError('backward error'))


    def computeGains(self, model, data, t):
        try:
            if model.nu > 0 and model.nh>0 and model.ng > 0:
                S = np.diag(self.s[t])
                Lamda = np.diag(self.lamda[t])
                Jc = data.Hu
                M = self.Quu[t] + np.dot(data.Gu.T,np.dot(np.linalg.inv(S),np.dot(Lamda,data.Gu)))
                Mhat = np.dot(Jc,np.dot(np.linalg.inv(M),Jc.T))

                Minv = np.linalg.inv(M)
                Mhat_inv = np.linalg.inv(Mhat)
                L1 = Minv - np.dot(Minv,np.dot(np.dot(np.dot(Jc.T,Mhat_inv),Jc),Minv))
                L2 = np.dot(Minv,np.dot(Jc.T,Mhat_inv))
                L3 = np.dot(np.dot(Mhat_inv,Jc),Minv)
                L4 = - Mhat_inv
                SinvLamda = np.dot(np.linalg.inv(S),Lamda)

                Psi = -self.Qxu[t].T-np.dot(data.Gu.T,np.dot(SinvLamda,data.Gx))
                Phi = -self.Qu[t] - np.dot(data.Hu.T,self.pi[t]) - 3*np.dot(data.Gu.T,self.lamda[t]) \
                 - np.dot(data.Gu.T,np.dot(np.dot(np.linalg.inv(S),Lamda),data.g)) + \
                    np.dot(data.Gu.T,np.dot(np.linalg.inv(S),self.tau[t]))
                self.k[t] =  -(np.dot(L1,Phi) -np.dot(L2,data.h))
                self.K[t] = -(np.dot(L1,Psi) - np.dot(L2,data.Hx))
                self.pi_k[t] = (np.dot(L3,Phi) + np.dot(L4,data.h))
                self.pi_K[t] = (np.dot(L3,Psi) + np.dot(L4,data.Hx))

            if model.nu > 0 and model.ng > 0:
                S = np.diag(self.s[t])
                Lamda = np.diag(self.lamda[t])
                SinvLamda = np.dot(np.linalg.inv(S),Lamda)

                M = self.Quu[t] + np.dot(data.Gu.T,np.dot(np.linalg.inv(S),np.dot(Lamda,data.Gu)))
                Minv = np.linalg.inv(M)
                Psi = -self.Qxu[t].T - np.dot(data.Gu.T,np.dot(SinvLamda,data.Gx))
                Phi = -self.Qu[t] - 3*np.dot(data.Gu.T,self.lamda[t])  \
                    +np.dot(data.Gu.T,np.dot(np.linalg.inv(S),self.tau[t])) - np.dot(data.Gu.T,np.dot(np.dot(np.linalg.inv(S),Lamda),data.g)) 

                self.k[t][:] = -np.dot(Minv, Phi)
                self.K[t][:,:] = -np.dot(Minv,Psi) 
        except scl.LinAlgError:
            print('backward error at ' + str(t) + ': constrained step')
            raise ArithmeticError('backward error')

    def forwardPass(self, stepLength, warning='ignore'):
        xs, us = self.xs, self.us
        xtry, utry = self.xs_try, self.us_try
        ctry = 0
        for t, (m, d) in enumerate(zip(self.problem.runningModels, self.problem.runningDatas)):
            utry[t] = us[t] - self.k[t] * stepLength - np.dot(self.K[t], m.state.diff(xs[t], xtry[t]))
            if m.nh > 0:
                self.pi[t] += - self.pi_k[t] * stepLength - np.dot(self.pi_K[t], m.state.diff(xs[t], xtry[t]))
            with np.warnings.catch_warnings():
                np.warnings.simplefilter(warning)
                m.calc(d, xtry[t], utry[t])
                xnext, cost = d.xnext, d.cost
            xtry[t + 1] = xnext.copy()  # not sure copy helpful here.
            ctry += cost

            #updating the lagrange multipliers 'lamda', and also the slack variable 's'
            if m.nh>0:
                S = np.diag(self.s[t])
                Lamda = np.diag(self.lamda[t])
                SinvLamda = np.dot(np.linalg.inv(S),Lamda)
                rs = self.tau[t]-np.dot(S,self.lamda[t])
                g = np.dot(d.Gu,utry[t]-us[t]) + np.dot(d.Gx,xtry[t]-xs[t]) + d.g + self.s[t] + np.dot(np.linalg.inv(Lamda),rs)
                del_Lamda = np.dot(SinvLamda,g)

                alphas = [2**(-n) for n in range(100)]
                alphal = [2**(-n) for n in range(100)]
                for i in alphas:
                    self.s[t] += i*(np.dot(np.linalg.inv(Lamda),rs) - np.dot(np.linalg.inv(SinvLamda),del_Lamda))
                    self.lamda[t] += i*(del_Lamda)
                    if all(self.s[t]>0) and all(self.lamda[t]>0):
                        break
                # for i in alphal:
                    # if all(self.lamda[t]>0):
                        # break
            # need to update pi 



            elif m.ng>0:
                S = np.diag(self.s[t])
                Lamda = np.diag(self.lamda[t])
                SinvLamda = np.dot(np.linalg.inv(S),Lamda)
                rs = self.tau[t]-np.dot(S,self.lamda[t])
                g = np.dot(d.Gu,utry[t]-us[t]) + np.dot(d.Gx,xtry[t]-xs[t]) + d.g + self.s[t] + np.dot(np.linalg.inv(Lamda),rs)
                del_Lamda = np.dot(SinvLamda,g)

                alphas = [2**(-n) for n in range(10)]
                alphal = [2**(-n) for n in range(10)]
                for i in alphas:
                    self.s[t] += i*(np.dot(np.linalg.inv(Lamda),rs) - np.dot(np.linalg.inv(SinvLamda),del_Lamda))
                    self.lamda[t] += i*(del_Lamda)
                    if all(self.s[t]>0) and all(self.lamda[t]>0):
                        break
                # for i in alphal:
                    # if all(self.lamda[t]>0):
                        # break

            raiseIfNan([ctry, cost], ArithmeticError('forward error'))
            raiseIfNan(xtry[t + 1], ArithmeticError('forward error'))
        with np.warnings.catch_warnings():
            np.warnings.simplefilter(warning)
            self.problem.terminalModel.calc(self.problem.terminalData, xtry[-1])
            ctry += self.problem.terminalData.cost
        raiseIfNan(ctry, ArithmeticError('forward error'))
        self.cost_try = ctry
        return xtry, utry, ctry

    def computeDirection(self, recalc=True):
        if recalc:
            self.calcDiff()
        self.backwardPass()
        return [np.nan] * (self.problem.T + 1), self.k, self.Vx

    def tryStep(self, stepLength=1):
        self.forwardPass(stepLength)
        return self.cost - self.cost_try


    def expectedImprovement(self):
        d1 = sum([np.dot(q.T, k) for q, k in zip(self.Qu, self.k)])
        d2 = sum([-np.dot(k.T, np.dot(q, k)) for q, k in zip(self.Quu, self.k)])
        return np.array([d1, d2])

    def stoppingCriteria(self):
        #return sum([np.dot(q.T, q) for q in self.Qu])
        return(self.d[0] + .5 * self.d[1])

    def calcDiff(self):
        if self.iter == 0:
            self.problem.calc(self.xs, self.us)
        self.cost = self.problem.calcDiff(self.xs, self.us)
        if not self.isFeasible:
            self.fs[0] = self.problem.runningModels[0].state.diff(self.xs[0], self.problem.x0)
            for i, (m, d, x) in enumerate(zip(self.problem.runningModels, self.problem.runningDatas, self.xs[1:])):
                self.fs[i + 1] = m.state.diff(x, d.xnext)
        return self.cost

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
        
        self.pi_K = [np.zeros([m.nu, m.state.ndx]) for m in self.problem.runningModels]
        self.pi_k = [np.zeros([m.nu]) for m in self.problem.runningModels]

        self.lamda = [np.ones([m.ng]) for m in self.problem.runningModels]
        self.pi = [np.ones([m.nh]) for m in self.problem.runningModels]
        self.s = [np.ones([m.ng]) for m in self.problem.runningModels]
        self.tau = [2*np.ones([m.ng]) for m in self.problem.runningModels]
        self.k_b =0.33
        self.th_1 = 1e-8
        self.th_2 = 1e-8

        self.xs_try = [self.problem.x0] + [np.nan * self.problem.x0] * self.problem.T
        self.us_try = [np.nan] * self.problem.T
        self.fs = [np.zeros(self.problem.runningModels[0].state.ndx)
                   ] + [np.zeros(m.state.ndx) for m in self.problem.runningModels]
