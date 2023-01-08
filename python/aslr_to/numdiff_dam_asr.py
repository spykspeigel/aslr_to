import numpy as np
import pinocchio
import crocoddyl


class NumDiffASRFwdDynamicsModel(crocoddyl.DifferentialActionModelAbstract):
    def __init__(self, model, disturbance):
        # try:
        #     if model.actuation is not None:
        #         crocoddyl.DifferentialActionModelAbstract.__init__(self, model.state, model.actuation.nu, model.costs.nr)
        # except:
        crocoddyl.DifferentialActionModelAbstract.__init__(self, model.state, model.nu, model.nr)
        self.model = model
        self.disturbance = disturbance
        # if K is None:
        #     self.K = 1e-1*np.eye(int(state.nv/2))
        # else:
        #     self.K = K
        # if B is None:
        #     self.B = 1e-3*np.eye(int(state.nv/2))
        # else:
        #     self.B = B

    def calc(self, data, x, u=None):

        self.model.calc(data.data_0,  x, u)
        # Computing the motor side dynamics
        try:
            data.cost = data.data_0.costs.cost
        except:
            data.cost = data.data_0.cost

        data.xout = data.data_0.xout

    def calcDiff(self, data, x, u=None):
        
        xn0 = data.data_0.xout
        c0 = data.data_0.cost

        #Computing the d action(x,u) / dx
        disturbance = self.disturbance
        dx = np.zeros(self.state.ndx)   # can be included in the initialisation of the data 

        for i in range(self.state.ndx):
            
            dx[i] = disturbance
            x_p = self.model.state.integrate(x,dx) # x = x + dx
            # x_p = x+dx
            data.data_nd_x = self.model.createData()
            self.model.calc(data.data_nd_x,  x_p, u)
            xp = data.data_nd_x.xout
            cp = data.data_nd_x.cost
            data.Fx[:,i] = (xp - xn0) / disturbance
            
            # print(data.data_nd_x[i].r)
            # data.Rx[:,i] = (data.data_nd_x[i].r - data.data_0.r)/disturbance
            data.Lx[i] = (cp - c0) / disturbance

            # x_n = x-dx
            x_n = self.model.state.integrate(x,-dx) # x = x + dx
            data.data_nd_x = self.model.createData()
            self.model.calc(data.data_nd_x,  x_n, u)
            cn = data.data_nd_x.cost
            data.Lxx[i,i]= (cp-2*c0+cn)/(disturbance**2)

            for j in range(i+1, self.state.ndx):
                dx[j] = disturbance
                x_pp = self.model.state.integrate(x,dx) # x = x - dx
                # x_pp = x + dx
                data.data_nd_x = self.model.createData()
                self.model.calc(data.data_nd_x,  x_pp, u)
                c_pp = data.data_nd_x.cost  
                
                # dx[i] = 0

                # x_pp = self.model.state.integrate(x,dx) # x = x + dx_i +dx_j
                # # x_pp = x + dx
                # data.data_nd_x = self.model.createData()
                # self.model.calc(data.data_nd_x,  x_pp, u)
                # c_np = data.data_nd_x.cost
                
                dx[i] = -disturbance
                dx[j] = disturbance
                x_np = self.model.state.integrate(x,dx) # x = x - dx_i +dx_j
                # x_np = x + dx
                data.data_nd_x = self.model.createData()
                self.model.calc(data.data_nd_x,  x_np, u)
                c_np = data.data_nd_x.cost

                dx[i] = disturbance
                dx[j] = -disturbance
                x_pn = self.model.state.integrate(x,dx) # x = x + dx_i  -dx_j
                # x_pn = x+dx
                data.data_nd_x = self.model.createData()
                self.model.calc(data.data_nd_x,  x_pn, u)
                c_pn = data.data_nd_x.cost

                dx[i] = -disturbance
                dx[j] = -disturbance
                x_nn = self.model.state.integrate(x,dx) # x = x - dx_i -dx_j
                data.data_nd_x = self.model.createData()
                self.model.calc(data.data_nd_x,  x_nn, u)
                c_nn = data.data_nd_x.cost
                data.Lxx[i,j] = (c_pp - c_np - c_pn + c_nn)/(4* disturbance**2)
                # data.Lxx[i,j] = (c_pp - cp- c_np +c0)/(disturbance**2)
            
                data.Lxx[j,i] = data.Lxx[i,j]
                dx[j] = 0.
                dx[i] = disturbance

            dx = np.zeros(self.state.ndx)



        # Computing the d action(x,u) / du
        disturbance = self.disturbance
        du = np.zeros(self.model.nu)  # can be included in the initialisation of the data 

        for i in range(self.model.nu):
            du[i] = disturbance
            data.data_nd_u = self.model.createData()
            self.model.calc(data.data_nd_u, x, u + du)

            xn = data.data_nd_u.xout
            cp = data.data_nd_u.cost
            data.Fu[:,i] = (xn - xn0) / disturbance

            data.Lu[i] = (cp - c0) / disturbance
            # data.Ru[:,i] = (data.data_nd_u[i].r - data.data_0.r) / disturbance


            data.data_nd_u = self.model.createData()
            self.model.calc(data.data_nd_u,  x, u-du)
            cn = data.data_nd_u.cost
            data.Luu[i,i]= (cp-2*c0+cn)/(disturbance**2)


            for j in range(i+1, self.model.nu):
                du[j] = disturbance
                data.data_nd_u = self.model.createData()
                self.model.calc(data.data_nd_u,  x, u+du)
                c_pp = data.data_nd_u.cost

                
                
                du[i] = -disturbance
                du[j] = disturbance
                data.data_nd_u = self.model.createData()
                self.model.calc(data.data_nd_u,  x, u+du)
                c_np = data.data_nd_u.cost

                du[i] = disturbance
                du[j] = -disturbance
                data.data_nd_u = self.model.createData()
                self.model.calc(data.data_nd_u,  x, u+du)
                c_pn = data.data_nd_u.cost

                du[i] = -disturbance
                du[j] = -disturbance
                data.data_nd_u = self.model.createData()
                self.model.calc(data.data_nd_u,  x, u+du)
                c_nn = data.data_nd_u.cost
                data.Luu[i,j] = (c_pp - c_np - c_pn + c_nn)/(4* disturbance**2)
                data.Luu[j,i] = data.Luu[i,j]
                du[j] = 0.
                du[i] = disturbance


            du = np.zeros(self.model.nu)




        # data.Lxx = data.data_0.Lxx
        # data.Lxu = data.data_0.Lxu
        # data.Luu = data.data_0.Luu

        # data.Lxx = np.dot(data.Rx.T, data.Rx)
        # data.Lxu = np.dot(data.Rx.T, data.Ru)
        # data.Luu = np.dot(data.Ru.T, data.Ru)
        
    def createData(self):
        data = NumDiffASRFwdDynamicsData(self)
        return data

class NumDiffASRFwdDynamicsData(crocoddyl.DifferentialActionDataAbstract):
    def __init__(self, model):
        crocoddyl.DifferentialActionDataAbstract.__init__(self, model)
        try:
            self.pinocchio = pinocchio.Model.createData(model.model.state.pinocchio)
            self.multibody = crocoddyl.DataCollectorMultibody(self.pinocchio)
            self.actuation = model.model.actuation.createData()
            self.costs = model.model.costs.createData(self.multibody)
            self.costs.shareMemory(self)
        except:
            pass
        self.Minv = None
        self.Binv = None
        self.Lxx = np.zeros([model.model.state.ndx,model.model.state.ndx])
        self.data_0 = model.model.createData()
        # self.data_nd_x = [model.model.createData() for i in range(model.model.state.ndx)] 
        
        # self.data_nd_u = [model.model.createData() for i in range(model.model.nu)]
        self.Rx = np.zeros([model.model.nr, model.model.state.ndx])
        self.Ru = np.zeros([model.model.nr, model.model.nu])