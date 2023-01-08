import os
import sys

#This unittest is for easier reference

import crocoddyl
import pinocchio
import numpy as np
import aslr_to
import example_robot_data
import time
import statistics

# two_dof = example_robot_data.load('asr_twodof')
two_dof = example_robot_data.load('talos_arm')
robot_model = two_dof.model
state = aslr_to.StateMultibodyASR(robot_model)
actuation = aslr_to.ASRActuation(state)
nu = actuation.nu 
costs = crocoddyl.CostModelSum(state, nu)
t=time.time()
model = aslr_to.DifferentialFreeASRFwdDynamicsModel(state, actuation, costs)
numerror = []
anaTime=[]
numTime = []
max_der = 0
for i in range(50):
    x =   model.state.rand()
    u = np.zeros(model.nu)
    data =  model.createData()
    MODEL_ND = crocoddyl.DifferentialActionModelNumDiff(model)
    MODEL_ND.disturbance *= 100
    DATA_ND = MODEL_ND.createData()
    model.calc(data, x, u)
    t1_an=time.time()
    model.calcDiff(data, x, u)
    t2_an=time.time()
    MODEL_ND.calc(DATA_ND, x, u)
    t1_num = time.time()
    MODEL_ND.calcDiff(DATA_ND, x, u)
    t2_num = time.time()

    numerror.append(np.linalg.norm(data.Fx -DATA_ND.Fx))
    anaTime.append( t1_an-t2_an)
    numTime.append(t1_num-t2_num)
    if np.linalg.norm(max_der) <np.linalg.norm(data.Fx):
        max_der = data.Fx
avg_error = sum(numerror)/50
std_numError = statistics.pstdev(numerror)

avg_anaTime = sum(anaTime)/50
std_anaTime = statistics.pstdev(anaTime)

avg_numTime = sum(numTime)/50
std_numTime = statistics.pstdev(numTime)
print("_____________--")
print(avg_error)
print(std_numError)
print("_____________--")
print(avg_anaTime)
print(std_anaTime)
print("_____________--")
print(avg_numTime)
print(std_numTime)

print("max_der")
print(np.linalg.norm(max_der))