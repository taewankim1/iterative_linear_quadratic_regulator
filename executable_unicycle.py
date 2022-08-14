import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import scipy.linalg
import time
import random
def print_np(x):
    print ("Type is %s" % (type(x)))
    print ("Shape is %s" % (x.shape,))
#     print ("Values are: \n%s" % (x))

import sys
# import model

from model.UnicycleModel import UnicycleModel
from cost.UnicycleCost import UnicycleCost
from iLQR import iLQR

if __name__ == "__main__" : 

    xf = np.zeros(3)
    xf[0] = 2.0
    xf[1] = 2.0
    xf[2] = 0

    ix = 3
    iu = 2
    N = 30
    tf = 3
    delT = tf/N
    myModel = UnicycleModel('Hello',ix,iu,linearization="numeric_central")
    myCost = UnicycleCost('Hello',xf,N)

    maxIter= 100

    x0 = np.zeros(3)
    x0[0] = -2.0
    x0[1] = -2.0
    x0[2] = 0

    u0 = np.random.rand(N,iu)
    i1 = iLQR('unicycle',delT,N,tf,maxIter,myModel,myCost,discretization="Euler")
    xbar, ubar, Quu_save, Quu_inv_save, L, l = i1.update(x0,u0)

    t_index = np.array(range(N+1))*delT

    plt.figure(figsize=(10,10))
    fS = 18
    plt.subplot(221)
    plt.plot(xbar[:,0], xbar[:,1],'-', linewidth=2.0)
    plt.plot(xf[0],xf[1],"o",label='goal')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.axis([-3, 3, -3, 3])
    plt.xlabel('X (m)', fontsize = fS)
    plt.ylabel('Y (m)', fontsize = fS)
    plt.subplot(222)
    plt.plot(t_index, xbar[:,0], linewidth=2.0,label='naive')
    plt.xlabel('time (s)', fontsize = fS)
    plt.ylabel('x1 (m)', fontsize = fS)
    plt.subplot(223)
    plt.plot(t_index, xbar[:,1], linewidth=2.0,label='naive')
    plt.xlabel('time (s)', fontsize = fS)
    plt.ylabel('x2 (m)', fontsize = fS)
    plt.subplot(224)
    plt.plot(t_index, xbar[:,2], linewidth=2.0,label='naive')
    plt.xlabel('time (s)', fontsize = fS)
    plt.ylabel('x3 (rad)', fontsize = fS)
    plt.legend(fontsize=fS)
    plt.show()

    plt.figure()
    plt.subplot(121)
    plt.step(t_index, [*ubar[:N,0],ubar[N-1,0]],alpha=1.0,where='post',linewidth=2.0)
    plt.xlabel('time (s)', fontsize = fS)
    plt.ylabel('v (m/s)', fontsize = fS)
    plt.subplot(122)
    plt.step(t_index, [*ubar[:N,1],ubar[N-1,1]],alpha=1.0,where='post',linewidth=2.0)
    plt.xlabel('time (s)', fontsize = fS)
    plt.ylabel('w (rad/s)', fontsize = fS)
    plt.show()
