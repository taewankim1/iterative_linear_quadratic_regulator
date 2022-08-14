from __future__ import division
import matplotlib.pyplot as plt
import numpy as np
import time
import random
def print_np(x):
    print ("Type is %s" % (type(x)))
    print ("Shape is %s" % (x.shape,))
    # print ("Values are: \n%s" % (x))


from cost.cost import OptimalcontrolCost

class UnicycleCost(OptimalcontrolCost):
    def __init__(self,name,xf,N):
        super().__init__(name)
       
        self.ix = 3
        self.iu = 2
        self.xf = xf
        self.N = N

        # weight for cost
        self.R = 1 * np.identity(self.iu)

        # weight for constraint
        self.wf = 1e6
        self.w_input_constaint = 1e3
        
    def estimate_cost(self,x,u,final=False):
        # dimension
        ndim = np.ndim(x)
        if ndim == 1: # 1 step state & input
            N = 1
            x = np.expand_dims(x,axis=0)
            u = np.expand_dims(u,axis=0)
        else :
            N = np.size(x,axis = 0)

        cost = np.zeros(N)
        if final == False : 
            # cost for input
            u_mat = np.expand_dims(u,axis=2)
            R_mat = np.tile(self.R,(N,1,1))
            lu = np.squeeze( np.matmul(np.matmul(np.transpose(u_mat,(0,2,1)),R_mat),u_mat) )
            
            cost += (lu)

            # cost for input constaint
            v = u[:,0]
            flag_violation = (v - 2) > 0  
            v_violation = (v - 2) * flag_violation
            lv = self.w_input_constaint * (v_violation ** 2)
            # lv = self.w_input_constaint * np.sum(v_violation)
            cost += lv
            #     print("v",v )
                # print("v_vio",v_violation)
            #     print("lv",lv)

        else :
            x_diff = np.copy(x)
            x_diff[:,0] = x_diff[:,0] - self.xf[0]
            x_diff[:,1] = x_diff[:,1] - self.xf[1]
            x_diff[:,2] = x_diff[:,2] - self.xf[2]

            x_mat = np.expand_dims(x_diff,2)
            Q_mat = self.wf*np.tile(np.eye(self.ix),(N,1,1))
            lx = np.squeeze(np.matmul(np.matmul(np.transpose(x_mat,(0,2,1)),Q_mat),x_mat))
            
            cost += lx
        
        return cost
    