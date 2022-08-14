
# coding: utf-8

# In[1]:

from __future__ import division
import matplotlib.pyplot as plt
import numpy as np
import time
import random
def print_np(x):
    print ("Type is %s" % (type(x)))
    print ("Shape is %s" % (x.shape,))
    # print ("Values are: \n%s" % (x))
class OptimalcontrolCost(object) :
    def __init__(self,name) :
        self.name = name

    def estimate_cost(self) :
        print("this is in parent class")
        pass

    def diff_cost(self,x,u,final):
        
        # state & input size
        ix = self.ix
        iu = self.iu
        
        ndim = np.ndim(x)
        if ndim == 1: # 1 step state & input
            N = 1
        else :
            N = np.size(x,axis = 0)

        # numerical difference
        h = pow(2,-17)
        eps_x = np.identity(ix)
        eps_u = np.identity(iu)

        # expand to tensor
        x_mat = np.expand_dims(x,axis=2)
        u_mat = np.expand_dims(u,axis=2)

        # diag
        x_diag = np.tile(x_mat,(1,1,ix))
        u_diag = np.tile(u_mat,(1,1,iu))

        # augmented = [x_diag x], [u, u_diag]
        x_aug = x_diag + eps_x * h
        x_aug = np.dstack((x_aug,np.tile(x_mat,(1,1,iu))))
        x_aug = np.reshape( np.transpose(x_aug,(0,2,1)), (N*(iu+ix),ix))
        
        u_aug = u_diag + eps_u * h
        u_aug = np.dstack((np.tile(u_mat,(1,1,ix)),u_aug))
        u_aug = np.reshape( np.transpose(u_aug,(0,2,1)), (N*(iu+ix),iu))

        # numerical difference
        c_nominal = self.estimate_cost(x,u,final)
        c_change = self.estimate_cost(x_aug,u_aug,final)
        c_change = np.reshape(c_change,(N,1,iu+ix))

        c_diff = ( c_change - np.reshape(c_nominal,(N,1,1)) ) / h
        c_diff = np.reshape(c_diff,(N,iu+ix))
            
        return  np.squeeze(c_diff)


    def diff_cost_central(self,x,u,final):
        # state & input size
        ix = self.ix
        iu = self.iu
        
        ndim = np.ndim(x)
        if ndim == 1: # 1 step state & input
            N = 1

        else :
            N = np.size(x,axis = 0)

        # numerical difference
        h = pow(2,-17) / 2
        eps_x = np.identity(ix)
        eps_u = np.identity(iu)

        # expand to tensor
        x_mat = np.expand_dims(x,axis=2)
        u_mat = np.expand_dims(u,axis=2)

        # diag
        x_diag = np.tile(x_mat,(1,1,ix))
        u_diag = np.tile(u_mat,(1,1,iu))

        # augmented = [x_diag x], [u, u_diag]
        x_aug_m = x_diag - eps_x * h
        x_aug_m = np.dstack((x_aug_m,np.tile(x_mat,(1,1,iu))))
        x_aug_m = np.reshape( np.transpose(x_aug_m,(0,2,1)), (N*(iu+ix),ix))
        
        u_aug_m = u_diag - eps_u * h
        u_aug_m = np.dstack((np.tile(u_mat,(1,1,ix)),u_aug_m))
        u_aug_m = np.reshape( np.transpose(u_aug_m,(0,2,1)), (N*(iu+ix),iu))

        x_aug_p = x_diag + eps_x * h
        x_aug_p = np.dstack((x_aug_p,np.tile(x_mat,(1,1,iu))))
        x_aug_p = np.reshape( np.transpose(x_aug_p,(0,2,1)), (N*(iu+ix),ix))
        
        u_aug_p = u_diag + eps_u * h
        u_aug_p = np.dstack((np.tile(u_mat,(1,1,ix)),u_aug_p))
        u_aug_p = np.reshape( np.transpose(u_aug_p,(0,2,1)), (N*(iu+ix),iu))

        # numerical difference
        c_change_m = self.estimate_cost(x_aug_m,u_aug_m,final)
        c_change_p = self.estimate_cost(x_aug_p,u_aug_p,final)
        c_change_m = np.reshape(c_change_m,(N,1,iu+ix))
        c_change_p = np.reshape(c_change_p,(N,1,iu+ix))

        c_diff = ( c_change_p - c_change_m ) / (2*h)
        c_diff = np.reshape(c_diff,(N,iu+ix))
            
        return  np.squeeze(c_diff)
    
    def hess_cost(self,x,u,final):
        
        # state & input size
        ix = self.ix
        iu = self.iu
        
        ndim = np.ndim(x)
        if ndim == 1: # 1 step state & input
            N = 1
        else :
            N = np.size(x,axis = 0)
        
        # numerical difference
        h = pow(2,-17)
        eps_x = np.identity(ix)
        eps_u = np.identity(iu)

        # expand to tensor
        x_mat = np.expand_dims(x,axis=2)
        u_mat = np.expand_dims(u,axis=2)

        # diag
        x_diag = np.tile(x_mat,(1,1,ix))
        u_diag = np.tile(u_mat,(1,1,iu))

        # augmented = [x_diag x], [u, u_diag]
        x_aug = x_diag + eps_x * h
        x_aug = np.dstack((x_aug,np.tile(x_mat,(1,1,iu))))
        x_aug = np.reshape( np.transpose(x_aug,(0,2,1)), (N*(iu+ix),ix))

        u_aug = u_diag + eps_u * h
        u_aug = np.dstack((np.tile(u_mat,(1,1,ix)),u_aug))
        u_aug = np.reshape( np.transpose(u_aug,(0,2,1)), (N*(iu+ix),iu))


        # numerical difference
        c_nominal = self.diff_cost(x,u,final)
        c_change = self.diff_cost(x_aug,u_aug,final)
        c_change = np.reshape(c_change,(N,iu+ix,iu+ix))
        c_hess = ( c_change - np.reshape(c_nominal,(N,1,ix+iu)) ) / h
        c_hess = np.reshape(c_hess,(N,iu+ix,iu+ix))
         
        return np.squeeze(c_hess)


    def hess_cost_central(self,x,u,final):
        
        # state & input size
        ix = self.ix
        iu = self.iu
        
        ndim = np.ndim(x)
        if ndim == 1: # 1 step state & input
            N = 1
        else :
            N = np.size(x,axis = 0)
        
        # numerical difference
        h = pow(2,-17)
        eps_x = np.identity(ix)
        eps_u = np.identity(iu)

        # expand to tensor
        x_mat = np.expand_dims(x,axis=2)
        u_mat = np.expand_dims(u,axis=2)

        # diag
        x_diag = np.tile(x_mat,(1,1,ix))
        u_diag = np.tile(u_mat,(1,1,iu))

        # augmented = [x_diag x], [u, u_diag]
        x_aug_m = x_diag - eps_x * h
        x_aug_m = np.dstack((x_aug_m,np.tile(x_mat,(1,1,iu))))
        x_aug_m = np.reshape(np.transpose(x_aug_m,(0,2,1)), (N*(iu+ix),ix))

        u_aug_m = u_diag - eps_u * h
        u_aug_m = np.dstack((np.tile(u_mat,(1,1,ix)),u_aug_m))
        u_aug_m = np.reshape( np.transpose(u_aug_m,(0,2,1)), (N*(iu+ix),iu))

        x_aug_p = x_diag + eps_x * h
        x_aug_p = np.dstack((x_aug_p,np.tile(x_mat,(1,1,iu))))
        x_aug_p = np.reshape( np.transpose(x_aug_p,(0,2,1)), (N*(iu+ix),ix))

        u_aug_p = u_diag + eps_u * h
        u_aug_p = np.dstack((np.tile(u_mat,(1,1,ix)),u_aug_p))
        u_aug_p = np.reshape( np.transpose(u_aug_p,(0,2,1)), (N*(iu+ix),iu))


        # numerical difference
        c_change_m = self.diff_cost_central(x_aug_m,u_aug_m,final)
        c_change_p = self.diff_cost_central(x_aug_p,u_aug_p,final)
        c_change_m = np.reshape(c_change_m,(N,iu+ix,iu+ix))
        c_change_p = np.reshape(c_change_p,(N,iu+ix,iu+ix))
        c_hess = ( c_change_p - c_change_m ) / h
        c_hess = np.reshape(c_hess,(N,iu+ix,iu+ix))
         
        return np.squeeze(c_hess)



    
