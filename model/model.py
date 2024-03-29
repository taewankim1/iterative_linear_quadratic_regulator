
# coding: utf-8

# In[ ]:

from __future__ import division
from tkinter import W
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import numpy as np
import time
import random
import IPython
def print_np(x):
    print ("Type is %s" % (type(x)))
    print ("Shape is %s" % (x.shape,))
    # print ("Values are: \n%s" % (x))


class OptimalcontrolModel(object) :
    def __init__(self,name,ix,iu,linearization) :
        self.name = name
        self.ix = ix
        self.iu = iu
        self.type_linearization = linearization

    def forward(self,x,u,idx=None):
        print("this is in parent class")
        pass

    def forward_Euler(self,x,u,delT) :
        f = self.forward(x,u)        
        return np.squeeze(x+f*delT)
    
    def forward_RK3(self,x,u,delT):
        k1 = delT * self.forward(x,u)        
        k2 = delT * self.forward(x + 0.5*k1,u)        
        k3 = delT * self.forward(x - k1 + 2*k2,u)        
        return np.squeeze(x + 1/6*(k1+4*k2+k3))

    def forward_RK4(self,x,u,delT):
        k1 = delT * self.forward(x,u)        
        k2 = delT * self.forward(x + 0.5*k1,u)        
        k3 = delT * self.forward(x + 0.5*k2,u)        
        k4 = delT * self.forward(x + k3,u)        
        return np.squeeze(x + 1/6*(k1+2*k2+2*k3+k4))

    def diff(self) :
        print("this is in parent class")
        pass

    def diff_numeric_central(self,x,u) :
        # state & input size
        ix = self.ix
        iu = self.iu
        
        ndim = np.ndim(x)
        if ndim == 1: # 1 step state & input
            N = 1
            x = np.expand_dims(x,axis=0)
            u = np.expand_dims(u,axis=0)
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

        # augmented = [x_aug x], [u, u_aug]
        x_aug_m = x_diag - eps_x * h
        x_aug_m = np.dstack((x_aug_m,np.tile(x_mat,(1,1,iu))))
        x_aug_m = np.reshape( np.transpose(x_aug_m,(0,2,1)), (N*(iu+ix),ix))

        u_aug_m = u_diag - eps_u * h
        u_aug_m = np.dstack((np.tile(u_mat,(1,1,ix)),u_aug_m))
        u_aug_m = np.reshape( np.transpose(u_aug_m,(0,2,1)), (N*(iu+ix),iu))

        # augmented = [x_aug x], [u, u_aug]
        x_aug_p = x_diag + eps_x * h
        x_aug_p = np.dstack((x_aug_p,np.tile(x_mat,(1,1,iu))))
        x_aug_p = np.reshape( np.transpose(x_aug_p,(0,2,1)), (N*(iu+ix),ix))

        u_aug_p = u_diag + eps_u * h
        u_aug_p = np.dstack((np.tile(u_mat,(1,1,ix)),u_aug_p))
        u_aug_p = np.reshape( np.transpose(u_aug_p,(0,2,1)), (N*(iu+ix),iu))

        # numerical difference
        f_change_m = self.forward(x_aug_m,u_aug_m,0)
        f_change_p = self.forward(x_aug_p,u_aug_p,0)
        f_change_m = np.reshape(f_change_m,(N,ix+iu,ix))
        f_change_p = np.reshape(f_change_p,(N,ix+iu,ix))
        f_diff = (f_change_p - f_change_m) / (2*h)
        f_diff = np.transpose(f_diff,[0,2,1])
        fx = f_diff[:,:,0:ix]
        fu = f_diff[:,:,ix:ix+iu]
        
        # return np.squeeze(fx), np.squeeze(fu)
        return fx,fu

    def diff_numeric(self,x,u) :
        # state & input size
        ix = self.ix
        iu = self.iu
        
        ndim = np.ndim(x)
        if ndim == 1: # 1 step state & input
            N = 1
            x = np.expand_dims(x,axis=0)
            u = np.expand_dims(u,axis=0)
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

        # augmented = [x_aug x], [u, u_aug]
        x_aug_m = x_diag - eps_x * 0
        x_aug_m = np.dstack((x_aug_m,np.tile(x_mat,(1,1,iu))))
        x_aug_m = np.reshape( np.transpose(x_aug_m,(0,2,1)), (N*(iu+ix),ix))

        u_aug_m = u_diag - eps_u * 0
        u_aug_m = np.dstack((np.tile(u_mat,(1,1,ix)),u_aug_m))
        u_aug_m = np.reshape( np.transpose(u_aug_m,(0,2,1)), (N*(iu+ix),iu))

        # augmented = [x_aug x], [u, u_aug]
        x_aug_p = x_diag + eps_x * h
        x_aug_p = np.dstack((x_aug_p,np.tile(x_mat,(1,1,iu))))
        x_aug_p = np.reshape( np.transpose(x_aug_p,(0,2,1)), (N*(iu+ix),ix))

        u_aug_p = u_diag + eps_u * h
        u_aug_p = np.dstack((np.tile(u_mat,(1,1,ix)),u_aug_p))
        u_aug_p = np.reshape( np.transpose(u_aug_p,(0,2,1)), (N*(iu+ix),iu))

        # numerical difference
        f_change_m = self.forward(x_aug_m,u_aug_m,0)
        f_change_p = self.forward(x_aug_p,u_aug_p,0)
        f_change_m = np.reshape(f_change_m,(N,ix+iu,ix))
        f_change_p = np.reshape(f_change_p,(N,ix+iu,ix))
        f_diff = (f_change_p - f_change_m) / (h)
        f_diff = np.transpose(f_diff,[0,2,1])
        fx = f_diff[:,:,0:ix]
        fu = f_diff[:,:,ix:ix+iu]
        
        return np.squeeze(fx), np.squeeze(fu)

    def diff_discrete_Euler(self,x,u,delT,tf=None) :
        ix = self.ix
        iu = self.iu

        ndim = np.ndim(x)
        if ndim == 1: # 1 step state & input
            N = 1
            x = np.expand_dims(x,axis=0)
            u = np.expand_dims(u,axis=0)
        else :
            N = np.size(x,axis = 0)

        if self.type_linearization == "numeric_central" :
            fx,fu = self.diff_numeric_central(x,u)
        elif self.type_linearization == "numeric_forward" :
            fx,fu = self.diff_numeric(x,u)
        elif self.type_linearization == "analytic" :
            fx,fu = self.diff(x,u)
        eye = np.eye(ix)
        eye_3d = np.tile(eye,(N,1,1))
        A = eye_3d + delT * fx
        B = delT * fu
        return A,B

    def diff_discrete_RK3(self,x,u,delT,tf=None) :
        ix = self.ix
        iu = self.iu

        ndim = np.ndim(x)
        if ndim == 1: # 1 step state & input
            N = 1
            x = np.expand_dims(x,axis=0)
            u = np.expand_dims(u,axis=0)
        else :
            N = np.size(x,axis = 0)

        eye = np.eye(ix)
        eye_3d = np.tile(eye,(N,1,1))
        k1 = delT * self.forward(x,u)        
        k2 = delT * self.forward(x + 0.5*k1,u)        
        k3 = delT * self.forward(x - k1 + 2*k2,u)        

        def get_fxfu(x,u) :
            if self.type_linearization == "numeric_central" :
                fx,fu = self.diff_numeric_central(x,u)
            elif self.type_linearization == "numeric_forward" :
                fx,fu = self.diff_numeric(x,u)
            elif self.type_linearization == "analytic" :
                fx,fu = self.diff(x,u)
            return fx,fu
        
        A1,B1 = get_fxfu(x,u)
        A2,B2 = get_fxfu(x+0.5*k1,u)
        A3,B3 = get_fxfu(x-k1+2*k2,u)

        dA1 = delT*A1
        dA2 = delT*A2@(eye_3d + 0.5*dA1)
        dA3 = delT*A3@(eye_3d - dA1 + 2*dA2)
        dB1 = delT*B1
        dB2 = delT*B2 + 0.5*delT*A2@dB1
        dB3 = delT*B3 + delT*A3@(2*dB2-dB1)

        A = eye_3d + 1/6*(dA1+4*dA2+dA3)
        B = 1/6 * (dB1+4*dB2+dB3)
        return A,B

    def diff_discrete_RK4(self,x,u,delT,tf=None) :
        ix = self.ix
        iu = self.iu

        ndim = np.ndim(x)
        if ndim == 1: # 1 step state & input
            N = 1
            x = np.expand_dims(x,axis=0)
            u = np.expand_dims(u,axis=0)
        else :
            N = np.size(x,axis = 0)

        eye = np.eye(ix)
        eye_3d = np.tile(eye,(N,1,1))
        k1 = delT * self.forward(x,u)        
        k2 = delT * self.forward(x + 0.5*k1,u)        
        k3 = delT * self.forward(x + 0.5*k2,u)        
        k4 = delT * self.forward(x + k3,u)        

        def get_fxfu(x,u) :
            if self.type_linearization == "numeric_central" :
                fx,fu = self.diff_numeric_central(x,u)
            elif self.type_linearization == "numeric_forward" :
                fx,fu = self.diff_numeric(x,u)
            elif self.type_linearization == "analytic" :
                fx,fu = self.diff(x,u)
            return fx,fu
        
        A1,B1 = get_fxfu(x,u)
        A2,B2 = get_fxfu(x+0.5*k1,u)
        A3,B3 = get_fxfu(x+0.5*k2,u)
        A4,B4 = get_fxfu(x+k3,u)

        dA1 = delT*A1
        dA2 = delT*A2@(eye_3d + 0.5*dA1)
        dA3 = delT*A3@(eye_3d + 0.5*dA2)
        dA4 = delT*A4@(eye_3d + dA3)
        dB1 = delT*B1
        dB2 = delT*B2 + 0.5*delT*A2@dB1
        dB3 = delT*B3 + 0.5*delT*A3@dB2
        dB4 = delT*B4 + delT*A4@dB3

        A = eye_3d + 1/6*(dA1+2*dA2+2*dA3+dA4)
        B = 1/6 * (dB1+2*dB2+2*dB3+dB4)
        return A,B

    def diff_discrete_zoh(self,x,u,delT,tf=None) :
        # delT = self.delT
        ix = self.ix
        iu = self.iu

        ndim = np.ndim(x)
        if ndim == 1: # 1 step state & input
            N = 1
            x = np.expand_dims(x,axis=0)
            u = np.expand_dims(u,axis=0)
        else :
            N = np.size(x,axis = 0)

        def dvdt(t,V,u,length) :
            assert len(u) == length
            V = V.reshape((length,ix + ix*ix + ix*iu + ix + ix)).transpose()
            x = V[:ix].transpose()
            Phi = V[ix:ix*ix + ix]
            Phi = Phi.transpose().reshape((length,ix,ix))
            Phi_inv = np.linalg.inv(Phi)
            f = self.forward(x,u)
            if self.type_linearization == "numeric_central" :
                A,B = self.diff_numeric_central(x,u)
            elif self.type_linearization == "numeric_forward" :
                A,B = self.diff_numeric(x,u)
            elif self.type_linearization == "analytic" :
                A,B = self.diff(x,u)
            # IPython.embed()
            dpdt = np.matmul(A,Phi).reshape((length,ix*ix)).transpose()
            dbdt = np.matmul(Phi_inv,B).reshape((length,ix*iu)).transpose()
            dsdt = np.squeeze(np.matmul(Phi_inv,np.expand_dims(f,2))).transpose() / tf
            dzdt = np.squeeze(np.matmul(Phi_inv,-np.matmul(A,np.expand_dims(x,2)) - np.matmul(B,np.expand_dims(u,2)))).transpose()
            dv = np.vstack((f.transpose(),dpdt,dbdt,dsdt,dzdt))
            return dv.flatten(order='F')
        
        A0 = np.eye(ix).flatten()
        B0 = np.zeros((ix*iu))
        s0 = np.zeros(ix)
        z0 = np.zeros(ix)
        V0 = np.array([np.hstack((x[i],A0,B0,s0,z0)) for i in range(N)]).transpose()
        V0_repeat = V0.flatten(order='F')

        sol = solve_ivp(dvdt,(0,delT),V0_repeat,args=(u,N),method='RK45',rtol=1e-6,atol=1e-10)
        # sol = solve_ivp(dvdt,(0,delT),V0_repeat,args=(u,N),method='RK45',max_step=1e-2,rtol=1e-6,atol=1e-10)
        # sol = solve_ivp(dvdt,(0,delT),V0_repeat,args=(u,N),method='RK45',max_step=1e-2)
        # IPython.embed()
        idx_state = slice(0,ix)
        idx_A = slice(ix,ix+ix*ix)
        idx_B = slice(ix+ix*ix,ix+ix*ix+ix*iu)
        idx_s = slice(ix+ix*ix+ix*iu,ix+ix*ix+ix*iu+ix)
        idx_z = slice(ix+ix*ix+ix*iu+ix,ix+ix*ix+ix*iu+ix+ix)
        sol = sol.y[:,-1].reshape((N,-1))
        xnew = np.zeros((N+1,ix))
        xnew[0] = x[0]
        xnew[1:] = sol[:,:ix]
        x_prop = sol[:,idx_state].reshape((-1,ix))
        A = sol[:,idx_A].reshape((-1,ix,ix))
        B = np.matmul(A,sol[:,idx_B].reshape((-1,ix,iu)))
        s = np.matmul(A,sol[:,idx_s].reshape((-1,ix,1))).squeeze()
        z = np.matmul(A,sol[:,idx_z].reshape((-1,ix,1))).squeeze()

        return A,B,s,z,x_prop
