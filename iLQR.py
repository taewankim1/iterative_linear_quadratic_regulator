
# coding: utf-8

# In[4]:
from __future__ import division
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import scipy.linalg
import time
import random
def print_np(x):
    print ("Type is %s" % (type(x)))
    print ("Shape is %s" % (x.shape,))
    # print ("Values are: \n%s" % (x))


# In[5]:
import cost
import model


# In[7]:

class iLQR:
    def __init__(self,name,horizon,maxIter,Model,Cost):
        self.name = name
        self.model = Model
        self.cost = Cost
        self.N = horizon
        
        # cost optimization
        self.verbosity = True
        self.dlamda = 1.0
        self.lamda = 1.0
        self.lamdaFactor = 1.6
        self.lamdaMax = 1e10
        self.lamdaMin = 1e-6
        self.tolFun = 1e-7
        self.tolGrad = 1e-4
        self.maxIter = maxIter
        self.zMin = 0
        self.last_head = True
        
        self.initialize()
        
    def initialize(self) :
        
        self.dV = np.zeros((1,2))
        self.x0 = np.zeros(self.model.ix)
        self.x = np.zeros((self.N+1,self.model.ix))
        self.u = np.ones((self.N,self.model.iu))
        self.xnew = np.zeros((self.N+1,self.model.ix))
        self.unew = np.zeros((self.N,self.model.iu))
        self.Alpha = np.power(10,np.linspace(0,-3,11))
        self.l = np.zeros((self.N,self.model.iu))
        self.L = np.zeros((self.N,self.model.iu,self.model.ix))
        self.fx = np.zeros((self.N,self.model.ix,self.model.ix))
        self.fu = np.zeros((self.N,self.model.ix,self.model.iu))  
        self.c = np.zeros(self.N+1)
        self.cnew = np.zeros(self.N+1)
        self.cx = np.zeros((self.N+1,self.model.ix))
        self.cu = np.zeros((self.N,self.model.iu))
        self.cxx = np.zeros((self.N+1,self.model.ix,self.model.ix))
        self.cxu = np.zeros((self.N,self.model.ix,self.model.iu))
        self.cuu = np.zeros((self.N,self.model.iu,self.model.iu))
        self.Vx = np.zeros((self.N+1,self.model.ix))
        self.Vxx = np.zeros((self.N+1,self.model.ix,self.model.ix))
    
    def forward(self,x0,u,K,x,k,alpha):
        # TODO - change integral method to odefun
        # horizon
        N = self.N
        
        # x-difference
        dx = np.zeros(self.model.ix)
        
        # variable setting
        xnew = np.zeros((N+1,self.model.ix))
        unew = np.zeros((N,self.model.iu))
        cnew = np.zeros(N+1)
        
        # initial state
        xnew[0,:] = x0
        
        # roll-out
        for i in range(N):
            dx = xnew[i,:] - x[i,:]
            unew[i,:] = u[i,:] + k[i,:] * alpha + np.dot(K[i,:,:],dx)
            xnew[i+1,:] = self.model.forward(xnew[i,:],unew[i,:],i)
            cnew[i] = self.cost.estimate_cost(xnew[i,:],unew[i,:])
            
        cnew[N] = self.cost.estimate_cost(xnew[N,:],np.zeros(self.model.iu))
        return xnew,unew,cnew
        
    def backward(self):
        diverge = False
        
        # state & input size
        ix = self.model.ix
        iu = self.model.iu
        
        # V final value
        self.Vx[self.N,:] = self.cx[self.N,:]
        self.Vxx[self.N,:,:] = self.cxx[self.N,:,:]
        
        # Q function
        Qu = np.zeros(iu)
        Qx = np.zeros(ix)
        Qux = np.zeros([iu,ix])
        Quu = np.zeros([iu,iu])
        Quu_save = np.zeros([self.N,iu,iu]) # for saving
        Quu_inv_save = np.zeros([self.N,iu,iu])
        Qxx = np.zeros([ix,ix])
        
        Vxx_reg = np.zeros([ix,ix])
        Qux_reg = np.zeros([ix,iu])
        QuuF = np.zeros([iu,iu])
        
        # open-loop gain, feedback gain
        k_i = np.zeros(iu)
        K_i = np.zeros([iu,ix])
        
        self.dV[0,0] = 0.0
        self.dV[0,1] = 0.0
        
        diverge_test = 0
        for i in range(self.N-1,-1,-1):
            # print(i)
            Qu = self.cu[i,:] + np.dot(self.fu[i,:].T, self.Vx[i+1,:])
            Qx = self.cx[i,:] + np.dot(self.fx[i,:].T, self.Vx[i+1,:])
 
            Qux = self.cxu[i,:,:].T + np.dot( np.dot(self.fu[i,:,:].T, self.Vxx[i+1,:,:]),self.fx[i,:,:])
            Quu = self.cuu[i,:,:] + np.dot( np.dot(self.fu[i,:,:].T, self.Vxx[i+1,:,:]),self.fu[i,:,:])
            Qxx = self.cxx[i,:,:] + np.dot( np.dot(self.fx[i,:,:].T, self.Vxx[i+1,:,:]),self.fx[i,:,:])
            
            Vxx_reg = self.Vxx[i+1,:,:] + self.lamda * np.identity(ix)
            Qux_reg = self.cxu[i,:,:].T + np.dot(np.dot(self.fu[i,:,:].T, Vxx_reg), self.fx[i,:,:])
            QuuF = self.cuu[i,:,:] + np.dot(np.dot(self.fu[i,:,:].T, Vxx_reg), self.fu[i,:,:]) + 0*self.lamda * np.identity(iu)
            Quu_save[i,:,:] = QuuF
            # TODO : put input constraints
        
            
            # control gain      
            try:
                R = sp.linalg.cholesky(QuuF,lower=False)
            except sp.linalg.LinAlgError as err:
                diverge_test = i+1
                return diverge_test, Quu_save, Quu_inv_save
                        
            R_inv = sp.linalg.inv(R)
            QuuF_inv = np.dot(R_inv,np.transpose(R_inv))
            # Quu_inv_save[i,:,:] = np.linalg.inv(Quu)
            Quu_inv_save[i,:,:] = QuuF_inv
            k_i = - np.dot(QuuF_inv, Qu)
            K_i = - np.dot(QuuF_inv, Qux_reg)
            # print k_i, K_i
            
            # update cost-to-go approximation
            self.dV[0,0] = np.dot(k_i.T, Qu) + self.dV[0,0]
            self.dV[0,1] = 0.5*np.dot( np.dot(k_i.T, Quu), k_i) + self.dV[0,1]
            self.Vx[i,:] = Qx + np.dot(np.dot(K_i.T,Quu),k_i) + np.dot(K_i.T,Qu) + np.dot(Qux.T,k_i)
            self.Vxx[i,:,:] = Qxx + np.dot(np.dot(K_i.T,Quu),K_i) + np.dot(K_i.T,Qux) + np.dot(Qux.T,K_i)
            self.Vxx[i,:,:] = 0.5 * ( self.Vxx[i,:,:].T + self.Vxx[i,:,:] )
                                                                                               
            # save the control gains
            self.l[i,:] = k_i
            self.L[i,:,:] = K_i
            
        return diverge_test, Quu_save, Quu_inv_save
                   
        
    def update(self,x0,u0):
        # current position
        self.x0 = x0
        
        # initial input
        self.u = u0
        
        # state & input & horizon size
        ix = self.model.ix
        iu = self.model.iu
        N = self.N
        
        # timer setting
        # trace for iteration
        # timer, counters, constraints
        # timer begin!!
        
        # generate initial trajectory
        diverge = False
        stop = False

        self.x[0,:] = self.x0
        for j in range(np.size(self.Alpha,axis=0)):   
            for i in range(self.N):
                self.x[i+1,:] = self.model.forward(self.x[i,:],self.Alpha[j]*self.u[i,:],i)       
                self.c[i] = self.cost.estimate_cost(self.x[i,:],self.Alpha[j]*self.u[i,:])
                if  np.max( self.x[i+1,:] ) > 1e8 :                
                    diverge = True
                    print("initial trajectory is already diverge")
                    pass
            self.c[self.N] = self.cost.estimate_cost(self.x[self.N,:],np.zeros(self.model.iu))
            if diverge == False:
                break
                pass
            pass
        # iterations starts!!
        flgChange = True
        for iteration in range(self.maxIter) :
            # differentiate dynamics and cost
            if flgChange == True:
                start = time.time()
                self.fx, self.fu = self.model.diff(self.x[0:N,:],self.u)
                c_x_u = self.cost.diff_cost(self.x[0:N,:],self.u)
                c_xx_uu = self.cost.hess_cost(self.x[0:N,:],self.u)
                c_xx_uu = 0.5 * ( np.transpose(c_xx_uu,(0,2,1)) + c_xx_uu )
                self.cx[0:N,:] = c_x_u[:,0:self.model.ix]
                self.cu[0:N,:] = c_x_u[:,self.model.ix:self.model.ix+self.model.iu]
                self.cxx[0:N,:,:] = c_xx_uu[:,0:ix,0:ix]
                self.cxu[0:N,:,:] = c_xx_uu[:,0:ix,ix:(ix+iu)]
                self.cuu[0:N,:,:] = c_xx_uu[:,ix:(ix+iu),ix:(ix+iu)]
                c_x_u = self.cost.diff_cost(self.x[N:,:],np.zeros((1,iu)))
                c_xx_uu = self.cost.hess_cost(self.x[N:,:],np.zeros((1,iu)))
                c_xx_uu = 0.5 * ( c_xx_uu + c_xx_uu.T)
                self.cx[N,:] = c_x_u[0:self.model.ix]
                self.cxx[N,:,:] = c_xx_uu[0:ix,0:ix]
                flgChange = False
                pass

            time_derivs = (time.time() - start)

            # backward pass
            backPassDone = False
            while backPassDone == False:
                start =time.time()
                diverge,Quu_save,Quu_inv_save = self.backward()
                time_backward = time.time() - start
                if diverge != 0 :
                    if self.verbosity == True:
                        print("Cholesky failed at %s" % (diverge))
                        pass
                    self.dlamda = np.maximum(self.dlamda * self.lamdaFactor, self.lamdaFactor)
                    self.lamda = np.maximum(self.lamda * self.dlamda,self.lamdaMin)
                    if self.lamda > self.lamdaMax :
                        break
                        pass
                    continue
                    pass
                backPassDone = True
            # check for termination due to small gradient
            g_norm = np.mean( np.max( np.abs(self.l) / (np.abs(self.u) + 1), axis=1 ) )
            if g_norm < self.tolGrad and self.lamda < 1e-5 :
                self.dlamda = np.minimum(self.dlamda / self.lamdaFactor, 1/self.lamdaFactor)
                if self.lamda > self.lamdaMin :
                    temp_c = 1
                    pass
                else :
                    temp_c = 0
                    pass       
                self.lamda = self.lamda * self.dlamda * temp_c 
                if self.verbosity == True:
                    print("SUCCEESS : gradient norm < tolGrad")
                    pass
                break
                pass
            # step3. line-search to find new control sequence, trajectory, cost
            fwdPassDone = False
            if backPassDone == True :
                start = time.time()
                for i in self.Alpha :
                    self.xnew,self.unew,self.cnew = self.forward(self.x0,self.u,self.L,self.x,self.l,i)
                    # print np.sum(self.unew-self.u)
                    dcost = np.sum( self.c ) - np.sum( self.cnew )
                    expected = -i * (self.dV[0,0] + i * self.dV[0,1])
                    if expected > 0. :
                        z = dcost / expected
                    else :
                        z = np.sign(dcost)
                        print("non-positive expected reduction: should not occur")
                        pass
                    # print(i)
                    if z > self.zMin :
                        fwdPassDone = True
                        break          
                if fwdPassDone == False :
                    alpha_temp = 1e8 # % signals failure of forward pass
                    pass
                time_forward = time.time() - start
            else :
                dcost = 0
                expected = 0
                
            # step4. accept step, draw graphics, print status 
            if self.verbosity == True and self.last_head == True:
                self.last_head = False
                print("iteration   cost        reduction   expected    gradient    log10(lambda)")
                pass

            if fwdPassDone == True:
                if self.verbosity == True:
                    print("%-12d%-12.6g%-12.3g%-12.3g%-12.3g%-12.1f" % ( iteration,np.sum(self.c),dcost,expected,g_norm,np.log10(self.lamda)) )     
                    pass

                # decrese lamda
                self.dlamda = np.minimum(self.dlamda / self.lamdaFactor, 1/self.lamdaFactor)
                if self.lamda > self.lamdaMin :
                    temp_c = 1
                    pass
                else :
                    temp_c = 0
                    pass
                self.lamda = self.lamda * self.dlamda * temp_c 

                # accept changes
                self.u = self.unew
                self.x = self.xnew
                self.c = self.cnew
                # self.Mu = self.Munew
                flgChange = True
                # print(time_derivs)
                # print(time_backward)
                # print(time_forward)
                # abc
                # terminate?
                if dcost < self.tolFun :
                    if self.verbosity == True:
                        print("SUCCEESS: cost change < tolFun",dcost)
                        pass
                    break
                    pass

            else : # no cost improvement
                # increase lamda
                # ssprint(iteration)
                self.dlamda = np.maximum(self.dlamda * self.lamdaFactor, self.lamdaFactor)
                self.lamda = np.maximum(self.lamda * self.dlamda,self.lamdaMin)

                # print status
                if self.verbosity == True :
                    print("%-12d%-12s%-12.3g%-12.3g%-12.3g%-12.1f" %
                        ( iteration,'NO STEP', dcost, expected, g_norm, np.log10(self.lamda) ));
                    pass

                if self.lamda > self.lamdaMax :
                    if self.verbosity == True:
                        print("EXIT : lamda > lamdaMax")
                        pass
                    break
                    pass
                pass
            pass


        return self.x, self.u, Quu_save, Quu_inv_save, self.L, self.l
        


        
        
        
        
        
        
        
        
        
        
        
        


