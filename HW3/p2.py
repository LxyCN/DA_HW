import numpy as np 
from numpy import matmul as mt 
from numpy.linalg import inv
import matplotlib.pyplot as plt
from math import sin,cos

q1 = 0.2
q2 = 0.1
irr = 50
r = 1   #observe
A = np.array([[1, 0.01, 0],
              [0, 1, 0],
              [0, 0, 1]])
Q = np.array([[0.01**3/3*q1, 0.5*0.01**2*q1, 0],
              [0.5*0.01**2*q1, 0.01*q1, 0],
              [0, 0, 0.01*q2]])

q = np.random.multivariate_normal(np.zeros(3), Q, size=irr-1).T
#observe model
def h(x):
    return mt(x[2,:].T, sin(x[0,:])) + np.random.normal(r)

x_T = np.zeros([3,irr])
x_T[:,0] = np.array([0,10,1]).T
for i in range(1, irr):
    x_T[:,i] = mt(A, x_T[:,i-1].T).flatten() + q[:,i-1]

y_ob = h(x_T)
x = np.zeros([3,irr])
p = np.diag([3,3,3])

for i in range(1,irr):
    p = Q + mt(mt(A, p), A.T)
    x[:,i] =  mt(A,x[:,i-1])
    K = mt(p, mt(h(x[:,i]).T, inv(mt(h(x[:,i]).T,mt(p, h(x[:,i]).T)) + R)))
    x[:,i] = x[:,i] + mt(K, (y_ob[:,i] - mt(h(x[:,i]).T,x[:,i])))
    p = p - mt(K, mt(h(x[:,i]).T,p))    
