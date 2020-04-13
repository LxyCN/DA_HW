from math import sqrt
import matplotlib.pyplot as plt
import numpy as np
from numpy import cos, sin
from numpy import matmul as mt 
from scipy.linalg import cholesky


np.random.seed(10)
q1 = 100
q2 = 0.1
irr = 100
r = 0.1   #observe

n = 3
alpha = 0.3
beta = 2
kappa = 3 - n
lamb = alpha**2*(n + kappa) - n

def h(x):
    return x[2,:]*sin(x[0,:]) 

A = np.array([[1, 0.01, 0],
              [0, 1, 0],
              [0, 0, 1]])
Q = np.array([[0.01**3/3*q1, 0.5*0.01**2*q1, 0],
              [0.5*0.01**2*q1, 0.01*q1, 0],
              [0, 0, 0.01*q2]])

q = np.random.multivariate_normal(np.zeros(3), Q, size=irr).T
R = np.random.normal(scale=r, size=irr)

def UT(mean_x, Cov_x):
    lamb = alpha**2*(n + kappa) - n
    sigmax = np.zeros([3,2*n+1])
    sigmax[:,0] = mean_x

    for i in range(n):
        sigmax[:,i+1] = mean_x + sqrt(n + lamb) * cholesky(Cov_x,lower=True)[:,i]
        sigmax[:,i + 1 + n] = mean_x - sqrt(n + lamb) * cholesky(Cov_x,lower=True)[:,i]

    y_ut = h(sigmax)

    weight_m = np.zeros([2*n+1])
    weight_c = np.zeros([2*n+1])
    weight_m[0] = lamb/(n+lamb)
    weight_c[0] = weight_m[0] + 1 - alpha**2 + beta
    weight_m[1:2*n+1] = np.ones([2*n]) * 0.5 / (n+lamb)
    weight_c[1:2*n+1] = np.ones([2*n]) * 0.5 / (n+lamb)

    mu = (weight_m*y_ut).sum()
    S = (weight_c*(y_ut - mu)*(y_ut - mu)).sum() + r
    C = (weight_c*(y_ut - mu)*(sigmax - mean_x.reshape(3,-1))).sum(axis=1)

    return mu, S, C



x_T = np.zeros([3, irr])
x_T[:, 0] = np.array([0, 10, 1])
for i in range(1, irr):
    x_T[:, i] = mt(A, x_T[:, i-1]) + q[:, i-1]

y_ob = h(x_T) + R
x = np.zeros([3, irr])
x[:,0] = np.array([0, 10, 1])
p = np.diag([3, 3, 3])

for i in range(1, irr):
    p = Q + mt(mt(A, p), A.T)
    x[:, i] = mt(A, x[:, i-1])
    mu, S, C = UT(x[:,i],p)
    K = C/S
    x[:,i] = x[:,i] + K*(y_ob[i] - mu)
    p = p - S * mt(K.reshape(3,1),K.reshape(1,3))


fig, ax = plt.subplots()
xco = np.linspace(1,irr,irr)
ax.plot(xco,x[2,:]*sin(x[0,:]),label='Filtered')
ax.plot(xco,x_T[2,:]*sin(x_T[0,:]), alpha=0.5,label='Truth')
ax.plot(xco,y_ob,'.',label='Data')
ax.legend()
plt.show()