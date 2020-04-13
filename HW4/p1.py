from math import sqrt
import matplotlib.pyplot as plt
import numpy as np
from numpy import matmul as mt
from scipy.linalg import cholesky


n = 2
size = 10000
Cov_x = np.array([[32,15],[15,40]])
mean_x = np.array([[0],[1]])
J = np.array([[1,1],[0,2]])
alpha = 0.5
beta = 2
kappa = 3 - n
lamb = alpha**2*(n + kappa) - n

def forward(x):
    return np.array([x.sum(),0.5*x[0]**2 + x[1]**2])

x = np.random.multivariate_normal(mean_x.reshape(-1), Cov_x, size)
y_line_mean = forward(mean_x.reshape(-1))
y = np.zeros([size,2])
for i in range(size):
    y[i,:] = forward(x[i,:])

def UT(n,alpha,beta,kappa,Cov_x):
    lamb = alpha**2*(n + kappa) - n
    sigmax = np.zeros([2*n+1,2])
    sigmax[0,:] = mean_x.reshape(-1)

    for i in range(n):
        sigmax[i+1,:] = sigmax[0,:] + sqrt(n + lamb) * cholesky(Cov_x,lower=True)[:,i]
        sigmax[i + 1 + n,:] = sigmax[0,:] - sqrt(n + lamb) * cholesky(Cov_x,lower=True)[:,i]

    y_ut = np.zeros([2*n+1,2])

    for i in range(2*n+1):
        y_ut[i,:] = forward(sigmax[i,:])

    weight_m = np.zeros([2*n+1])
    weight_c = np.zeros([2*n+1])
    weight_m[0] = lamb/(n+lamb)
    weight_c[0] = weight_m[0] + 1 - alpha**2 + beta
    weight_m[1:2*n+1] = np.ones([2*n]) * 0.5 / (n+lamb)
    weight_c[1:2*n+1] = np.ones([2*n]) * 0.5 / (n+lamb)

    E = np.zeros([2])
    Cov = np.zeros([2,2])
    E = (weight_m.reshape(-1,1)*y_ut).sum(axis=0)
    for i in range(2*n+1):
        Cov += weight_c[i]*mt((y_ut[i,:] - E).reshape(2,-1),(y_ut[i,:] - E).reshape(-1,2))
    return sigmax, E, Cov

sigmax, E, Cov = UT(n,alpha,beta,kappa,Cov_x)
print(Cov,Cov_x)

fig1, ax1 = plt.subplots(1,2)
ax1[0].scatter(x[:,0], x[:,1], alpha = 0.3, c='#C0C0C0', label='MC')
ax1[0].scatter(x[:,0].mean(), x[:,1].mean(), c='r', label='MC mean')
ax1[0].legend()

ax1[1].scatter(y[:,0], y[:,1], alpha = 0.3, c='#C0C0C0', label='MC')
ax1[1].scatter(y[:,0].mean(), y[:,1].mean(), label='MC mean')
ax1[1].scatter(y_line_mean[0],y_line_mean[1],label='Linear Transform mean')
ax1[1].legend()

fig2, ax2 = plt.subplots(1,2)
ax2[0].scatter(x[:,0], x[:,1], alpha = 0.3, c='#C0C0C0', label='MC')
ax2[0].scatter(x[:,0].mean(), x[:,1].mean(), c='r', label='MC mean')
ax2[0].scatter(sigmax[:,0], sigmax[:,1], marker = 'x',c='#ff7f0e',label = 'Sigma Points')
ax2[0].legend()
ax2[1].scatter(y[:,0], y[:,1], alpha = 0.3, c='#C0C0C0', label='MC')
ax2[1].scatter(y[:,0].mean(), y[:,1].mean(), label='MC mean')
ax2[1].scatter(E[0],E[1], marker = 'x', label='UT mean')
ax2[1].legend()

plt.show()



