import numpy as np
from numpy import matmul as mt 
from numpy.linalg import inv 
import matplotlib.pyplot as plt

k = 1   #observe covariance

np.random.seed(123)
t = 0.5
irr = 50
A = np.array([[1, 0, t, 0, 0.5*t**2, 0],
              [0, 1, 0, t, 0, 0.5*t**2],
              [0, 0, 1, 0, t, 0],
              [0, 0, 0, 1, 0, t],
              [0, 0, 0, 0, 1, 0],
              [0, 0, 0, 0, 0, 1]])

Q = np.array([[0.05*t**5, 0, 0.125*t**4, 0, t**3/6, 0],
              [0, 0.05*t**5, 0, 0.125*t**4, 0, t**3/6],
              [0.125*t**4, 0, t**3/3, 0, 0.5*t**2, 0],
              [0, 0.125*t**4, 0, t**3/3, 0, 0.5*t**2],
              [t**3/6, 0, 0.5*t**2, 0, t, 0],
              [0, t**3/6, 0, 0.5*t**2, 0, t]])

H = np.array([[1, 0, 0, 0, 0, 0],
              [0, 1, 0, 0, 0, 0]])

R = np.diag([k,k])

q = np.random.multivariate_normal(np.zeros(6), Q, size=irr-1).T
r = np.random.multivariate_normal(np.zeros(2), R, size=irr).T

x_T = np.zeros([6,irr])
x_T[:,0] = np.array([0,0,1,0.6,0.4,0.8])

#generate truth
for i in range(1, irr):
    x_T[:,i] = mt(A, x_T[:,i-1].T).flatten() + q[:,i-1]

y_ob = mt(H, x_T) + r

x = np.zeros([6,irr]) #mean
p = np.diag([0.1,0.1,0.1,0.1,0.5,0.5])

#forward Ax + q q~Q
#observe Hx + r r~R
for i in range(1,irr):
    p = Q + mt(mt(A, p), A.T)
    x[:,i] =  mt(A,x[:,i-1])
    K = mt(p, mt(H.T, inv(mt(H,mt(p, H.T)) + R)))
    x[:,i] = x[:,i] + mt(K, (y_ob[:,i] - mt(H,x[:,i])))
    p = p - mt(K, mt(H,p))

fig, ax = plt.subplots(1,3)

for i in range(3):
    ax[i].plot(x[2*i,:],x[2*i+1,:],'o--',label='predict')
    ax[i].plot(x_T[2*i,:], x_T[2*i+1,:], label='truth')
    ax[i].legend()

ax[0].set_title('Position')
ax[1].set_title('Velocity')
ax[2].set_title('Acceleration')
plt.show()