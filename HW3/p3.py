import matplotlib.pyplot as plt
import numpy as np
from numpy import cos
from numpy import matmul as mt
from numpy import sin
from numpy.linalg import inv

#np.random.seed(123)
q1 = 0.2
q2 = 0.1
irr = 100
r = 0.1   #observe

def h(x):
    return x[2]*sin(x[0]) + np.random.normal(scale=r)

A = np.array([[1, 0.01, 0],
              [0, 1, 0],
              [0, 0, 1]])
Q = np.array([[0.01**3/3*q1, 0.5*0.01**2*q1, 0],
              [0.5*0.01**2*q1, 0.01*q1, 0],
              [0, 0, 0.01*q2]])

#observe model               


q = np.random.multivariate_normal(np.zeros(3), Q, size=irr).T
R = np.random.normal(scale=r, size=irr)

x_T = np.zeros([3, irr])
x_T[:, 0] = np.array([0, 10, 1])
for i in range(1, irr):
    x_T[:, i] = mt(A, x_T[:, i-1]) + q[:, i-1]
y_ob = x_T[2, :]*sin(x_T[0, :]) + R
x = np.zeros([3, irr])
x[:,0] = np.array([0, 10, 1])
p = np.diag([3, 3, 3])
H = np.zeros([irr, 3])
H[0, :] = np.array([x[2, 0]*cos(x[0, 0]), 0, sin(x[0, 0])])

for i in range(1, irr):
    p = Q + mt(mt(A, p), A.T)
    x[:, i] = mt(A, x[:, i-1])
    H[i, :] = np.array([x[2, i]*cos(x[0, i]), 0, sin(x[0, i])])
    S = mt(mt(H[i, :], p), H[i, :].T) + r
    K = mt(p, H[i, :].T)/S
    x[:, i] = x[:, i] + K*(y_ob[i] - h(x[:, i]))
    p = p - mt(K*S, K.T)


fig, ax = plt.subplots()
xco = np.linspace(1,irr,irr)
ax.plot(xco,x[2,:]*sin(x[0,:]),label='Filtered')
ax.plot(xco,x_T[2,:]*sin(x_T[0,:]), label='Truth')
ax.plot(xco,y_ob,'.',label='Data')
ax.legend()
plt.show()
