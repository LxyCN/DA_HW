'''
import numpy as np 

np.random.seed(123)
t =0.5
truth = np.array([0,0,1,0.6,0.4,0.8])[np.newaxis].T
x = np.zeros(6)[np.newaxis].T
a = np.array([[1, 0, t, 0, 0.5*t**2, 0],
              [0, 1, 0, t, 0, 0.5*t**2],
              [0, 0, 1, 0, t, 0],
              [0, 0, 0, 0, 1, 0],
              [0, 0, 0, 0, 0, 1]])
q = np.array([[0.05*t**5, 0, 0.125*t**4, 0, t**3/6, 0],
              [0, 0.05*t**5, 0, 0.125*t**4, 0, t**3/6],
              [0.125*t**4, 0, t**3/6, 0, 0.5*t**2, 0],
              [0, 0.125*t**4, 0, t**3/6, 0, 0.5*t**2],
              [t**3/6, 0, 0.5*t**2, 0, t, 0],
              [0, t**3/6, 0, 0.5*t**2, 0, t]])
h = np.array([[1, 0, 0, 0, 0, 0],
              [0, 1, 0, 0, 0, 0]])'''

import numpy as np
import sys
import time
import os
total = 49
os.system('clear')
print('Deleting Operating System...')
for i in range(total+1):
    sys.stdout.write("\r[" + "=" * i +  " "*(total - i) + "]" +  format(i / total*100, '.0f') + "%")
    time.sleep(np.random.uniform(high=0.5))
    sys.stdout.flush()
print('\nSystem Deleted!')
time.sleep(0.5)
while 1:
    print(chr(int(100*np.random.uniform())),end = '')
    time.sleep(0.01)


