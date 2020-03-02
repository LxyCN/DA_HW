from scipy.special import gamma
import numpy as np 
import matplotlib.pyplot as plt

N_samples = 1000000
M = 3.1

def beta(x):
    return gamma(6)/gamma(2)/gamma(4)*x*(1-x)**3

resample = []
sample = np.random.uniform(size=N_samples)
judge = np.random.uniform(size=N_samples)

for i in range(N_samples):
    if judge[i] < beta(sample[i])/M :
        resample.append(sample[i])

resample = np.array(resample)
accp_ratio = len(resample)/N_samples
print(accp_ratio)

x = np.linspace(0, 1, num = 50)
fig, ax = plt.subplots()
ax.hist(resample, 50)
ax.set_xlabel('x')
ax.set_ylabel('Number of Accepted Samples')

ax1 = ax.twinx()
ax1.plot(x, beta(x),'r', label = 'Truth')
ax1.set_ylabel('Probability of Beta Distribution')
#ax1.plot(x, M*np.ones(x.shape), 'k', label = 'Proposal Distribution')
ax1.legend()
plt.show()

#M = 2.1, accp 0.476968
#M = 1.1, accp 0.636942
#M = 3.1, accp 0.322347
