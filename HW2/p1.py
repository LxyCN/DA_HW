import numpy as np
import matplotlib.pyplot as plt

N_samples = 1000000
Bn = np.zeros(15)
dist = np.zeros(N_samples)
p, F = np.zeros(15), np.zeros(15)
low_Bound  = 0.9

def probability(x):
    B_count, S_count= 0, 0
    for i in range(N_samples):
        distance = np.sqrt((x[:,i]*x[:,i]).sum())
        if distance <= 1:
            B_count += 1
            if distance >= low_Bound:
                S_count += 1
    return B_count/N_samples, S_count/B_count


for i in range(15):
    #print(i)
    sample = np.zeros([i+1, N_samples])
    for j in range(i+1):
        sample[j] = np.random.uniform(low = -1, size=N_samples)
    p[i], F[i] = probability(sample)
    Bn[i] = p[i] * pow(2,i+1)

fig,(ax1,ax2) = plt.subplots(2,1)
ax1.plot(Bn, 'b.-')
ax1.set_ylabel('Volume of Bn')
ax2.plot(F, 'r.--')
ax2.set_ylabel('Fraction of Bn occupied by Sn')
ax2.set_xlabel('Dimension')
plt.tight_layout()
plt.show()

