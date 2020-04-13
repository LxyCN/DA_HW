import scipy.stats as sts
import numpy as np
import matplotlib.pyplot as plt

def target(x):
    return 0.4*np.exp(-0.8*abs(x))

N_sample = 10000
sample1 = np.random.normal(size = N_sample)
sample1_pdf = sts.norm.pdf(sample1)
target1_pdf = target(sample1)
unnorm_weight1 = target1_pdf/sample1_pdf
z1 = unnorm_weight1.sum()/N_sample
weight1 = unnorm_weight1/z1
predicted_target_pdf1 = weight1*sample1_pdf

sample2 = sts.t.rvs(0.8, size = N_sample)
sample2_pdf = sts.t.pdf(sample2, 0.8)
target2_pdf = target(sample2)
unnorm_weight2 = target2_pdf/sample2_pdf
z2 = unnorm_weight2.sum()/N_sample
weight2 = unnorm_weight2/z2
predicted_target_pdf2 = weight2*sample2_pdf

x = np.linspace(-6, 6, num = 100)
fig1, ax = plt.subplots()
ax.plot(sample2, predicted_target_pdf2, 'r.', label = 'T Proposal')
ax.plot(sample1, predicted_target_pdf1, 'b.', label = 'Gaussian Proposal')
ax.plot(x, target(x), 'k--', label = 'Truth')
ax.set_xlabel('x')
ax.set_ylabel('Probability Density Function')
ax.legend()
plt.xlim(left = -6, right = 6)

fig2, ax2 = plt.subplots()
ax2.plot(sample2, weight2, 'r.', label = 'Weight of T Proposal')
ax2.plot(sample1, weight1, 'b.', label = 'Weight of Gaussian Proposal')
plt.xlim(left = -6, right = 6)
plt.ylim(top = 10, bottom = -0.1)
plt.show()