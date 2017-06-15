import numpy as np, matplotlib.pylab as plt, pandas as pd, mpmath as mp, scipy.special as ss
from scipy.special import gammainc as lower_gamma
from scipy.special import gammaincc as upper_gamma
from ars import ARS
from pdf_ccdf import pdf_ccdf
from statsmodels.base.model import GenericLikelihoodModel
import scipy.integrate as sint
from pdf_functions import *
%matplotlib inline

fig = plt.figure(figsize=(15,8))
mu_o = 1.
b_o = 1.5
nu_o = 5.
ars = ARS(f, fprima, xi = np.logspace(-10,2,1000), b=b_o, nu=nu_o, Cn=Cn_calc(b_o, nu_o), mu=mu_o)
years_data = 10.
sample_raw = ars.draw(years_data * 365.)

sample = sample_raw/sample_raw.mean()
Qe = np.percentile(sample, range(1,100))
# plt.semilogy(Qe)
q = np.logspace(np.log10(sample.min()), np.log10(sample.max()), 10000)
pdfq = pdfq_calc(b_o, nu_o, 1)#, sample.min(), sample.max())
pdf = pdfq(q)
dq = np.diff(q).mean()
X =  sint.cumtrapz(pdf,q)
Qt = np.zeros(99)
for ii in range(1,100):
    try:
        idx = np.where(X>=ii/100.)[0][0]
        Qt[ii-1] = q[idx]
    except:
        Qt[ii-1] = np.nan

plt.plot(Qe, Qt, 'ro')
print(1. - (np.sum((Qt - Qe)**2) / np.sum((Qe - Qe.mean())**2)))
