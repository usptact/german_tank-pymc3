#
# German Tank problem demo
#

import sys

from random import sample

import pymc3 as pm
import numpy as np
import matplotlib.pyplot as plt

# true number of tanks
N = 1000

# number of captured tanks
K = 20

y = sample( range(1,N+1), K )
y = np.array( y )

print( y )

model = pm.Model()

with model:
    N = pm.DiscreteUniform( 'N', lower=np.max(y), upper=10000 )
    lam = pm.HalfNormal( 'lam', sd=10 )
    nobs = pm.Poisson( 'nobs', mu=N*lam, observed=len(y) )
    y_obs = pm.DiscreteUniform( 'y_obs', lower=1, upper=N, observed=y )
    step = pm.Metropolis()
    trace = pm.sample( 1000000, step=step, njobs=4, progressbar=True )

pm.summary( trace )

pm.traceplot( trace )
plt.show()
