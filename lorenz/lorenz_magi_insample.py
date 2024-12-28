import numpy as np
import pandas as pd
import tensorflow as tf
import sys, pickle

# core MAGI-TFP class
import magi_v2

# model governing the Lorenz system, appropriate for tensorflow vectorization
def f_vec(t, X, thetas):
    '''
    1. X - array containing (X, Y, Z) components. Suppose it is (N x D) for vectorization.
    2. theta - array containing (beta, rho, sigma) components.
    '''
    return tf.concat([thetas[2] * (X[:,1:2] - X[:,0:1]), # dx/dt = sigma * (y-x)
                      X[:,0:1] * (thetas[1] - X[:,2:3]) - X[:,1:2], # dy/dt = x * (rho - z) - y
                      X[:,0:1]*X[:,1:2] - thetas[0]*X[:,2:3], # dz/dt = x*y - beta*z
                     ], axis=1)

# create our list of settings (200x settings) - let's go fully observed for now
settings = []
for rho in [23.0, 28.0]:
    for alpha in [0.05]:
        for d_obs in [10]:

            # compute the implied discretization
            discretization = int(np.log2(40 // d_obs))
            for seed in range(0, 100):
                settings.append((rho, alpha, d_obs, discretization, seed))
            
# which setting are we running?
rho, alpha, d_obs, discretization, seed = settings[int(sys.argv[1])]

# fixed settings
t_max, hmc_steps = 8.0, 3000

# no missing data!
comp_obs = [True, True, True]

# load in our data, thinning based on density of observations
raw_data = pd.read_csv(f"data/LORENZ_rho={rho}_alpha={alpha}_seed={seed}.csv").query(f"t <= {t_max}")
obs_data = raw_data.iloc[::int((raw_data.index.shape[0] - 1) / (d_obs * t_max))]

# extract out the time vector + noisy observations
ts_obs = obs_data.t.values.astype(np.float64)

# get the noisy observations
X_obs = obs_data[["X_obs", "Y_obs", "Z_obs"]].to_numpy().astype(np.float64) # S is implicit!

# make certain components missing if necessary
for i, comp_obs_val in enumerate(comp_obs):
    if comp_obs_val != True:
        X_obs[:,i] = np.nan

# create our model - f_vec is the ODE governing equations function defined earlier.
model = magi_v2.MAGI_v2(D_thetas=3, ts_obs=ts_obs, X_obs=X_obs, bandsize=None, f_vec=f_vec)

# fit Matern kernel hyperparameters (phi1, phi2) as well as (Xhat_init, sigma_sqs_init, thetas_init)
model.initial_fit(discretization=discretization, no_noise_on_inferred=False, verbose=False)

# collect our samples from NUTS posterior sampling - toggle tempering=True to use log-tempering.
results = model.predict(num_results=hmc_steps, num_burnin_steps=hmc_steps, tempering=False, verbose=False)

# get our filename for this run
fname = f"rho={rho}_alpha={alpha}_dobs={d_obs}_discret={discretization}_seed={seed}.pickle"

# save our results to a file - we'll visualize later.
with open(f"results/magi/{fname}", "wb") as file:
    pickle.dump(results, file)