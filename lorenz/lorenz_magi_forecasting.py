import numpy as np
import pandas as pd
import tensorflow as tf
from scipy.integrate import solve_ivp
import os, sys, pickle

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


# initial data settings
d_obs = 20 # no. of observations per unit time
t_max = 2.0 # length of observation interval (in-sample)
comp_obs = [True, True, True] # which components are observed?

# discretization on the in-sample? (for out-of-sample, just doing 40 pts / unit time)
discretization = 1

# create our list of 200 settings to run for forecasting
settings = []
for rho in [23.0, 28.0]:
    for seed in range(0, 100):
        settings.append((rho, seed))
        
# command-line argument for rho + seed
rho, seed = settings[int(sys.argv[1])]

# fix alpha (our noise level)
alpha = 0.0005

# how long do we want to forecast into the future, and how big is each stepsize? 
t_forecast, t_stepsize = 5.0, 1.0

'''
Lorenz forecasting best practices is to reestimate phi after each sequential prediction step.
There are two methods of doing this:
1. Refitting phi on the previous timestep's entire trajectory.
2. Refitting phi on the previous timestep's last length-1 interval ("online pilot")
'''
# are we using online pilot or not?
online_pilot = True

# how many hmc_steps?
hmc_steps = 3000

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
model.initial_fit(discretization=discretization, verbose=False)

# initial in-sample fit.
results = model.predict(
    num_results=hmc_steps, 
    num_burnin_steps=hmc_steps, 
    tempering=False, 
    verbose=False)

# encode ODEs for solve_ivp numerical-integration
def lorenz(t, y, beta, rho, sigma):

    # unpack y
    X, Y, Z = tuple(y)

    # dXdt = sigma * (Y-X); dYdt = x(rho - z) - y; dZdt = xy - beta*z
    dXdt = sigma * (Y-X)
    dYdt = (X * (rho - Z)) - Y
    dZdt = (X*Y) - (beta*Z)

    # return only the derivatives
    return np.array([dXdt, dYdt, dZdt])

# how many steps does this forecasting task require?
n_steps = int((t_forecast - t_max) // t_stepsize)

# initialize our t_max_step
t_max_step_old = t_max

# create a dictionary to store our results for each sequential step
all_results = {0 : results}

# forecast sequentially: e.g. (0.0, t_max + step*t_stepsize)
for step in range(n_steps):
    
    # create our new I interval - what is the end of this step?
    t_max_step = t_max_step_old + t_stepsize
    I_new_out = np.linspace(t_max_step_old, t_max_step, int(t_stepsize * 40 + 1))[1:].reshape(-1, 1)
    I_new = np.vstack([model.I, I_new_out])
    
    # get our thetas to do the numerical integration (last ODE)
    thetas_last = results["thetas_samps"][-1]
    
    # ODE integrate last trajectory forward to build Xhat_init
    t_eval = np.vstack([model.I[-1, 0], I_new_out])
    Xhat_init_out = solve_ivp(fun=lorenz, t_span=(t_max_step_old, t_max_step), 
                              y0=results["X_samps"][-1, -1], 
                              t_eval=t_eval.flatten(), 
                              atol=1e-10, rtol=1e-10, args=tuple(thetas_last)).y.T[1:,:]
    Xhat_init = np.vstack([results["X_samps"][-1], Xhat_init_out])
    
    # get our mean prediction from previous inference step
    X_mean_prev = results["X_samps"].mean(axis=0)
    
    # refit our (phi1, phi2) based on our previous predictions (but only use the last timestep inferred)
    if online_pilot:
        pilot_indices = (model.I >= t_max_step_old - t_stepsize).flatten()
        I_pilot, X_pilot = model.I[pilot_indices], X_mean_prev[pilot_indices]
        hparams_new = model._fit_kernel_hparams(I=I_pilot, X_filled=X_pilot, verbose=True)
    
    # use the full previous timestep interval
    else:
        hparams_new = model._fit_kernel_hparams(I=model.I, X_filled=X_mean_prev, verbose=True)
        
    
    # update our model
    model.update_kernel_matrices(I_new=I_new, 
                                 phi1s_new=hparams_new["phi1s"], 
                                 phi2s_new=hparams_new["phi2s"])
    
    # will need to update mu_ds too
    model.mu_ds = Xhat_init.mean(axis=0)
    
    # warm-start: theta_init, Xhat_init, sigma_sq_init
    model.Xhat_init = Xhat_init
    model.thetas_init = results["thetas_samps"][-1]
    model.sigma_sqs_init = results["sigma_sqs_samps"].mean(axis=0)
    
    # collect our forecasting samples
    results = model.predict(
        num_results=hmc_steps, 
        num_burnin_steps=hmc_steps, 
        tempering=False, 
        verbose=True)
    
    # finally, update t_max_step_old
    t_max_step_old = t_max_step
    
    # add the results to our dictionary
    all_results[step + 1] = results

# get our filename for this run
fname = f"rho={rho}_alpha={alpha}_dobs={d_obs}_discret={discretization}_seed={seed}.pickle"

# save our results to a file - we'll visualize later.
with open(f"results/magi_forecasting/{fname}", "wb") as file:
    pickle.dump(all_results, file)

