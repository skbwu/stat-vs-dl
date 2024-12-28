import numpy as np
import tensorflow as tf
import pandas as pd
from src import LorenzPINN
import sys, copy, os, shutil, time, math
from tqdm.notebook import tqdm

# fixed settings
TT, TFC, t_max, n_epochs = 0.0, 0.0, 8.0, 60000
comp_obs, active_comps = [True, True, True], "XYZ" # all components observed

# assemble all of our settings (1000x)
settings = [];
for lmbda in [0.1, 1.0, 10.0, 100.0, 1000.0]:
    for rho in [23.0, 28.0]:
        for alpha in [0.05]:
            for d_obs in [10]:
                for seed in range(0, 100):
                        
                    # create the tuple that we'll be using for this experiment
                    settings.append((lmbda, rho, alpha, d_obs, seed))
                    
# which setting are we running?
lmbda, rho, alpha, d_obs, seed = settings[int(sys.argv[1])]

# load in our data, thinning based on density of observations
orig_data = pd.read_csv(f"data/LORENZ_rho={rho}_alpha={alpha}_seed={seed}.csv")
data = orig_data.query(f"t <= {t_max}")
data = data.iloc[::int((data.index.shape[0] - 1) / (d_obs * t_max))]

# extract out the time vector + noisy observations
ts_obs = data.t.values.astype(np.float64)
x_obs = data[["X_obs", "Y_obs", "Z_obs"]].to_numpy().astype(np.float64)

# make certain components missing if necessary
for i, comp_obs_val in enumerate(comp_obs):
    if comp_obs_val != True:
        x_obs[:,i] = np.nan
        
# alias to make loss calculations more intuitive
X_obs, Y_obs, Z_obs = x_obs[:,0:1], x_obs[:,1:2], x_obs[:,2:3]
        
# create our lorenz_data structure that the PINN needs (USING NOISED OBSERVATIONS)
lorenz_data = [ts_obs.reshape(-1, 1), X_obs, Y_obs, Z_obs]

# set a seed for reproducibility
tf.random.set_seed(seed)

# dataframe to store our results
logs = pd.DataFrame(data=None, columns=["epoch", "beta", "rho", "sigma", 
                                        "total_loss", "l2_X", "l2_Y", "l2_Z", 
                                        "physics_X", "physics_Y", "physics_Z"])

# they are doing batch-norm and log-transforming all of their beta, rho, and sigma (showcased settings)
pinn = LorenzPINN(bn=True, log_opt=True, lr=1e-2, layers=3, 
                  layer_width=32, lmbda=lmbda)

# reference points for physics-based loss evaluation
t_max_pred = 8.0 # purely concerned with in-sample!
t_physics = tf.convert_to_tensor(np.linspace(TT, t_max_pred, 
                                             int(((t_max_pred - TT) * 40) + 1))\
                                 .reshape(-1, 1))

# train the PINN for 60K epochs ON IN-SAMPLE OBSERVATIONS ONLY!
for epoch in range(n_epochs):

    # one optimization step (we're not forecasting!)
    pinn.fit(observed_data=lorenz_data, 
             TT=TT, TM=t_max_pred, TFC=TFC, is_forecasting=False,
             epochs=1, verbose=False, active_comps=active_comps)
    
    # get our theta_hat = (beta, rho, sigma) at this epoch
    beta_hat = np.exp(pinn.c3.value.numpy())
    rho_hat = np.exp(pinn.c2.value.numpy())
    sigma_hat = np.exp(pinn.c1.value.numpy())
    theta_hat = np.array([beta_hat, rho_hat, sigma_hat])
    
    # get our in-sample predictions + convert to numpy
    X_hat, Y_hat, Z_hat = pinn.NN(lorenz_data[0])
    X_hat, Y_hat, Z_hat = X_hat.numpy(), Y_hat.numpy(), Z_hat.numpy()
    
    # L2-losses (X, Y, Z) - will nan out if needed
    l2_X = ((X_hat - X_obs) ** 2).mean()
    l2_Y = ((Y_hat - Y_obs) ** 2).mean()
    l2_Z = ((Z_hat - Z_obs) ** 2).mean()
    
    # physics-based losses
    with tf.GradientTape(persistent=True) as g:
        g.watch(t_physics)
        [X_physics, Y_physics, Z_physics] = pinn.NN(t_physics)

    # gradients w.r.t. t_physics (implied by autograd)
    dXdt_agd = g.gradient(X_physics, t_physics).numpy()
    dYdt_agd = g.gradient(Y_physics, t_physics).numpy()
    dZdt_agd = g.gradient(Z_physics, t_physics).numpy()
    
    # convert the physics outputs to numpy
    X_physics = X_physics.numpy()
    Y_physics = Y_physics.numpy()
    Z_physics = Z_physics.numpy()
    
    # what are the ODE-based derivatives?
    dXdt_phys = sigma_hat * (Y_physics - X_physics)
    dYdt_phys = (X_physics * (rho_hat - Z_physics)) - Y_physics
    dZdt_phys = (X_physics * Y_physics) - (beta_hat * Z_physics)
    
    # compute physics-based loss per component
    physics_X = ((dXdt_agd - dXdt_phys) ** 2).mean()
    physics_Y = ((dYdt_agd - dYdt_phys) ** 2).mean()
    physics_Z = ((dZdt_agd - dZdt_phys) ** 2).mean()
    
    # what's our total loss?
    total_loss = (lmbda * np.nansum([l2_X, l2_Y, l2_Z])) + (physics_X + physics_Y + physics_Z)
    
    # record epoch, beta, rho, sigma, total loss, L2-loss (X, Y, Z), physics-loss (X, Y, Z).
    row = [epoch] + list(theta_hat) + [total_loss, l2_X, l2_Y, l2_Z, physics_X, physics_Y, physics_Z]
    logs.loc[len(logs.index)] = row

# make a directory
foldername = f"lmbda={lmbda}_rho={rho}_alpha={alpha}_dobs={d_obs}_seed={seed}"
if foldername not in os.listdir("results/pinn"):
    os.mkdir(f"results/pinn/{foldername}")
    
'''
Need to save the following:
1. Logs of loss decomposition + params over time.
2. Model weights of final PINN's NN layers.
3. Point estimates of theta 
4. Point predictions on (X, Y, Z) at in-sample timesteps
5. Physics-based Residuals (autograd minus ODE-implied)
6. Raw data.
7. Interval inferred trajectories (including forecasted, if called for)
'''
# 1. start with the loss logs
logs.to_csv(f"results/pinn/{foldername}/logs.csv", index=False)

# 2. PINN weights after last epoch.
pinn.NN.save_weights(f"results/pinn/{foldername}/pinn.weights.h5")

# 3. final estimate of theta_hat
theta_hat_final = pd.DataFrame(data=theta_hat.reshape(1, -1), columns=["beta", "rho", "sigma"])
theta_hat_final.to_csv(f"results/pinn/{foldername}/theta_hats.csv", index=False)

# 4. in-sample predictions of (X, Y, Z)
preds = pd.DataFrame(data=np.hstack([lorenz_data[0], X_hat, Y_hat, Z_hat]), columns=["t", "X", "Y", "Z"])
preds.to_csv(f"results/pinn/{foldername}/in-sample_preds.csv", index=False)

# 5. physics-residuals
physics_residuals = np.hstack([t_physics, dXdt_agd - dXdt_phys, dYdt_agd - dYdt_phys, dZdt_agd - dZdt_phys])
physics_logs = pd.DataFrame(data=physics_residuals, columns=["t", "X", "Y", "Z"]) # agd minus implied
physics_logs.to_csv(f"results/pinn/{foldername}/physics_residuals.csv", index=False)

# 6. noisy observations
noisy_obs = pd.DataFrame(data=np.hstack([ts_obs.reshape(-1, 1), X_obs, Y_obs, Z_obs]), 
                         columns=["t", "X", "Y", "Z"])
noisy_obs.to_csv(f"results/pinn/{foldername}/noisy_obs.csv", index=False)

# 7. out-of-sample predictions
oos_preds = pd.DataFrame(data=np.hstack([t_physics, np.hstack(pinn.NN(t_physics))]), 
                         columns=["t", "X", "Y", "Z"])
oos_preds.to_csv(f"results/pinn/{foldername}/oos_preds.csv", index=False)