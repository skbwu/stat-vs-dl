#!/usr/bin/env python
# coding: utf-8

# In[6]:


import numpy as np
import tensorflow as tf
import pandas as pd
from pinn.src import SEIRPINN_v2
import sys, copy, os, shutil, time, math


# In[2]:


# we're not forecasting. fix number of epochs too, following Scannell. N=1.0, standardized.
TFC, n_epochs = 0.0, 60000

# assemble all of our settings
settings = []; TT, d_obs = 0, 40/60
for params in [(0.2, 0.08, 0.1)]:
    for alpha in [0.15]:
        for seed in range(0, 100):
            for t_max in [60.0]:
                for lmbda in [10.0, 1.0, 100.0, 0.1, 1000.0]:
                    # also control all possible 0 vs. 1 vs. 2 missing component settings
                    for comp_obs in [[False, True, True], [True, True, True]]:
                        # create the tuple that we'll be using for this experiment
                        settings.append((params[0], params[1], params[2], alpha, seed, t_max, lmbda, comp_obs))
                        
# which setting are we running? (400x total for SEIR)
print(f"Running setting {int(sys.argv[1])} out of {len(settings)} total settings.")
b, gamma, s, alpha, seed, t_max, lmbda, comp_obs = settings[int(sys.argv[1])]
# b, gamma, s, alpha, seed, t_max, lmbda, comp_obs = settings[2]

# In[3]:


# convert comp_obs to active_comps (S is now deterministic function of E, I, R)
active_comps = ""
for i in range(3):
    active_comps += ["E", "I", "R"][i] if comp_obs[i] else ""
    
# make a directory
obs_desc = ""
for i in range(3):
    obs_desc += "_" + str(["E", "I", "R"][i] + "=" + str(comp_obs[i]))
foldername = f"PINN_logSEIR_beta={b}_gamma={gamma}_sigma={s}_alpha={alpha}_seed={seed}_TM={t_max}_lmbda={lmbda}{obs_desc}"
if foldername not in os.listdir("results"):
    os.mkdir(f"results/{foldername}")

output_folder = f"results/{foldername}/figures"
if os.path.isfile(f"{output_folder}/parameter_estimates.png"):
    sys.exit("This setting has already been run.")

# In[4]:


# load in our data, thinning based on density of observations
orig_data = pd.read_csv(f"tfpigp/data/logSEIR_beta={b}_gamma={gamma}_sigma={s}_alpha={alpha}_seed={seed}.csv")
data = orig_data.query(f"t <= {t_max}")
data = data.iloc[::int((data.index.shape[0] - 1) / (d_obs * t_max))]

# extract out the time vector + noisy observations
ts_obs = data.t.values.astype(np.float64)
x_obs = data[["E_obs", "I_obs", "R_obs"]].to_numpy().astype(np.float64)

# make certain components missing if necessary
for i, comp_obs_val in enumerate(comp_obs):
    if comp_obs_val != True:
        x_obs[:,i] = np.nan

x_obs = np.log(x_obs)
        
# alias to make loss calculations more intuitive
E_obs, I_obs, R_obs = x_obs[:,0:1], x_obs[:,1:2], x_obs[:,2:3]
        
# create our seir_data structure that the PINN needs (USING NOISED OBSERVATIONS)
seir_data = [ts_obs.reshape(-1, 1), E_obs, I_obs, R_obs]


# In[7]:


# set a seed for reproducibility
tf.random.set_seed(seed)

# dataframe to store our results
logs = pd.DataFrame(data=None, columns=["epoch", "beta", "gamma", "sigma",
                                        "total_loss", 
                                        "l2_E", "l2_I", "l2_R", 
                                        "physics_E", "physics_I", "physics_R"])

# let's also do batch-norm and log-transforming of params
pinn = SEIRPINN_v2(bn=True, log_opt=True, lr=1e-2, layers=5, layer_width=32, lmbda=lmbda)

# reference points for physics-based loss evaluation
t_max = 120
t_physics = tf.convert_to_tensor(np.linspace(TT, t_max, 321).reshape(-1, 1))

# train the PINN for 60K epochs ON IN-SAMPLE OBSERVATIONS ONLY!
for epoch in range(n_epochs):

    # one optimization step (we're not forecasting!)
    pinn.fit(observed_data=seir_data, 
             TT=TT, TM=t_max, TFC=TFC, is_forecasting=False, 
             epochs=1, verbose=False, active_comps=active_comps)
    
    # get our theta_hat = (beta, gamma) at this epoch
    beta_hat = np.exp(pinn.log_beta.numpy())
    gamma_hat = np.exp(pinn.log_gamma.numpy())
    sigma_hat = np.exp(pinn.log_sigma.numpy())
    theta_hat = np.array([beta_hat, gamma_hat, sigma_hat])
    
    # get our in-sample predictions + convert to numpy
    E_hat, I_hat, R_hat = pinn.NN(seir_data[0])
    E_hat, I_hat, R_hat = E_hat.numpy(), I_hat.numpy(), R_hat.numpy()
    
    # compute our S_hat deterministically
    S_hat = 1.0 - (E_hat + I_hat + R_hat)
    
    # L2-losses (E, I, R) - will nan out if needed
    l2_E = ((E_hat - E_obs) ** 2).mean()
    l2_I = ((I_hat - I_obs) ** 2).mean()
    l2_R = ((R_hat - R_obs) ** 2).mean()
    
    # physics-based losses (only on the three component)
    with tf.GradientTape(persistent=True) as g:
        g.watch(t_physics)
        [log_E_physics, log_I_physics, log_R_physics] = pinn.NN(t_physics)

    # gradients w.r.t. t_physics (implied by autograd)
    dlog_Edt_agd = g.gradient(log_E_physics, t_physics).numpy()
    dlog_Idt_agd = g.gradient(log_I_physics, t_physics).numpy()
    dlog_Rdt_agd = g.gradient(log_R_physics, t_physics).numpy()
    
    # convert the physics outputs to numpy
    E_physics = np.exp(log_E_physics.numpy())
    I_physics = np.exp(log_I_physics.numpy())
    R_physics = np.exp(log_R_physics.numpy())
    
    # get the implied S_physics
    S_physics = 1.0 - (E_physics + I_physics + R_physics)
    
    # what are the ODE-based derivatives?
    dEdt_phys = (beta_hat * S_physics * I_physics) - (sigma_hat * E_physics)
    dIdt_phys = (sigma_hat * E_physics) - (gamma_hat * I_physics) 
    dRdt_phys = gamma_hat * I_physics
    
    # compute physics-based loss per component
    physics_E = ((dlog_Edt_agd - dEdt_phys / E_physics) ** 2).mean()
    physics_I = ((dlog_Idt_agd - dIdt_phys / I_physics) ** 2).mean()
    physics_R = ((dlog_Rdt_agd - dRdt_phys / R_physics) ** 2).mean()
    
    # what's our total loss?
    total_loss = (lmbda * np.nansum([l2_E, l2_I, l2_R])) + (physics_E + physics_I + physics_R)
    
    # record epoch, (beta, gamma), total loss, L2-loss (E, I, R), physics-loss (E, I, R).
    row = [epoch] + list(theta_hat) + [total_loss, l2_E, l2_I, l2_R, physics_E, physics_I, physics_R]
    logs.loc[len(logs.index)] = row
    
'''
Need to save the following:
1. Logs of loss decomposition + params over time.
2. Model weights of final PINN's NN layers.
3. Point estimates of theta 
4. Point predictions on (E, I, R) at in-sample timesteps (note S is fully-determined!)
5. Physics-based Residuals (autograd minus ODE-implied)
'''

# 1. start with the loss logs
logs.to_csv(f"results/{foldername}/logs.csv", index=False)

# 2. PINN weights after last epoch.
pinn.NN.save_weights(f"results/{foldername}/pinn.weights.h5")

# 3. final estimate of theta_hat
theta_hat_final = pd.DataFrame(data=theta_hat.reshape(1, -1), columns=["beta", "gamma", "sigma"])
theta_hat_final.to_csv(f"results/{foldername}/theta_hats.csv", index=False)

# 4. in-sample predictions of (E, I, R)
preds = pd.DataFrame(data=np.hstack([seir_data[0], E_hat, I_hat, R_hat]), columns=["t", "E", "I", "R"])
preds.to_csv(f"results/{foldername}/preds.csv", index=False)

# 5. physics-residuals
physics_residuals = np.hstack([t_physics, dlog_Edt_agd - dEdt_phys / E_physics, dlog_Idt_agd - dIdt_phys / I_physics, dlog_Rdt_agd - dRdt_phys / R_physics])
physics_logs = pd.DataFrame(data=physics_residuals, columns=["t_physics", "E", "I", "R"]) # agd minus implied
physics_logs.to_csv(f"results/{foldername}/physics_residuals.csv", index=False)

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

truth = orig_data[["t", "E_true", "I_true", "R_true"]].query(f"t <= {t_max}")

# Load predictions, observations, and parameter logs
t_forecast = np.linspace(TT, t_max, int(((t_max - TT) * 40) + 1)).reshape(-1, 1)
E_hat, I_hat, R_hat = pinn.NN(t_forecast)
E_hat, I_hat, R_hat = E_hat.numpy(), I_hat.numpy(), R_hat.numpy()
preds_forecast = pd.DataFrame(data=np.hstack([t_forecast, E_hat, I_hat, R_hat]), columns=["t", "E", "I", "R"])

observations = pd.DataFrame(data=np.hstack([seir_data[0], np.exp(E_obs), np.exp(I_obs), np.exp(R_obs)]),
                            columns=["t", "E_obs", "I_obs", "R_obs"])
logs = pd.read_csv(f"results/{foldername}/logs.csv")
theta_truth = {"beta": b, "gamma": gamma, "sigma": s}

output_folder = f"results/{foldername}/figures"
os.makedirs(output_folder, exist_ok=True)  # Create the output folder if it doesn't exist

# 1. Plot E, I, R in original scale
plt.figure(figsize=(12, 8))
for comp, color in zip(["E", "I", "R"], ["blue", "green", "red"]):
    plt.plot(preds_forecast["t"], np.exp(preds_forecast[comp]), label=f"{comp} Predicted", color=color)
    plt.scatter(observations["t"], observations[f"{comp}_obs"], label=f"{comp} Observed", color=color, marker='x')
    plt.plot(truth["t"], truth[f"{comp}_true"], label=f"{comp} True", color=color, linestyle='--')
plt.title("E, I, R Predictions vs. Observations (Original Scale)")
plt.xlabel("Time")
plt.ylabel("Value")
plt.legend()
plt.grid(True)
plt.savefig(f"{output_folder}/original_scale.png")  # Save figure
plt.show()

# 2. Plot log E, log I, log R
plt.figure(figsize=(12, 8))
log_preds = preds[["E", "I", "R"]]
log_obs = np.log(observations[["E_obs", "I_obs", "R_obs"]].replace(0, np.nan))
for comp, color in zip(["E", "I", "R"], ["blue", "green", "red"]):
    plt.plot(preds_forecast["t"], preds_forecast[comp], label=f"log {comp} Predicted", color=color)
    plt.scatter(observations["t"], log_obs[f"{comp}_obs"], label=f"log {comp} Observed", color=color, marker='x')
    plt.plot(truth["t"], np.log(truth[f"{comp}_true"]), label=f"{comp} True", color=color, linestyle='--')
plt.title("log(E), log(I), log(R) Predictions vs. Observations")
plt.xlabel("Time")
plt.ylabel("Log Value")
plt.legend()
plt.grid(True)
plt.savefig(f"{output_folder}/log_scale.png")  # Save figure
plt.show()

# 3. Plot parameter estimates vs. ground truth
plt.figure(figsize=(12, 8))
epochs = logs["epoch"]
for param, color in zip(["beta", "gamma", "sigma"], ["blue", "green", "red"]):
    plt.plot(epochs, logs[param], label=f"Estimated {param}", color=color)
    plt.axhline(y=theta_truth[param], color=color, linestyle="--", label=f"True {param}")
plt.title("Parameter Estimates vs. Ground Truth")
plt.xlabel("Epoch")
plt.ylabel("Value")
plt.legend()
plt.grid(True)
plt.savefig(f"{output_folder}/parameter_estimates.png")  # Save figure
plt.show()
