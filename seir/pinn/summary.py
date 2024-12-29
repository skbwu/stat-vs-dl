import os
import pandas as pd
import numpy as np
from glob import glob
from pinn.src import SEIRPINN_v2

# Define the base results directory
results_dir = "results"
data_dir = "tfpigp/data"
summary = {"EIR_True": {}, "EIR_Partial": {}}  # Nested dict for different lmbda values
all_predictions_true = {}  # Nested dict for lmbda
all_predictions_partial = {}

t_pred = np.linspace(0, 4.0, 101).reshape(-1, 1)

# Iterate through each folder in the results directory
for foldername in os.listdir(results_dir):
    run_dir = os.path.join(results_dir, foldername)
    if not os.path.isdir(run_dir):
        continue  # Skip non-directory files

    # Extract configuration and seed
    if "seed=" not in foldername or "E=" not in foldername or "I=" not in foldername or "R=" not in foldername:
        continue  # Skip if key information is missing
    seed = foldername.split("seed=")[1].split("_")[0]
    config = foldername.split("E=")[1].split("_")[0] + "_" + foldername.split("I=")[1].split("_")[0] + "_" + foldername.split("R=")[1].split("_")[0]

    # Extract lambda from folder name
    if "lmbda=" not in foldername:
        continue  # Skip if lmbda information is missing
    lmbda = float(foldername.split("lmbda=")[1].split("_")[0])

    # Initialize entries for lmbda if not already present
    if lmbda not in summary["EIR_True"]:
        summary["EIR_True"][lmbda] = []
        summary["EIR_Partial"][lmbda] = []
    if lmbda not in all_predictions_true:
        all_predictions_true[lmbda] = {"Time": [], "E": [], "I": [], "R": []}
    if lmbda not in all_predictions_partial:
        all_predictions_partial[lmbda] = {"Time": [], "E": [], "I": [], "R": []}

    # Load the PINN model
    pinn_forecast = pd.read_csv(os.path.join(run_dir, "pinn_forecast.csv"))
    E_pred = pinn_forecast["E_pred"]
    I_pred = pinn_forecast["I_pred"]
    R_pred = pinn_forecast["R_pred"]

    # Load predictions and parameter estimates
    preds_file = os.path.join(run_dir, "preds.csv")
    theta_file = os.path.join(run_dir, "theta_hats.csv")
    data_file = glob(os.path.join(data_dir, f"logSEIR_beta=*_alpha=0.15_seed={seed}.csv"))

    if not os.path.exists(preds_file) or not os.path.exists(theta_file) or len(data_file) != 1:
        print(f"Skipping run {foldername}: Missing necessary files")
        continue  # Skip if necessary files are missing or ground truth file is ambiguous

    # Load ground truth data
    data = pd.read_csv(data_file[0])

    # Load predictions
    preds = pd.read_csv(preds_file)

    observed_indices_in_true = np.isin(np.round(data['t'], 5), np.round(preds['t'], 5)).nonzero()[0]
    truth = data.loc[observed_indices_in_true]

    # Compute RMSE for compartments
    rmse_logE = np.sqrt(np.mean(((preds["E"]) - np.log(truth["E_true"])) ** 2))
    rmse_logI = np.sqrt(np.mean(((preds["I"]) - np.log(truth["I_true"])) ** 2))
    rmse_logR = np.sqrt(np.mean(((preds["R"]) - np.log(truth["R_true"])) ** 2))

    rmse_E = np.sqrt(np.mean((np.exp(preds["E"]) - (truth["E_true"])) ** 2))
    rmse_I = np.sqrt(np.mean((np.exp(preds["I"]) - (truth["I_true"])) ** 2))
    rmse_R = np.sqrt(np.mean((np.exp(preds["R"]) - (truth["R_true"])) ** 2))

    rmspe_E = np.sqrt(np.mean( (np.exp(preds["E"] - np.log(truth["E_true"])) - 1) ** 2))
    rmspe_I = np.sqrt(np.mean( (np.exp(preds["I"] - np.log(truth["I_true"])) - 1) ** 2))
    rmspe_R = np.sqrt(np.mean( (np.exp(preds["R"] - np.log(truth["R_true"])) - 1) ** 2))

    # Load parameter estimates and compute RMSE for parameters
    theta_hat = pd.read_csv(theta_file).iloc[0]
    beta_hat, gamma_hat, sigma_hat = theta_hat["beta"], theta_hat["gamma"], theta_hat["sigma"]
    beta_true, gamma_true, sigma_true = float(data_file[0].split("beta=")[1].split("_")[0]), \
                                        float(data_file[0].split("gamma=")[1].split("_")[0]), \
                                        float(data_file[0].split("sigma=")[1].split("_")[0])
    r0_true = beta_true / gamma_true
    param_rmse_beta = (beta_hat - beta_true)
    param_rmse_gamma = (gamma_hat - gamma_true)
    param_rmse_sigma = (sigma_hat - sigma_true)
    param_rmse_r0 = (beta_hat / gamma_hat - r0_true)

    forecast_I = pd.DataFrame(I_pred)
    forecast_I.index = t_pred.flatten()
    forecast_I = np.exp(forecast_I)
    peak_intensity = forecast_I.max(axis=0).item()
    peak_timing = forecast_I.idxmax(axis=0).item()

    error_peak_intensity = np.abs(peak_intensity - data['I_true'].max())
    error_peak_timing = np.abs(peak_timing - data['t'].iloc[data['I_true'].idxmax()])

    # Store results based on configuration
    result = {
        "run": foldername,
        "seed": seed,
        "rmse_logE": rmse_logE,
        "rmse_logI": rmse_logI,
        "rmse_logR": rmse_logR,
        "rmse_E": rmse_E,
        "rmse_I": rmse_I,
        "rmse_R": rmse_R,
        "rmspe_E": rmspe_E,
        "rmspe_I": rmspe_I,
        "rmspe_R": rmspe_R,
        "rmse_beta": param_rmse_beta,
        "rmse_gamma": param_rmse_gamma,
        "rmse_sigma": param_rmse_sigma,
        'param_rmse_r0': param_rmse_r0,
        'error_peak_intensity': error_peak_intensity,
        'error_peak_timing': error_peak_timing,
    }
    if config == "True_True_True":
        all_predictions_true[lmbda]["Time"].append(t_pred.flatten())
        all_predictions_true[lmbda]["E"].append(E_pred)
        all_predictions_true[lmbda]["I"].append(I_pred)
        all_predictions_true[lmbda]["R"].append(R_pred)
        summary["EIR_True"][lmbda].append(result)
    elif config == "False_True_True":
        all_predictions_partial[lmbda]["Time"].append(t_pred.flatten())
        all_predictions_partial[lmbda]["E"].append(E_pred)
        all_predictions_partial[lmbda]["I"].append(I_pred)
        all_predictions_partial[lmbda]["R"].append(R_pred)
        summary["EIR_Partial"][lmbda].append(result)

summary_dataframes_true = dict()
summary_dataframes_partial = dict()
for lmbda in summary["EIR_True"]:
    # Convert summary to DataFrame
    summary_dataframes_true[lmbda] = pd.DataFrame(summary["EIR_True"][lmbda]).describe()
    summary_dataframes_partial[lmbda] = pd.DataFrame(summary["EIR_Partial"][lmbda]).describe()

# Save the summary and predictions
import pickle
pinn_save_dict = {
    'summary': summary,
    'all_predictions_true': all_predictions_true,
    'all_predictions_partial': all_predictions_partial,
    'summary_dataframes_true': summary_dataframes_true,
    'summary_dataframes_partial': summary_dataframes_partial,
    "data": data,
}

with open("results/pinn_save_dict.pkl", "wb") as f:
    pickle.dump(pinn_save_dict, f)
