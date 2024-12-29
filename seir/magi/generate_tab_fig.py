from tfpigp.evaluation import *

# True parameter values used in simulations
true_params = [6.0, 0.6, 1.8]  # [beta, gamma, sigma]

# Define observed time points based on simulation settings
d_obs = 20          # Observations per unit time
t_max = 2.0         # Observation interval length
num_observations = int(d_obs * t_max) + 1
observed_time_points = np.linspace(0, t_max, num_observations)

# Path to the directory containing all simulation result subdirectories
results_dir = "large run/"  # Replace with your actual path

# Generate the summary DataFrame
summary_df = summarize_simulation_results(results_dir, true_params, observed_time_points)

# Display the summary table
print("Summary of Simulation Results:")
print(summary_df)

summary_df_print = summary_df.copy()
summary_df_print.columns = [x.replace('_', ' ') for x in summary_df_print.columns]

generate_latex_table(summary_df_print, "summary_table.tex")
plot_df, true_forecast, ts_obs, example_observations = visualize_forecast_means(results_dir, observed_time_points, results_dir)
import pickle
save_dict = {
    'summary_df': summary_df,
    'plot_df': plot_df,
    'true_forecast': true_forecast,
    'ts_obs': ts_obs,
    'example_observations': example_observations
}
with open(f'{results_dir}/seir_magi_plot_data.pkl', 'wb') as f:
    pickle.dump(save_dict, f)


import matplotlib.pyplot as plt

# Assuming summary_df is already loaded as a DataFrame

# Generate 3-axis histogram plot for parameter estimation errors
fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)

parameters = ['Beta Error', 'Gamma Error', 'Sigma Error']
errors = [
    summary_df['Beta_Error'],
    summary_df['Gamma_Error'],
    summary_df['Sigma_Error']
]

for ax, param, error_data in zip(axes, parameters, errors):
    ax.hist(error_data, bins=20, alpha=0.75, edgecolor='black')
    ax.set_title(f"Histogram of {param}")
    ax.set_xlabel('Error')
    ax.set_ylabel('Frequency')
    ax.grid(axis='y')

plt.tight_layout()
plt.savefig(f'{results_dir}/parameter_error_histograms.png')
plt.show()

summary_df.iloc[:, 1:].describe()
summary_df.sort_values(by='Beta_Error', ascending=True, inplace=True)
summary_df


import os
import pickle
all_posterior_means = []
results_dir = "fully_observed/"  # Replace with your actual path
# Iterate through each simulation directory
for sim_dir in os.listdir(results_dir):
    sim_path = os.path.join(results_dir, sim_dir)
    if os.path.isdir(sim_path):
        pickle_file = os.path.join(sim_path, "simulation_results.pkl")
        if os.path.exists(pickle_file):
            try:
                with open(pickle_file, "rb") as f:
                    data = pickle.load(f)
            except Exception as e:
                print(f"Error loading {pickle_file}: {e}")
                continue

            results_forecast = data["results_forecast"]
            X_samps = results_forecast["X_samps"]  # Shape (num_samples, T_forecast, D)

            # Compute posterior mean for this dataset
            posterior_mean = X_samps.mean(axis=0)  # Shape (T_forecast, D)
            all_posterior_means.append(posterior_mean)

with open(f'{results_dir}/all_posterior_means.pkl', 'wb') as f:
    pickle.dump(all_posterior_means, f)

ts_forecast = data["results_forecast"]["I"].flatten()  # Shape (T_forecast,)
