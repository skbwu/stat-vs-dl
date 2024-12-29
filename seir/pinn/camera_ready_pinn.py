import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

################################################################################
# MAIN PLOTTING FUNCTION
################################################################################
def plot_seir_trajectories(
        all_predictions_dict,
        all_magi_posterior_means,
        data,
        out_png="example_pinn_trajectories.png",
        title_prefix="SEIR PINN (True) "
):
    """
    all_predictions_dict is something like all_predictions_true[lmbda],
    i.e. a dictionary of form:
        all_predictions_dict[lmbda]["Time"] -> list of arrays (one per seed)
        all_predictions_dict[lmbda]["E"]    -> list of arrays (one per seed)
        all_predictions_dict[lmbda]["I"]    -> ...
        all_predictions_dict[lmbda]["R"]    -> ...
    """

    # 1) figure out which λ values are available
    lambdas = sorted(all_predictions_dict.keys())

    # 2) We will do: 1 row for the "Fake MAGI" sine wave, plus 1 row for each λ
    nrows = 1 + len(lambdas)  # top row is the sine wave
    ncols = 3  # columns: E, I, R

    fig, ax = plt.subplots(nrows, ncols, dpi=150, figsize=(14, 2.5 * nrows), sharex=True)

    # If there's only one row, `ax` might be 1D. Force 2D indexing for convenience:
    if nrows == 1:
        ax = np.array([ax])  # shape (1, 3)

    ############################################################################
    # PLOT THE “FAKE MAGI” SINE-WAVE ROW (row = 0)
    ############################################################################
    # generate some “fake ground truth” for reference
    row_idx_magi = 0
    truth = data.query('t <= 4')

    example_observations = data[['t', 'E_obs', 'I_obs', 'R_obs']]
    ts_obs = np.linspace(0, 2, 41)
    observed_indices_in_true = np.isin(np.round(example_observations['t'], 5), np.round(ts_obs, 5)).nonzero()[0]
    example_observations = example_observations.loc[observed_indices_in_true]

    # plot black ground-truth lines + highlight in yellow
    for j, var in enumerate(["E_true", "I_true", "R_true"]):
        ax[0, j].plot(truth["t"], truth[var], color="black", linewidth=2.5, zorder=2000)
        ax[0, j].plot(truth["t"], truth[var],
                      color="yellow", linewidth=5.0, alpha=0.5)
        # label & grid
        ax[0, j].grid(True)
        ax[0, j].tick_params("both", labelsize=10)
        if j == 0:
            ax[0, j].set_ylabel("MAGI", fontsize=12)

    # Plot each seed in thin blue lines
    T_forecast = all_magi_posterior_means[0].shape[0]
    # e.g. maybe your time range is 0..4 with T_forecast points:
    t_magi = np.linspace(0, 4, T_forecast)
    all_magi = []
    for seed_idx, posterior_mean in enumerate(all_magi_posterior_means):
        # posterior_mean is shape (T_forecast, D=3). Suppose [E_col, I_col, R_col]
        E_col = np.exp(posterior_mean[:, 0])
        I_col = np.exp(posterior_mean[:, 1])
        R_col = np.exp(posterior_mean[:, 2])

        ax[row_idx_magi, 0].plot(t_magi, E_col, color="blue", linewidth=0.5, alpha=1.0)
        ax[row_idx_magi, 1].plot(t_magi, I_col, color="blue", linewidth=0.5, alpha=1.0)
        ax[row_idx_magi, 2].plot(t_magi, R_col, color="blue", linewidth=0.5, alpha=1.0)

        all_magi.append(posterior_mean)  # for later averaging

    # Then plot the mean across seeds in thick, partially transparent line
    magi_array = np.stack(all_magi, axis=0)  # shape: (N_runs, T_forecast, 3)
    magi_mean = magi_array.mean(axis=0)  # shape: (T_forecast, 3)
    magi_mean = np.exp(magi_mean)
    ax[row_idx_magi, 0].plot(t_magi, magi_mean[:, 0], color="blue", linewidth=3.0, alpha=0.3)
    ax[row_idx_magi, 1].plot(t_magi, magi_mean[:, 1], color="blue", linewidth=3.0, alpha=0.3)
    ax[row_idx_magi, 2].plot(t_magi, magi_mean[:, 2], color="blue", linewidth=3.0, alpha=0.3)

    for idx in range(3):
        ax[row_idx_magi, idx].scatter(example_observations['t'], example_observations.iloc[:, 1+idx], color="orange",
                                      marker="o", s=50, edgecolors='k', label="Example Observations", zorder=3)

    # (You could also do multiple seeds’ worth of "fake MAGI" if you like,
    #  or a "mean" highlight, etc.)

    ############################################################################
    # LOOP OVER LAMBDAS (PINN ROWS), THEN WITHIN EACH ROW: loop seeds, plot
    ############################################################################
    for i, lmbda in enumerate(lambdas):
        row_idx = i + 1  # because row 0 was the sine wave
        # gather the list-of-lists for Time, E, I, R
        times = all_predictions_dict[lmbda]["Time"]  # list of arrays
        E_list = all_predictions_dict[lmbda]["E"]
        I_list = all_predictions_dict[lmbda]["I"]
        R_list = all_predictions_dict[lmbda]["R"]

        # we can also attempt a “ground truth” if you have it
        # but here I’ll just re-use the “fake_truth” for illustration
        for j, var in enumerate(["E_true", "I_true", "R_true"]):
            ax[row_idx, j].plot(truth["t"], truth[var], color="black", linewidth=2.5, zorder=2000)
            ax[row_idx, j].plot(truth["t"], truth[var],
                                color="yellow", linewidth=5.0, alpha=0.5)
            ax[row_idx, j].grid(True)
            ax[row_idx, j].tick_params("both", labelsize=10)
            if j == 0:
                ax[row_idx, j].set_ylabel(f"PINN ($\\lambda={lmbda}$)", fontsize=12)

        # For each seed, we have times[k], E_list[k], etc.
        # Plot them in thin red lines
        # Then at the end we can plot a "mean" highlight in thicker red
        # (If you want to do the mean, you must collect them in e.g. a Python list
        #  then do an average.)

        n_seeds = len(times)
        allE = []
        allI = []
        allR = []
        # -- loop seeds
        for k in range(n_seeds):
            t_k = times[k]
            E_k = np.exp(E_list[k])
            I_k = np.exp(I_list[k])
            R_k = np.exp(R_list[k])

            # “thin red line”
            ax[row_idx, 0].plot(t_k, E_k, color="red", linewidth=0.5, alpha=0.8)
            ax[row_idx, 1].plot(t_k, I_k, color="red", linewidth=0.5, alpha=0.8)
            ax[row_idx, 2].plot(t_k, R_k, color="red", linewidth=0.5, alpha=0.8)

            allE.append(E_k)
            allI.append(I_k)
            allR.append(R_k)

        # after all seeds, compute mean
        meanE = np.mean(allE, axis=0)  # shape (#points,)
        meanI = np.mean(allI, axis=0)
        meanR = np.mean(allR, axis=0)

        # highlight with thicker, lighter red
        ax[row_idx, 0].plot(times[0], meanE, color="red", linewidth=3.0, alpha=0.3)
        ax[row_idx, 1].plot(times[0], meanI, color="red", linewidth=3.0, alpha=0.3)
        ax[row_idx, 2].plot(times[0], meanR, color="red", linewidth=3.0, alpha=0.3)

        for idx in range(3):
            ax[row_idx, idx].scatter(example_observations['t'], example_observations.iloc[:, 1 + idx], color="orange",
                                     marker="o", s=50, edgecolors='k', label="Example Observations", zorder=3)

    # final touches: column titles, a global suptitle, a legend, etc.
    # column titles
    col_names = ["E(t)", "I(t)", "R(t)"]
    for j in range(ncols):
        ax[0, j].set_title(col_names[j], fontsize=14)

    # global title
    plt.suptitle(f"{title_prefix}: MAGI + PINN Results", fontsize=16, y=1.01)

    custom_lines = [
        Line2D([0], [0], color="black", linewidth=1.0, label="Ground Truth"),
        Line2D([0], [0], color="yellow", linewidth=5, alpha=0.5, label="Truth (Emphasis)"),
        Line2D([0], [0], color="blue", linewidth=1.0, label="MAGI (per seed)"),
        Line2D([0], [0], color="red", linewidth=0.5, label="PINN (per seed)"),
        Line2D([0], [0], color="red", linewidth=3.0, alpha=0.3, label="PINN Mean (all seeds)"),

        # Here is the legend handle for your orange scatter points:
        Line2D(
            [0], [0],
            marker='o',  # draw a circle marker
            color='white',  # no connecting line
            markerfacecolor='orange',  # fill color
            markeredgecolor='k',  # black edge
            markersize=7,  # size of marker
            linestyle='None',  # no line
            label='Example Observations'  # legend label
        )
    ]

    fig.legend(
        handles=custom_lines,
        loc="lower center",
        ncol=3,
        fontsize=10,
        bbox_to_anchor=(0.5, -0.05)
    )
    plt.tight_layout()
    plt.savefig(out_png, facecolor="white", bbox_inches="tight")
    plt.close()

###############################################################################
# EXAMPLE USAGE:
# Suppose you already have your dictionary all_predictions_true which is of form:
#     all_predictions_true[lmbda]["Time"] -> list of length (# seeds), each an array of shape (#timepoints,)
#     all_predictions_true[lmbda]["E"]    -> list of length (# seeds), each an array of shape (#timepoints,)
#     ... etc ...
#
# Just call:
import pickle
magi_pickle_path = "fully_observed/all_posterior_means.pkl"
with open(magi_pickle_path, "rb") as f:
    all_magi_posterior_means_full = pickle.load(f)

with open("results/pinn_save_dict.pkl", "rb") as f:
    pinn_save_dict = pickle.load(f)
all_predictions_true = pinn_save_dict["all_predictions_true"]
data = pinn_save_dict["data"]

plot_seir_trajectories(all_predictions_true, all_magi_posterior_means_full, data,
                       out_png="seir_pinn_full_plot.png",
                       title_prefix="SEIR (True)")

pinn_subset={
    10.0: all_predictions_true[10.0],
    1000.0: all_predictions_true[1000.0]
}
plot_seir_trajectories(pinn_subset, all_magi_posterior_means_full, data,
                       out_png="seir_pinn_full_subset.png",
                       title_prefix="SEIR (Fully Observed)")


###############################################################################
import pickle
magi_pickle_path = "large run/all_posterior_means.pkl"
with open(magi_pickle_path, "rb") as f:
    all_magi_posterior_means_parital = pickle.load(f)

all_predictions_partial = pinn_save_dict["all_predictions_partial"]
data = pinn_save_dict["data"]
data['E_obs'] = np.nan

plot_seir_trajectories(all_predictions_partial, all_magi_posterior_means_parital, data,
                       out_png="seir_pinn_partial_plot.png",
                       title_prefix="SEIR (Partially Observed)")

pinn_subset={
    10.0: all_predictions_partial[10.0],
    1000.0: all_predictions_partial[1000.0]
}
plot_seir_trajectories(pinn_subset, all_magi_posterior_means_parital, data,
                       out_png="seir_pinn_partial_subset.png",
                       title_prefix="SEIR (Partially Observed)")


def boxplot_compartment_errors_magi_df(
        magi_df,  # a pandas DataFrame with columns: "RMSE_logE", "RMSE_logI", "RMSE_logR"
        pinn_summary,  # dict of lists: pinn_summary[lmbda] -> [ { "rmse_logE": ..., "rmse_logI": ..., ...}, ...]
        out_png="compartment_boxplot.png",
        title_str="RMSE in log-Compartment Values",
        use_log_scale=False  # <-- NEW FLAG: if True, set y-axis to log scale
):
    """
    Creates boxplots for RMSE in log(E, I, R):
      - 'RMSE_logE'  => comparing with 'rmse_logE' in the PINN summary
      - 'RMSE_logI'  => comparing with 'rmse_logI'
      - 'RMSE_logR'  => comparing with 'rmse_logR'

    The MAGI data come from the DataFrame columns.
    The PINN data come from the dictionary-of-lists, keyed by lambda.

    If use_log_scale=True, we do ax.set_yscale("log") for each subplot
    so that the boxplots appear on a logarithmic y-scale.
    """

    # We'll make 1 row by 3 columns for "RMSE_logE", "RMSE_logI", "RMSE_logR"
    compartments = ["RMSE_logE", "RMSE_logI", "RMSE_logR"]
    fig, axs = plt.subplots(1, 3, figsize=(12, 4), dpi=150)

    # In case of a single subplot, wrap in a list
    if len(compartments) == 1:
        axs = [axs]

    # Sort lambdas for consistent ordering
    sorted_lambdas = sorted(pinn_summary.keys())

    for idx, colname in enumerate(compartments):
        ax = axs[idx]
        ax.grid(True, alpha=0.3)

        # We'll gather data for boxplot in a list-of-lists
        data_for_boxplot = []
        x_positions = []

        # 1) MAGI values come from the DataFrame column
        magi_vals = magi_df[colname].dropna().values.tolist()
        data_for_boxplot.append(magi_vals)
        x_positions.append(0)

        # 2) PINN values come from summary
        #    We map "RMSE_logE" => "rmse_logE", etc.
        for i, lmbda in enumerate(sorted_lambdas):
            this_lambda_vals = []
            for run_dict in pinn_summary[lmbda]:
                # Map the DF column to the correct PINN key
                if colname == "RMSE_logE":
                    key_pinn = "rmse_logE"
                elif colname == "RMSE_logI":
                    key_pinn = "rmse_logI"
                elif colname == "RMSE_logR":
                    key_pinn = "rmse_logR"
                else:
                    key_pinn = None

                if key_pinn in run_dict:
                    this_lambda_vals.append(run_dict[key_pinn])

            data_for_boxplot.append(this_lambda_vals)
            x_positions.append(i + 1)

        # Create the boxplot (with patch_artist so we can style the boxes)
        box = ax.boxplot(
            data_for_boxplot,
            positions=x_positions,
            widths=0.6,
            patch_artist=True  # so we can remove fill
        )

        # ------------------
        # Remove the fill
        # ------------------
        # For each box, set facecolor='none'
        # Then color the edges (blue or red).
        # The first box is MAGI => edge = blue, others => red.
        #
        # Also, color whiskers and medians accordingly.
        #
        # NOTE: The lists in `box` go in the same order as data_for_boxplot.
        #       i_box = 0 => MAGI
        #       i_box = 1.. => PINN
        # ------------------

        # The "boxes", "whiskers", and "medians" each have
        # count = # of data sets, # of data sets*2, # of data sets, respectively.
        n_sets = len(data_for_boxplot)  # e.g. 1 (MAGI) + #lambdas
        for i_box in range(n_sets):
            # remove fill
            box["boxes"][i_box].set(facecolor="none")

            # set line width for the box outline
            box["boxes"][i_box].set_linewidth(1.5)

            # Whiskers for this i_box are 2*i_box and 2*i_box+1
            w1 = 2 * i_box
            w2 = 2 * i_box + 1

            # If i_box == 0 => MAGI (blue), else => PINN (red)
            if i_box == 0:
                box["boxes"][i_box].set_edgecolor("blue")
                if w1 < len(box["whiskers"]):
                    box["whiskers"][w1].set_color("blue")
                    box["whiskers"][w1].set_linewidth(1.5)
                if w2 < len(box["whiskers"]):
                    box["whiskers"][w2].set_color("blue")
                    box["whiskers"][w2].set_linewidth(1.5)
                box["medians"][i_box].set_color("blue")
                box["medians"][i_box].set_linewidth(1.5)
            else:
                box["boxes"][i_box].set_edgecolor("red")
                if w1 < len(box["whiskers"]):
                    box["whiskers"][w1].set_color("red")
                    box["whiskers"][w1].set_linewidth(1.5)
                if w2 < len(box["whiskers"]):
                    box["whiskers"][w2].set_color("red")
                    box["whiskers"][w2].set_linewidth(1.5)
                box["medians"][i_box].set_color("red")
                box["medians"][i_box].set_linewidth(1.5)

        # Label x-ticks
        ax.set_xticks(x_positions)
        ax.set_xticklabels(["MAGI"] + [f"PINN\n({l})" for l in sorted_lambdas], fontsize=9)

        # If log scale:
        if use_log_scale:
            ax.set_yscale("log")

        ax.set_title(colname, fontsize=10)

    fig.suptitle(title_str, fontsize=12)
    plt.tight_layout()
    plt.savefig(out_png, bbox_inches="tight", facecolor="white")
    plt.close()

pinn_save_dict['summary']["EIR_True"]

with open(f'fully_observed/seir_magi_plot_data.pkl', 'rb') as f:
    magi_dict_full = pickle.load(f)

with open(f'large run/seir_magi_plot_data.pkl', 'rb') as f:
    magi_dict_partial = pickle.load(f)

boxplot_compartment_errors_magi_df(
    magi_df=magi_dict_full['summary_df'],
    pinn_summary=pinn_save_dict['summary']["EIR_True"],
    out_png="boxplot_seir_traj_err_full_logscale.png",
    title_str="RMSE in log-Compartment Values (Fully Observed Case)",
    use_log_scale=True
)

boxplot_compartment_errors_magi_df(
    magi_df=magi_dict_partial['summary_df'],
    pinn_summary=pinn_save_dict['summary']["EIR_Partial"],
    out_png="boxplot_seir_traj_err_partial_logscale.png",
    title_str="RMSE in log-Compartment Values (Missing E Component)",
    use_log_scale=True
)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def boxplot_parameter_errors_magi_df(
        magi_df,
        pinn_summary,
        out_png="param_boxplot.png",
        title_str="Parameter Errors",
        use_log_scale=False
):
    """
    Creates boxplots for parameter errors from MAGI vs. PINN, with:
      - absolute value of the errors,
      - no fill color (just edges),
      - optional log-scale y-axis.

    Additionally returns a DataFrame containing mean and std of absolute errors
    for each parameter and method (MAGI, PINN with each lambda).

    Parameters
    ----------
    magi_df : pd.DataFrame
        Must have columns like:
           "Beta_Error", "Gamma_Error", "Sigma_Error", "R0_Error",
           "Peak_Timing_Error", "Peak_Intensity_Error", etc.
        These contain *signed* errors, but we will take absolute values inside this function.
    pinn_summary : dict
        A dict keyed by lambda -> list of dicts, each dict having
        keys like "rmse_beta", "rmse_gamma", "rmse_sigma", "param_rmse_r0",
        "error_peak_timing", "error_peak_intensity", etc.
    out_png : str
        Filename for the saved figure.
    title_str : str
        The figure suptitle.
    use_log_scale : bool
        If True, sets ax.set_yscale("log") for each subplot.

    Returns
    -------
    pd.DataFrame
        A DataFrame with columns ["Parameter", "Method", "Mean", "Std"]
        containing the mean and std of absolute errors across runs.
    """

    # Example set of param columns from your DataFrame
    param_cols = [
        "Beta_Error",
        "Gamma_Error",
        "Sigma_Error",
        "R0_Error",
        "Peak_Timing_Error",
        "Peak_Intensity_Error"
    ]

    nrows, ncols = 2, 3  # 2x3 grid for 6 parameters
    fig, axs = plt.subplots(nrows, ncols, figsize=(12, 6), dpi=150)
    axs = axs.flatten()

    sorted_lambdas = sorted(pinn_summary.keys())

    # We'll accumulate mean+std stats in a list, then convert to DataFrame at the end.
    stats_rows = []

    for idx, pcol in enumerate(param_cols):
        ax = axs[idx]
        ax.grid(True, alpha=0.3)

        # Build data for boxplot
        data_for_boxplot = []
        x_positions = []

        # -------------------------------
        # 1) MAGI data
        # -------------------------------
        magi_vals = np.abs(magi_df[pcol]).dropna().values.tolist()
        data_for_boxplot.append(magi_vals)
        x_positions.append(0)

        # Compute mean/std
        if len(magi_vals) > 0:
            magi_mean = np.mean(magi_vals)
            magi_std = np.std(magi_vals)
        else:
            magi_mean = np.nan
            magi_std = np.nan

        stats_rows.append({
            "Parameter": pcol,
            "Method": "MAGI",
            "Mean": magi_mean,
            "Std": magi_std
        })

        # -------------------------------
        # 2) PINN data, for each lambda
        # -------------------------------
        for i, lmbda in enumerate(sorted_lambdas):
            # Map DF column to the correct PINN dictionary key
            if pcol == "Beta_Error":
                key_pinn = "rmse_beta"
            elif pcol == "Gamma_Error":
                key_pinn = "rmse_gamma"
            elif pcol == "Sigma_Error":
                key_pinn = "rmse_sigma"
            elif pcol == "R0_Error":
                key_pinn = "param_rmse_r0"
            elif pcol == "Peak_Timing_Error":
                key_pinn = "error_peak_timing"
            elif pcol == "Peak_Intensity_Error":
                key_pinn = "error_peak_intensity"
            else:
                key_pinn = None

            pinn_vals = []
            for run_dict in pinn_summary[lmbda]:
                if key_pinn in run_dict:
                    pinn_vals.append(np.abs(run_dict[key_pinn]))

            data_for_boxplot.append(pinn_vals)
            x_positions.append(i + 1)

            # compute mean/std
            if len(pinn_vals) > 0:
                pinn_mean = np.mean(pinn_vals)
                pinn_std = np.std(pinn_vals)
            else:
                pinn_mean = np.nan
                pinn_std = np.nan

            stats_rows.append({
                "Parameter": pcol,
                "Method": f"PINN(lambda={lmbda})",
                "Mean": pinn_mean,
                "Std": pinn_std
            })

        # -------------------------------
        # Create the boxplot
        # -------------------------------
        box = ax.boxplot(data_for_boxplot, positions=x_positions, widths=0.6, patch_artist=True)

        # Remove fill and color edges
        n_sets = len(data_for_boxplot)  # 1 (MAGI) + (# of lambdas)
        for i_box in range(n_sets):
            box["boxes"][i_box].set(facecolor="none")
            box["boxes"][i_box].set_linewidth(1.5)

            # Each box has 2 whiskers at indices 2*i_box, 2*i_box+1
            w1 = 2 * i_box
            w2 = 2 * i_box + 1

            if i_box == 0:
                # MAGI => Blue
                box["boxes"][i_box].set_edgecolor("blue")
                if w1 < len(box["whiskers"]):
                    box["whiskers"][w1].set_color("blue")
                    box["whiskers"][w1].set_linewidth(1.5)
                if w2 < len(box["whiskers"]):
                    box["whiskers"][w2].set_color("blue")
                    box["whiskers"][w2].set_linewidth(1.5)
                box["medians"][i_box].set_color("blue")
                box["medians"][i_box].set_linewidth(1.5)
            else:
                # PINN => Red
                box["boxes"][i_box].set_edgecolor("red")
                if w1 < len(box["whiskers"]):
                    box["whiskers"][w1].set_color("red")
                    box["whiskers"][w1].set_linewidth(1.5)
                if w2 < len(box["whiskers"]):
                    box["whiskers"][w2].set_color("red")
                    box["whiskers"][w2].set_linewidth(1.5)
                box["medians"][i_box].set_color("red")
                box["medians"][i_box].set_linewidth(1.5)

        # Label x-ticks
        ax.set_xticks(x_positions)
        ax.set_xticklabels(["MAGI"] + [str(l) for l in sorted_lambdas], fontsize=9)

        # If log scale => log y-axis
        if use_log_scale:
            ax.set_yscale("log")

        ax.set_title(pcol, fontsize=10)

    fig.suptitle(title_str, fontsize=12)
    plt.tight_layout()
    plt.savefig(out_png, bbox_inches="tight", facecolor="white")
    plt.close()

    # -------------------------------
    # Build and return DataFrame of stats
    # -------------------------------
    df_stats = pd.DataFrame(stats_rows, columns=["Parameter", "Method", "Mean", "Std"])
    return df_stats

boxplot_parameter_errors_magi_df(
    magi_df=magi_dict_full['summary_df'],
    pinn_summary=pinn_save_dict['summary']["EIR_True"],
    out_png="boxplot_param_errors_full.png",
    title_str="Parameter Errors (Fully Observed Case)",
    use_log_scale=True  # or False, as you wish
)

boxplot_parameter_errors_magi_df(
    magi_df=magi_dict_partial['summary_df'],
    pinn_summary=pinn_save_dict['summary']["EIR_Partial"],
    out_png="boxplot_param_errors_partial.png",
    title_str="Parameter Errors (Missing E Component)",
    use_log_scale=True  # or False, as you wish
)
