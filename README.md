# Are Statistical Methods Obsolete in the Era of Deep Learning? A Comparison of Neural Network and Statistical Methods in the Inference of Mechanistic Models
This repository accompanies the paper "Are Statistical Methods Obsolete in the Era of Deep Learning? A Comparison of Neural Network and Statistical Methods in the Inference of Mechanistic Models," submitted to the JASA Special Issue on Statistics in AI.

**A note on data:** code for generating test datasets for the Lorenz system can be found at `lorenz/data_generation.ipynb` and for the SEIR system at `seir/magi/seir_simulation.py`.

**A note on dependencies:** the only packages that must be installed to run all code in this repository are the following, with the versions we used explicitly indicated꞉ `numpy=1.23.5`, `scipy=1.10.1`, `sklearn=1.2.2`, `tensorflow=2.17.0`, `tf_keras=2.17.0`, `tensorflow_probability=0.24.0`, `pandas=1.4.1`, `matplotlib=3.6.2`, `IPython=8.7.0`, and `tqdm=4.64.1`.

**A note on compute:** most computations in this paper were run on the FASRC Cannon cluster supported by the FAS Division of Science Research Computing Group at Harvard University using single-core instances of Intel Sapphire Rapids and Intel Cascade Lake CPUs, with a SLURM scheduler. Despite our code's compatibility with GPUs via `TensorFlow` and `TensorFlow Probability`, all jobs were run on CPUs. We include some of the `(...)_driver.sh` files we used to manage jobs with the SLURM system as examples of our workflow, but please alter these files accordingly to your own systems and clusters.

**Core MAGI Class**: the core `TensorFlow Probability`-powered MAGI class can be found at `lorenz/magi_v2.py` and `seir/magi/magi_v2.py`. Please see the detailed comments there for granular implementation specifications. The `lorenz/lorenz_magi_insample.py` script may be a good starting point for understanding the basic functionality of the MAGI class.

**Lorenz Experiments**
1. The `src` folder contains the source code for the PINN's main class and helper modules. It is a slightly-modified version of van Herten (2020)'s GitHub [repository](https://github.com/cianmscannell/pinns/tree/main), which itself reproduces the Lorenz example from Lu (2019)'s [DeepXDE](https://deepxde.readthedocs.io/en/stable/) package.
2. The `data_generation.ipynb` notebook generates synthetic Lorenz datasets on both testbed regimes.
3. 3D and 2D trajectory figures can be generated via the `regime_visualization.ipynb` notebook. 
4. The `lorenz_magi_insample.py` script runs one MAGI in-sample experimental setting. All such settings can be executed via `sbatch lorenz_magi_insample_driver.sh`.
5. The `lorenz_magi_forecasting.py` script runs one MAGI forecasting experimental setting. All such settings can be executed via `sbatch lorenz_magi_forecasting_driver.sh`.
6. The `lorenz_pinn_insample.py` script runs one PINN in-sample experimental setting. All such settings can be executed via `sbatch lorenz_pinn_insample_driver.sh`.
7. The `lorenz_pinn_forecasting.py` script runs one PINN forecasting experimental setting. All such settings can be executed via `sbatch lorenz_pinn_forecasting_driver.sh`.
9. As examples of our workflow, we also include empty directories intended to store files such as raw results, SLURM errors/outputs, and camera-ready figures for ease of reproducibility.
10. The `logger.ipynb` notebook summarizes all experiments' resultant metrics on parameter inference, trajectory reconstruction, etc. and stores the resultant summaries as `.csv` files for ease of visualization. For transparency, the `logs` folder contains the `.csv` log files we used to generate our boxplots in the paper.
11. The `camera_ready.ipynb` notebook generates the results figures that are in our paper.
12. Finally, the `pinn_over_time.ipynb` generates the figure in our paper that demonstrates how PINNs truly need many epochs of training before convergence.

**SEIR Experiments**
1. The `pinn/src` folder contains the source code for the PINN's main class and helper modules. It is a slightly-modified version of van Herten (2020)'s GitHub [repository](https://github.com/cianmscannell/pinns/tree/main), which itself reproduces the Lorenz example from Lu (2019)'s [DeepXDE](https://deepxde.readthedocs.io/en/stable/) package.
2. The `magi/seir_simulation.py` file generates synthetic SEIR datasets.
3. For PINN experiments, `pinn/seir_v2_pinn_main.py` houses the main experimental file, with `pinn/job.sh` used for running jobs at scale on high-performance cluster compute.
4. For MAGI experiments, `magi/seir_log_eir.py` and `magi/seir_log_missing_e.py` house the main experimental files for SEIR with fully-observed components and SEIR with the $E$ component missing, respectively. The `magi/visualization.py` file provides helper functions for initial sanity-check visualizations of the aforementioned experiments.
5. Logging is performed via `pinn/summary.py` and `magi/evaluation.py`, and visualizations can be generated with `pinn/camera_ready_pinn.py` and `magi/generate_tab_fig.py`.

**License꞉** all assets in our paper and accompanying repository are under CC BY‑NC 4.0.

