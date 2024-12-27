# Are Statistical Methods Obsolete in the Era of Deep Learning?
This repository accompanies the paper "Are Statistical Methods Obsolete in the Era of Deep Learning?" submitted to the JASA Special Issue on Statistics in AI.

**A note on data:** code for generating test datasets for the Lorenz system can be found at `lorenz/data_generation.ipynb`.

**A note on dependencies:** the only packages that must be installed to run all code in this repository are the following, with the versions we used explicitly indicated꞉ `numpy=1.23.5`, `scipy=1.10.1`, `sklearn=1.2.2`, `tensorflow=2.17.0`, `tf_keras=2.17.0`, `tensorflow_probability=0.24.0`, `pandas=1.4.1`, `matplotlib=3.6.2`, `IPython=8.7.0`, and `tqdm=4.64.1`.

**A note on compute:** most computations in this paper were run on the FASRC Cannon cluster supported by the FAS Division of Science Research Computing Group at Harvard University using single-core instances of Intel Sapphire Rapids and Intel Cascade Lake CPUs, with a SLURM scheduler. Despite our code's compatibility with GPUs via `TensorFlow` and `TensorFlow Probability`, all jobs were run on CPUs. We include the `(...)_driver.sh` files we used to manage jobs with the SLURM system out of transparency, but please alter these files accordingly to your own systems and clusters.

**Core MAGI Class**: the core `TensorFlow Probability`-powered MAGI class can be found at `lorenz/magi_v2.py` and `seir/magi_v2.py`. Please see the detailed comments there for granular implementation specifications.

**License꞉** all assets in our paper and accompanying repository are under CC BY‑NC 4.0.

