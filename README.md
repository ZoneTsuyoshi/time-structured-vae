# time-structured-vae
This repository includes deep-learning methods for obtaining slow collective variables (CVs) for biomolecules.
The included methods are following:

|Method|Reference|
|:---|:---|
|TAE|Wehmeyer, C.; Noé, F. Time-lagged autoencoders: Deep learning of slow collective variables for molecular kinetics. [J. Chem. Phys.](https://aip.scitation.org/doi/10.1063/1.5011399) 2018, 148, 241703.|
|TVAE|Hoffmann, M.; Scherer, M. K.; Hempel, T.; Mardt, A.; de Silva, B.; Husic, B. E.; Klus, S.; Wu, H.; Kutz, J. N.; Brunton, S.; Noé, F. Deeptime: a Python library for machine learning dynamical models from time series data. [Machine Learning: Science and Technology](https://iopscience.iop.org/article/10.1088/2632-2153/ac3de0) 2021.|
|VDE|Hernándeza, C. X.; Wayment-Steele, H. K.; Sultan, M. M.; Husic, B. E.; Pande, V. S. Variational encoding of complex dynamics. [Phys. Rev. E](https://journals.aps.org/pre/abstract/10.1103/PhysRevE.97.062412) 2018, 97, 062412.|
|SRV|Chen, W.; Sidky, H.; Ferguson, A. L. Nonlinear Discovery of Slow Molecular Modes using State-Free Reversible VAMPnets [J. Chem. Phys.](https://pubs.aip.org/aip/jcp/article/150/21/214114/197931/Nonlinear-discovery-of-slow-molecular-modes-using) 2019, 150, 214114.|
|tsVAE (ours)|Ishizone, T.; Matsunaga, Y.; Fuchigami, S.; Nakamura, K. Representation of Protein Dynamics Disentangled by Time-Structure-Based Prior. [J. Chem. Theory Comput.](https://pubs.acs.org/doi/full/10.1021/acs.jctc.3c01025) 2023, 20, 1, 436--450|
|tsTVAE (ours)|Ishizone, T.; Matsunaga, Y.; Fuchigami, S.; Nakamura, K. Representation of Protein Dynamics Disentangled by Time-Structure-Based Prior. [J. Chem. Theory Comput.](https://pubs.acs.org/doi/full/10.1021/acs.jctc.3c01025) 2023, 20, 1, 436--450|


## Requirement
We use following environment:
- numpy 1.19.5
- msmbuilder 3.8.0
- torch 1.10.1
- mdshare 0.4.2 (optional for fetching data)
- pyemma 2.5.7 (optional for MSM analysis)
- matplotlib 3.3.4 (optional for graphics)
- scikit-learn 0.18 (optional for preprocessing)
- scipy 1.2.1 (used in SRV, tsVAE, and tsTVAE)


## Usage
You can use deep-learning methods by

```python

import models
m = models.tsTVAE(input_dim=30, lagtime=1, n_epochs=10, latent_dim=2)
embed = m.fit_transform(data)
```

The example of the application is shown in `example.ipynb`.


## Citation
If you use this code in your work, please cite:

```
@article{tsvae,
  title = {Representation of Protein Dynamics Disentangled by Time-Structure-Based Prior},
  author = {Ishizone, Tsuyoshi and Matsunaga, Yasuhiro and Fuchigami, Sotaro and Nakamura, Kazuyuki},
  journal = {J. Chem. Theory Comput.},
  volume = {20},
  issue = {1},
  pages = {436--450},
  numpages = {15},
  year = {2024},
  month = {1},
  publisher = {American Chemical Society},
  url = {https://pubs.acs.org/doi/full/10.1021/acs.jctc.3c01025}
}
```