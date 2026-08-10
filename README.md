<p align="center"><img src="logo.png" align="left" alt="MOSFiT" width="300"/></p>
<a href="https://www.python.org"><img src="https://img.shields.io/badge/python-3.x-blue.svg" alt="Python Version"></a>
<a href="https://badge.fury.io/py/mosfit"><img src="https://badge.fury.io/py/mosfit.svg" alt="PyPI version"></a>
<a href="https://mosfit.readthedocs.io/en/latest/?badge=latest"><img src="https://readthedocs.org/projects/mosfit/badge/?version=latest" alt="Documentation Status"></a>
<a href="https://ascl.net/1710.006"><img src="https://img.shields.io/badge/ascl-1710.006-blue.svg?colorB=262255" alt="ascl:1710.006" /></a>

`MOSFiT` (**M**odular **O**pen-**S**ource **Fi**tter for **T**ransients) is a Python 3 package for fitting, sharing, and estimating the parameters of transients via user-contributed transient models. Data for a transient can be provided by the user in a wide range of formats (JSON, ASCII tables, CDS, LaTeX).<br clear="all">

## Installation

`MOSFiT` is available on `conda` and `pip`:

```bash
conda install -c conda-forge mosfit
```

or:

```bash
pip install mosfit
```

For a development install, clone the repository and install in editable mode:

```bash
git clone https://github.com/guillochon/MOSFiT.git
cd MOSFiT
pip install -e .
```

## Using MOSFiT

```bash
mosfit -e ./my_transient.json -m slsn
```

Fits write products under `products/` (including `walkers.h5`, and optionally `chain.h5` if run with `-c`). For detailed instructions, see the documentation on RTD: <https://mosfit.readthedocs.io/>
