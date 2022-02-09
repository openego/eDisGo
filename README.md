<img align="right" width="200" height="200" src="https://raw.githubusercontent.com/openego/eDisGo/dev/doc/images/edisgo_logo.png">


# Overview

[![Coverage Status](https://coveralls.io/repos/github/openego/eDisGo/badge.svg?branch=dev)](https://coveralls.io/github/openego/eDisGo?branch=dev)
[![Tests & coverage](https://github.com/openego/eDisGo/actions/workflows/tests-coverage.yml/badge.svg)](https://github.com/openego/eDisGo/actions/workflows/tests-coverage.yml)


# eDisGo

The python package eDisGo serves as a toolbox to evaluate flexibility measures
as an economic alternative to conventional grid expansion in
medium and low voltage grids.
See [documentation](https://edisgo.readthedocs.io/en/dev/) for further information.


# LICENSE

Copyright (C) 2017 Reiner Lemoine Institut gGmbH

This program is free software: you can redistribute it and/or modify it under
the terms of the GNU Affero General Public License as published by the Free
Software Foundation, either version 3 of the License, or (at your option) any
later version.

This program is distributed in the hope that it will be useful, but WITHOUT
ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
FOR A PARTICULAR PURPOSE. See the GNU Affero General Public License for more
details.

You should have received a copy of the GNU General Public License along with
this program. If not, see https://www.gnu.org/licenses/.


# Setting Up a Development Environment

eDisGo can be installed with pip. Please see the
[Installation Guide](https://edisgo.readthedocs.io/en/dev/quickstart.html#installation)
for more information. Installing eDisGo from source is only advised for development.


## Installation Using Linux

To set up a source installation using linux simply use a virtual environment and install
the source code with pip. Make sure to use `python3.7` or higher (recommended
`python3.8`). **After** setting up your virtual environment and activating it run the
following commands within your eDisGo directory:

```
pip install -e .[full]  # install eDisGo from source
pre-commit install  # install pre-commit hooks
```

That's it! You can now start developing.

## Installation Using Windows

Installation using Windows needs a bit of fiddling. Please use
[anaconda](https://www.anaconda.com/) and make sure your conda installation is
up-to-date by running

```
conda update -n base -c defaults conda
```

within your anaconda prompt first. After that create a virtual environment by either
using the provided
[`eDisGo_env.yml`](https://github.com/openego/eDisGo/blob/dev/eDisGo_env.yml) by running

```
conda env create -f path/to/eDisGo_env.yml
```

or independently setting up a virtual environment with conda and installing
[rasterio](https://github.com/conda-forge/rasterio-feedstock) and
[Fiona](https://github.com/conda-forge/fiona-feedstock) following the linked
installation guides. This may take a while to complete. Activate the newly created
environment with:

```
conda activate eDisGo_env  # or the name you gave your environment
```

Afterwards you are ready to install eDisGo by running the following commands within your
eDisGo directory:

```
pip install -e .[full]  # install eDisGo from source
pre-commit install  # install pre-commit hooks
```

## Installation Using macOS

We don't have any experience using macOS. You can help us by providing us information
about your experience using eDisGo with macOS!
