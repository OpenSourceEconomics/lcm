# Life Cycle Models

[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![pre-commit.ci status](https://results.pre-commit.ci/badge/github/OpenSourceEconomics/lcm/main.svg)](https://results.pre-commit.ci/latest/github/OpenSourceEconomics/lcm/main)
[![image](https://codecov.io/gh/OpenSourceEconomics/lcm/branch/main/graph/badge.svg)](https://codecov.io/gh/OpenSourceEconomics/lcm)

This package aims to generalize and facilitate the specification, solution, and
simulation of discrete-continuous dynamic choice models.

## Installation

`LCM` currently needs to be installed from GitHub and requires `jax` and `jaxlib`.

> [!IMPORTANT]
> We recommend installing at least version "0.4.34" of `jax` and `jaxlib`, due to memory
> issues with previous versions.

If you aim to run `LCM` on a GPU, you need to install `jaxlib` with CUDA support (for
Linux) or with support for AMD GPUs / ARM-based Silicon GPUs (for MacOS). In any case,
for installation of `jax` and `jaxlib`, please consult the `jax`
[docs](https://jax.readthedocs.io/en/latest/installation.html#supported-platforms).

> [!NOTE]
> We currently test GPU support for `LCM` only on Linux with CUDA 12.

## License

This project is licensed under the Apache License, Version 2.0. See the
[LICENSE](LICENSE) file for details.

Copyright (c) 2023 The LCM Authors
